# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import readline  # type: ignore # noqa
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import fire
import model_qwen as fast
import torch
from stats import Stats
from transformers import AutoTokenizer
import sample_utils
from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalWithOffsetPaddedKeysMask as AttnBias,
)
from tqdm import tqdm
import numpy as np
# import pydevd_pycharm
# pydevd_pycharm.settrace('10.238.129.171', port=12312, stdout_to_server=True, stderr_to_server=True, suspend=False)

@dataclass
class GenArgs:
    gen_length: int = 32
    gen_bsz: int = 1
    prompt_length: int = 64

    use_sampling: bool = False
    temperature: float = 0.8
    top_p: float = 0.9


class FastGen:
    GRAPH_WARMUPS: int = 1
    tokenizer: AutoTokenizer

    @staticmethod
    def build(
        ckpt_dir: str,
        gen_args: GenArgs,
        device: Union[torch.device, str],
        tokenizer_path: Optional[str] = None,
        num_layers: int = 13,
        use_full_vocab: bool = False,
        use_kernel: bool = False,
    ) -> "FastGen":
        start_time = time.time()

        model_args_prefill = fast.ModelArgs(use_kernel=False)
        model_args_decode_bf16 = fast.ModelArgs(use_kernel=False)
        model_args_decode_int2 = fast.ModelArgs(use_kernel=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        torch.set_default_device(device)
        torch.set_default_dtype(torch.bfloat16)

        prefill_model = fast.Transformer(model_args_prefill)
        decode_model_bf16 = fast.Transformer(model_args_decode_bf16)
        decode_model_int2 = fast.Transformer(model_args_decode_int2)

        fp16_ckpt_path = str(Path(ckpt_dir) / "model_state_fp16.pt")
        fp16_checkpoint = torch.load(fp16_ckpt_path, map_location="cpu")
        int2_ckpt_path = str(Path(ckpt_dir) / "model_state_int2.pt")
        int2_checkpoint = torch.load(int2_ckpt_path, map_location="cpu")
        prefill_model.load_state_dict(fp16_checkpoint, strict=True)
        decode_model_bf16.load_state_dict(fp16_checkpoint, strict=True)
        decode_model_int2.load_state_dict(int2_checkpoint, strict=True)

        torch.cuda.synchronize()
        print(f"loaded model in {time.time() - start_time:.2f} seconds")
        start_time = time.time()

        return FastGen(gen_args, model_args_prefill, prefill_model, decode_model_bf16, decode_model_int2, tokenizer)

    def __init__(
        self,
        args: GenArgs,
        model_args: fast.ModelArgs,
        prefill_model: fast.Transformer,
        decode_model_bf16: fast.Transformer,
        decode_model_int2: fast.Transformer,
        tokenizer: AutoTokenizer,
    ):
        self.gen_args = args
        self.max_seq_length = args.prompt_length + args.gen_length
        self.model_args = model_args
        self.prefill_model = prefill_model
        self.decode_model_bf16 = decode_model_bf16
        self.decode_model_int2 = decode_model_int2
        self.tokenizer = tokenizer
        self._prefill_cuda_graph, self._prefill_compile_model, self._prefill_inputs, self._prefill_logits = None, None, None, None
        self._generate_cuda_graph, self._generate_compile_model, self._generate_inputs, self._generate_logits = None, None, None, None
        self._cache = None
        start_time = time.time()
        self._prefill_compile_model = self.compile_prefill()
        print(f"compiled model in {time.time() - start_time:.2f} seconds")

    def compile_prefill(self):

        if self._cache is None:
            self._cache = fast.make_cache(
                args=self.model_args,
                length=self.gen_args.gen_bsz * self.max_seq_length,
            )

        seq_lens = [self.gen_args.prompt_length for _ in range(self.gen_args.gen_bsz)]

        bias = AttnBias.from_seqlens(
            q_seqlen=seq_lens,
            kv_seqlen=seq_lens,
            kv_padding=self.max_seq_length,
        )
        bias.q_seqinfo.to("cuda")
        bias.k_seqinfo.to("cuda")

        tokens = torch.IntTensor([1] * self.gen_args.gen_bsz * self.gen_args.prompt_length).cuda()
        self._prefill_inputs = (tokens, bias)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(s):
            _ = self.prefill_model.forward_with_attn_bias(
                token_values=self._prefill_inputs[0],
                attn_bias=self._prefill_inputs[1],
                cache=self._cache,
            )
        torch.cuda.current_stream().wait_stream(s)

        self._prefill_cuda_graph = torch.cuda.CUDAGraph()
        recording_kwargs = {}
        if "capture_error_mode" in torch.cuda.graph.__init__.__annotations__:
            recording_kwargs["capture_error_mode"] = "thread_local"
        with torch.cuda.graph(self._prefill_cuda_graph, **recording_kwargs):
            self._prefill_logits = self.prefill_model.forward_with_attn_bias(
                token_values=self._prefill_inputs[0],
                attn_bias=self._prefill_inputs[1],
                cache=self._cache,
            )

        def replay(tokens, seq_lens=None):
            self._prefill_inputs[0].copy_(tokens)
            if seq_lens is not None:
                self._prefill_inputs[1].k_seqinfo.seqlen.copy_(seq_lens)

            self._prefill_cuda_graph.replay()
            torch.cuda.synchronize()

            return self._prefill_logits

        return replay

    def compile_generate(self, use_kernel):

        if self._cache is None:
            self._cache = fast.make_cache(
                args=self.model_args,
                length=self.gen_args.gen_bsz * self.max_seq_length,
            )

        seq_lens = [1 for _ in range(self.gen_args.gen_bsz)]
        kv_seq_lens = [self.gen_args.prompt_length for _ in range(self.gen_args.gen_bsz)]

        bias = AttnBias.from_seqlens(
            q_seqlen=seq_lens,
            kv_seqlen=kv_seq_lens,
            kv_padding=self.max_seq_length,
        )
        bias.q_seqinfo.to("cuda")
        bias.k_seqinfo.to("cuda")

        tokens = torch.IntTensor([1] * self.gen_args.gen_bsz).cuda()
        self._generate_inputs = (tokens, bias)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        decode_model = self.decode_model_int2 if use_kernel else self.decode_model_bf16
        with torch.cuda.stream(s):
            _ = decode_model.forward_with_attn_bias(
                token_values=self._generate_inputs[0],
                attn_bias=self._generate_inputs[1],
                cache=self._cache,
            )
        torch.cuda.current_stream().wait_stream(s)

        self._generate_cuda_graph = torch.cuda.CUDAGraph()
        recording_kwargs = {}
        if "capture_error_mode" in torch.cuda.graph.__init__.__annotations__:
            recording_kwargs["capture_error_mode"] = "thread_local"
        with torch.cuda.graph(self._generate_cuda_graph, **recording_kwargs):
            self._generate_logits = decode_model.forward_with_attn_bias(
                token_values=self._generate_inputs[0],
                attn_bias=self._generate_inputs[1],
                cache=self._cache,
            )

        def replay(tokens, seq_lens):
            self._generate_inputs[0].copy_(tokens)
            self._generate_inputs[1].k_seqinfo.seqlen.copy_(seq_lens)

            self._generate_cuda_graph.replay()

            return self._generate_logits

        return replay

    @torch.inference_mode()
    def benchmark(
        self,
        n_prompt: int,
        n_generation: int,
        n_warmup: int = 3,
        n_repeat: int = 10,
    ) -> None:
        """
        Benchmark prefill and decode speed independently.

        Args:
            n_prompt:     number of prompt tokens per request
            n_generation: number of tokens to generate per request
            n_warmup:     warmup iterations (not timed)
            n_repeat:     timed iterations for stable average
        """
        bs = self.gen_args.gen_bsz
        max_seq_length = n_prompt + n_generation

        print(f"\n{'='*60}")
        print(f"Benchmark  |  bsz={bs}  n_prompt={n_prompt}  n_generation={n_generation}")
        print(f"{'='*60}")

        cache = fast.make_cache(
            args=self.model_args,
            length=bs * max_seq_length,
        )

        # ------------------------------------------------------------------ #
        # Prefill benchmark
        # ------------------------------------------------------------------ #
        prefill_bias = AttnBias.from_seqlens(
            q_seqlen=[n_prompt] * bs,
            kv_seqlen=[n_prompt] * bs,
            kv_padding=max_seq_length,
        )
        prefill_bias.q_seqinfo.to("cuda")
        prefill_bias.k_seqinfo.to("cuda")
        # prefill_tokens = torch.randint(151936, (bs * n_prompt,), dtype=torch.int, device="cuda")
        prefill_tokens = torch.ones(bs * n_prompt, dtype=torch.int, device="cuda")

        def run_prefill():
            return self.prefill_model.forward_with_attn_bias(
                token_values=prefill_tokens,
                attn_bias=prefill_bias,
                cache=cache,
            )

        for _ in range(n_warmup):
            run_prefill()
        prefill_throughput = []
        for _ in range(n_repeat):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_prefill()
            torch.cuda.synchronize()
            time_eplased = time.perf_counter() - t0
            prefill_throughput.append(bs * n_prompt / time_eplased)
        prefill_throughput_mean = np.mean(prefill_throughput)
        prefill_throughput_std = np.std(np.array(prefill_throughput) - prefill_throughput_mean)

        print(f"\n[Prefill]")
        print(f"  throughput : {prefill_throughput_mean:.2f} ± {prefill_throughput_std:.2f} tokens/s")

        # ------------------------------------------------------------------ #
        # Decode benchmark  (single step × n_generation, kv pre-filled)
        # ------------------------------------------------------------------ #
        def run_decode_steps():
            kv_seqlen.fill_(n_prompt)
            decode_model = self.decode_model_int2 if use_kernel else self.decode_model_bf16
            for _ in range(n_generation):
                decode_model.forward_with_attn_bias(
                    token_values=decode_tokens,
                    attn_bias=decode_bias,
                    cache=cache,
                )
                kv_seqlen.add_(kv_seqlen < max_seq_length)

        for use_kernel in [False, True]:
            self._generate_compile_model = self.compile_generate(use_kernel=use_kernel)
            decode_bias = AttnBias.from_seqlens(
                q_seqlen=[1] * bs,
                kv_seqlen=[n_prompt] * bs,
                kv_padding=max_seq_length,
            )
            decode_bias.q_seqinfo.to("cuda")
            decode_bias.k_seqinfo.to("cuda")
            decode_tokens = torch.ones(bs, dtype=torch.int, device="cuda")
            kv_seqlen = decode_bias.k_seqinfo.seqlen  # updated in-place each step

            print('Warmup decoding benchmark')
            for _ in tqdm(range(n_warmup)):
                run_decode_steps()

            print('Run decoding benchmark')
            decode_throughput = []
            for _ in tqdm(range(n_repeat)):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                run_decode_steps()
                torch.cuda.synchronize()
                time_elapsed = time.perf_counter() - t0
                decode_throughput.append(bs * n_generation / time_elapsed)
            decode_throughput_mean = np.mean(decode_throughput)
            decode_throughput_std = np.std(np.array(decode_throughput) - decode_throughput_mean)

            print(f"\n[Decode - {'INT2' if use_kernel else 'BF16'}]")
            print(f"  throughput : {decode_throughput_mean:.2f} ± {decode_throughput_std:.2f} tokens/s")


    @torch.inference_mode()
    def generate_all(
        self, prompts: list[list[int]], use_cuda_graphs: bool, use_sampling: bool
    ) -> Tuple[Stats, list[list[int]]]:
        bs = len(prompts)
        prompt_lens = [len(p) for p in prompts]
        padded_prompt_lens = [self.gen_args.prompt_length] * bs
        max_prompt_length = max(prompt_lens)
        gen_length = self.gen_args.gen_length
        max_seq_length = max_prompt_length + gen_length

        bias = AttnBias.from_seqlens(
            q_seqlen=padded_prompt_lens,
            kv_seqlen=prompt_lens,
            kv_padding=max_seq_length,
        )
        bias.q_seqinfo.to("cuda")
        bias.k_seqinfo.to("cuda")

        kv_seqlen = bias.k_seqinfo.seqlen
        prompts = [prompt + [self.tokenizer.pad_token_id] * (self.gen_args.prompt_length - len(prompt)) for prompt in prompts]
        tokens = torch.IntTensor(sum(prompts, [])).cuda()
        out_tokens = torch.zeros((max_seq_length, bs), dtype=torch.int)

        stats = Stats()
        torch.cuda.synchronize()
        stats.phase("prefill" if use_cuda_graphs else "total")

        output = self._prefill_compile_model(tokens, None)

        logits = output[kv_seqlen - 1, :]
        logits = logits.view(bs, self.model_args.vocab_size)

        if use_sampling:
            temp = 0.7
            top_p = 0.95
            probs = torch.softmax(logits / temp, dim=-1)
            next_token = sample_utils.top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits, dim=-1)

        next_token = next_token.reshape(bs)
        out_tokens[0, :] = next_token

        torch.cuda.synchronize()
        stats.phase("decode" if use_cuda_graphs else "total")

        eos_id = self.tokenizer.eos_token_id

        for niter in range(1, gen_length):
            kv_seqlen.add_(kv_seqlen < max_seq_length)
            output = self._generate_compile_model(next_token, kv_seqlen)

            logits = output.view(bs, self.model_args.vocab_size)

            if use_sampling:
                temp = 0.7
                top_p = 0.95
                probs = torch.softmax(logits / temp, dim=-1)
                next_token = sample_utils.top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)

            next_token = next_token.reshape(bs)
            out_tokens[niter, :] = next_token

            if next_token.eq(eos_id).any():
                break

        torch.cuda.synchronize()
        stats.end_phase(tokens=niter * bs)

        def trim_answer(prompt_len, tokens):
            tokens = tokens[: max_seq_length - prompt_len]
            if eos_id in tokens:
                return tokens[: tokens.index(eos_id) + 1]
            else:
                return tokens

        answers = [
            trim_answer(prompt_len, answer)
            for prompt_len, answer in zip(prompt_lens, out_tokens.t().tolist())
        ]
        return stats, answers


def get_prompts(interactive: bool) -> Iterable[list[str]]:
    if interactive:
        while True:
            try:
                prompts = input("enter prompt: ").split("\n")
            except EOFError:
                print("exiting")
                sys.exit(0)
            yield prompts
    else:
        yield [
            "Hello, my name is",
        ]


def main(
    ckpt_dir: str,
    interactive: bool = False,
    chat_format: bool = False,
    sampling: bool = False,
    tokenizer_path: Optional[str] = None,
    # benchmark flags
    benchmark: bool = False,
    n_prompt: int = 128,
    n_generation: int = 128,
    n_warmup: int = 1,
    n_repeat: int = 10,
):
    local_rank = 0
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)

    gen_args = GenArgs(
        prompt_length=n_prompt,
        gen_length=n_generation,
    ) if benchmark else GenArgs()

    g = FastGen.build(ckpt_dir, gen_args, device, tokenizer_path=tokenizer_path)

    if benchmark:
        g.benchmark(
            n_prompt=n_prompt,
            n_generation=n_generation,
            n_warmup=n_warmup,
            n_repeat=n_repeat,
        )
        return

    for prompts in get_prompts(interactive):
        if chat_format:
            tokens = [
                g.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=True,
                )
                for prompt in prompts
            ]
        else:
            tokens = [g.tokenizer.encode(prompt) for prompt in prompts]

        print(tokens)
        stats, out_tokens = g.generate_all(
            tokens, use_cuda_graphs="NO_CUDA_GRAPHS" not in os.environ, use_sampling=sampling,
        )

        for i, prompt in enumerate(prompts):
            print(f"> {prompt}")
            answer = g.tokenizer.decode(out_tokens[i], skip_special_tokens=False)
            print(answer)
            print("---------------")

        for phase_stats in stats.phases:
            print(phase_stats.show())

        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


if __name__ == "__main__":
    fire.Fire(main)