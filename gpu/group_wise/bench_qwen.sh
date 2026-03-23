#sudo nvidia-smi -i 1 -lmc 810
CUDA_VISIBLE_DEVICES=1 python /home/kyyx/github/LLM/benchmark/test_vram.py
CUDA_VISIBLE_DEVICES=1 python bench_qwen.py ../../../../models/qwen3-4b-lora64-train_no_quant_mu/i2_s/ --tokenizer_path ../../../../models/Qwen3-4B/ --n_prompt 8000 --n_generation 1000 --benchmark
