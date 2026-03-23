GPU_ID=3
#sudo nvidia-smi -i $GPU_ID -lgc 1050
#sudo nvidia-smi -i $GPU_ID -lmc 810
CUDA_VISIBLE_DEVICES=$GPU_ID python /home/kyyx/github/LLM/benchmark/test_vram.py
CUDA_VISIBLE_DEVICES=$GPU_ID python /home/kyyx/github/LLM/benchmark/test_flops.py
CUDA_VISIBLE_DEVICES=$GPU_ID python bench_kernel_latency_group_vs_tensor.py --sweep_name Qwen3-4B --n_repeat 200 --n_request 100
sudo nvidia-smi -i $GPU_ID -rgc
sudo nvidia-smi -i $GPU_ID -rmc