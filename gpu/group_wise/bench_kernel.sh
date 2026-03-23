sudo CUDA_VISIBLE_DEVICES=1 nvidia-smi -i 1 -lgc 1050
#sudo CUDA_VISIBLE_DEVICES=1 nvidia-smi -i 1 -lmc 810
CUDA_VISIBLE_DEVICES=1 python /home/kyyx/github/LLM/benchmark/test_vram.py
CUDA_VISIBLE_DEVICES=1 python /home/kyyx/github/LLM/benchmark/test_flops.py
CUDA_VISIBLE_DEVICES=1 python bench_kernel_latency_precise.py --sweep_name Qwen3-NEXT --n_repeat 200 --n_request 100
sudo CUDA_VISIBLE_DEVICES=1 nvidia-smi -i 1 -rgc
sudo CUDA_VISIBLE_DEVICES=1 nvidia-smi -i 1 -rmc