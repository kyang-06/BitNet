#FAKE_QUANT_MODEL_PATH='../../../../models/qwen3-4b-lora64-train_no_quant_mu/Qwen3-4B-SliderQuant-1.58bit-export/'
FAKE_QUANT_MODEL_PATH='../../../../models/new-qwen3-4b-lora64-train_no_quant_mu_g128_w1.58A8_bitnet/fake_quant'

#python convert_safetensors_qwen.py --safetensors_file ${FAKE_QUANT_MODEL_PATH}/model.safetensors  --output ${FAKE_QUANT_MODEL_PATH}/bitnet_gpu_model_state.pt --model_name Qwen3-4B
#python convert_checkpoint_qwen.py --input ${FAKE_QUANT_MODEL_PATH}/bitnet_gpu_model_state.pt --output_dir ${FAKE_QUANT_MODEL_PATH}/../i2_s

CUDA_VISIBLE_DEVICES=2 python generate_qwen.py ${FAKE_QUANT_MODEL_PATH}/../i2_s --interactive --chat_format --tokenizer_path ../../../../models/Qwen3-4B/
