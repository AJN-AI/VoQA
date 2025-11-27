MODEL_PATH=/path/to/tiny-llava-qwen2-0.5B-siglip-uni-finetune-fulldata-watermark-v2-only-qa
IMAGE_FILE=/path/to/vqav2/vqav2_watermark_rendering_image/1001.jpg
python tinyllava/eval/run_tiny_llava_uni.py \
    --model-path $MODEL_PATH\
    --image-file $IMAGE_FILE\
    --conv-mode phi_uni\
    --sep ','\
    --temperature 0\
    --num_beams 1\
    --max_new_tokens 512\
    --for_benchmark True
