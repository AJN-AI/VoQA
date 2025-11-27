MODEL_PATH=/root/autodl-fs/checkpoints/tiny-llava-with-text-image-qwen2-0.5B-siglip-base-stage3-finetune
IMAGE_FILE=/autodl-fs/data/eval/vqav2/test2015/COCO_test2015_000000262144.jpg
TEXT_IMAGE_FILE=/autodl-fs/data/eval/vqav2/vqav2_question_image/262144005.jpg

python tinyllava/eval/run_tiny_llava_with_text_image.py \
    --model_path $MODEL_PATH\
    --image_file $IMAGE_FILE\
    --text_image_file $TEXT_IMAGE_FILE\
    --conv_mode phi_inference\
    --sep ','\
    --temperature 0\
    --num_beams 1\
    --max_new_tokens 512\
    --for_benchmark True

