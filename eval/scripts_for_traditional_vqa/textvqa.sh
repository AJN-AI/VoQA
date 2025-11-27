#!/bin/bash
MODEL_PATH="$1"
EVAL_DIR="$2"

echo "$MODEL_PATH $EVAL_DIR Start!"

MODEL_NAME=$(basename ${MODEL_PATH})

python eval/eval_main.py \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/textvqa/llava_textvqa_val_v051_ocr_new_id_without_ocr_reference.jsonl \
    --image-folder $EVAL_DIR/textvqa/train_images \
    --answers-file $EVAL_DIR/textvqa/answers_for_original_benchmark/$MODEL_NAME.jsonl \
    --temperature 0 \
    --model-name $MODEL_NAME \
    --task textvqa \
    --batch-size 1 \
    --original_benchmark "true" 

python eval/eval_textvqa.py \
    --annotation-file $EVAL_DIR/textvqa/TextVQA_0.5.1_val_new_id.json \
    --result-file $EVAL_DIR/textvqa/answers_for_original_benchmark/$MODEL_NAME.jsonl \
    --filter_answer "false" \
    --original_benchmark "true" 