#!/bin/bash
MODEL_PATH="$1"
EVAL_DIR="$2"

echo "$MODEL_PATH $EVAL_DIR Start!"

MODEL_NAME=$(basename ${MODEL_PATH})

python eval/eval_main.py \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --image-folder $EVAL_DIR/pope/val2014 \
    --answers-file $EVAL_DIR/pope/answers_for_original_benchmark/$MODEL_NAME.jsonl \
    --temperature 0 \
    --model-name $MODEL_NAME \
    --task pope \
    --batch-size 1 \
    --original_benchmark "true" 

python eval/eval_pope.py \
    --annotation-dir $EVAL_DIR/pope/coco \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --result-file $EVAL_DIR/pope/answers_for_original_benchmark/$MODEL_NAME.jsonl \
    --filter_answer "false"
