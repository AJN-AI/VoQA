#!/bin/bash
MODEL_PATH="$1"
EVAL_DIR="$2"

echo "$MODEL_PATH $EVAL_DIR Start!"

MODEL_NAME=$(basename ${MODEL_PATH})

python eval/eval_main.py \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/scienceqa/llava_test_CQM-A_selected_mm.jsonl \
    --image-folder $EVAL_DIR/scienceqa/images/test \
    --answers-file $EVAL_DIR/scienceqa/answers_for_original_benchmark/$MODEL_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --model-name $MODEL_NAME \
    --task scienceqa \
    --batch-size 1 \
    --original_benchmark "true" 

python eval/eval_science_qa.py \
    --base-dir $EVAL_DIR/scienceqa \
    --result-file $EVAL_DIR/scienceqa/answers_for_original_benchmark/$MODEL_NAME.jsonl \
    --output-file $EVAL_DIR/scienceqa/answers_for_original_benchmark/${MODEL_NAME}_output.jsonl \
    --output-result $EVAL_DIR/scienceqa/answers_for_original_benchmark/${MODEL_NAME}_result.json \
    --filter_answer "false"
