#!/bin/bash
MODEL_PATH="$1"
METHOD_FOLDER="$2"
DIRECTION="$3"
PROMPT="$4"
PROMPT_ID="$5"
EVAL_DIR="$6"
FILTER_ANSWER="$7"
SPLIT_WORD="$8"
MODEL_TYPE="$9"

echo "$MODEL_PATH $METHOD_FOLDER $TASK $DIRECTION $EVAL_DIR $FILTER_ANSWER $SPLIT_WORD $MODEL_TYPE"
echo "prompt $PROMPT prompt_id $PROMPT_ID Start!"

MODEL_NAME=$(basename ${MODEL_PATH})

python eval/eval_main.py \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/textvqa/llava_textvqa_val_v051_ocr_new_id_without_ocr_reference.jsonl \
    --image-folder $EVAL_DIR/textvqa/$METHOD_FOLDER \
    --answers-file $EVAL_DIR/textvqa/answers_prompt${PROMPT_ID}/$METHOD_FOLDER/${MODEL_NAME}_${DIRECTION}.jsonl \
    --temperature 0 \
    --direction $DIRECTION \
    --model-name $MODEL_NAME \
    --task textvqa \
    --batch-size 1 \
    --prompt "$PROMPT" \
    --original_benchmark "false" 

python eval/eval_textvqa.py \
    --annotation-file $EVAL_DIR/textvqa/TextVQA_0.5.1_val_new_id.json \
    --result-file $EVAL_DIR/textvqa/answers_prompt${PROMPT_ID}/$METHOD_FOLDER/${MODEL_NAME}_${DIRECTION}.jsonl \
    --filter_answer $FILTER_ANSWER \
    --split_word $SPLIT_WORD \
    --model_type $MODEL_TYPE
