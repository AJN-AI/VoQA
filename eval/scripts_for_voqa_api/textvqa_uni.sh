#!/bin/bash
MODEL_NAME="$1"
METHOD_FOLDER="$2"
DIRECTION="$3"
PROMPT="$4"
PROMPT_ID="$5"
EVAL_DIR="$6"
FILTER_ANSWER="$7"
SPLIT_WORD="$8"
MODEL_TYPE="$9"
API_KEY="${10}"
THREAD_NUM="${11}"

echo "$MODEL_NAME $METHOD_FOLDER $DIRECTION $EVAL_DIR $FILTER_ANSWER $SPLIT_WORD $MODEL_TYPE $API_KEY $THREAD_NUM"
echo "prompt $PROMPT prompt_id $PROMPT_ID Start!"

python eval/eval_for_api.py \
    --question-file $EVAL_DIR/textvqa/llava_textvqa_val_v051_ocr_new_id_without_ocr_reference.jsonl \
    --image-folder $EVAL_DIR/textvqa/$METHOD_FOLDER \
    --answers_path $EVAL_DIR/textvqa/answers_prompt${PROMPT_ID}/$METHOD_FOLDER/${MODEL_NAME} \
    --temperature 0 \
    --direction $DIRECTION \
    --model-name $MODEL_NAME \
    --task textvqa \
    --batch-size 1 \
    --prompt "$PROMPT" \
    --original_benchmark "false" \
    --API_KEY "$API_KEY" \
    --thread_num $THREAD_NUM 

output_file=$EVAL_DIR/textvqa/answers_prompt${PROMPT_ID}/$METHOD_FOLDER/${MODEL_NAME}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((THREAD_NUM-1))); do
    cat $EVAL_DIR/textvqa/answers_prompt${PROMPT_ID}/$METHOD_FOLDER/${MODEL_NAME}/thread_${IDX}.jsonl >> "$output_file"
done

python eval/eval_textvqa.py \
    --annotation-file $EVAL_DIR/textvqa/TextVQA_0.5.1_val_new_id.json \
    --result-file $EVAL_DIR/textvqa/answers_prompt${PROMPT_ID}/$METHOD_FOLDER/${MODEL_NAME}/merge.jsonl \
    --filter_answer $FILTER_ANSWER \
    --split_word $SPLIT_WORD \
    --model_type $MODEL_TYPE
