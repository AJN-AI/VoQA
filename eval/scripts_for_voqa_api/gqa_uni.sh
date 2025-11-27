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

SPLIT="llava_gqa_testdev_balanced"
# SPLIT="llava_gqa_testdev_balanced_for_test"
GQADIR="$EVAL_DIR/gqa"

python eval/eval_for_api.py \
    --question-file $EVAL_DIR/gqa/$SPLIT.jsonl \
    --image-folder $EVAL_DIR/gqa/$METHOD_FOLDER \
    --answers_path $EVAL_DIR/gqa/answers/${SPLIT}_prompt${PROMPT_ID}/$METHOD_FOLDER/${MODEL_NAME} \
    --temperature 0 \
    --direction $DIRECTION \
    --model-name $MODEL_NAME \
    --task gqa \
    --batch-size 1 \
    --prompt "$PROMPT" \
    --original_benchmark "false" \
    --API_KEY "$API_KEY" \
    --thread_num $THREAD_NUM


output_file=$EVAL_DIR/gqa/answers/${SPLIT}_prompt${PROMPT_ID}/$METHOD_FOLDER/${MODEL_NAME}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((THREAD_NUM-1))); do
    cat $EVAL_DIR/gqa/answers/${SPLIT}_prompt${PROMPT_ID}/$METHOD_FOLDER/${MODEL_NAME}/thread_${IDX}.jsonl >> "$output_file"
done

python eval/convert_gqa_for_eval.py \
    --src $output_file \
    --dst $GQADIR/answers/${SPLIT}_prompt${PROMPT_ID}/$METHOD_FOLDER/${MODEL_NAME}/testdev_balanced_predictions.json \
    --filter_answer $FILTER_ANSWER \
    --split_word $SPLIT_WORD \
    --model_type $MODEL_TYPE

cd $GQADIR
python eval/eval.py \
    --tier testdev_balanced \
    --predictions answers/${SPLIT}_prompt${PROMPT_ID}/$METHOD_FOLDER/${MODEL_NAME}/testdev_balanced_predictions.json
