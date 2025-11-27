#!/bin/bash
MODEL_NAME="$1"
EVAL_DIR="$2"
API_KEY="$3"
THREAD_NUM="$4"

echo "$MODEL_NAME $EVAL_DIR $API_KEY $THREAD_NUM Start!"

SPLIT="llava_gqa_testdev_balanced"
# SPLIT="llava_gqa_testdev_balanced_for_test"
GQADIR="$EVAL_DIR/gqa"

python eval/eval_for_api.py \
    --question-file $EVAL_DIR/gqa/$SPLIT.jsonl \
    --image-folder $EVAL_DIR/gqa/images \
    --answers_path $EVAL_DIR/gqa/answers/${SPLIT}_for_original_benchmark/$MODEL_NAME \
    --temperature 0 \
    --model-name $MODEL_NAME \
    --task gqa \
    --batch-size 1 \
    --original_benchmark "true" \
    --API_KEY "$API_KEY" \
    --thread_num $THREAD_NUM

output_file=$EVAL_DIR/gqa/answers/${SPLIT}_for_original_benchmark/$MODEL_NAME/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((THREAD_NUM-1))); do
    cat $EVAL_DIR/gqa/answers/${SPLIT}_for_original_benchmark/$MODEL_NAME/thread_${IDX}.jsonl >> "$output_file"
done

python eval/convert_gqa_for_eval.py \
    --src $output_file \
    --dst $GQADIR/answers/${SPLIT}_for_original_benchmark/$MODEL_NAME/testdev_balanced_predictions.json \
    --filter_answer "false"

cd $GQADIR
python eval/eval.py \
    --tier testdev_balanced \
    --predictions answers/${SPLIT}_for_original_benchmark/$MODEL_NAME/testdev_balanced_predictions.json
