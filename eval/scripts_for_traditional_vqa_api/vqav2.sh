#!/bin/bash
MODEL_NAME="$1"
EVAL_DIR="$2"
API_KEY="$3"
THREAD_NUM="$4"

echo "$MODEL_NAME $EVAL_DIR $API_KEY $THREAD_NUM Start!"

SPLIT="llava_vqav2_mscoco_test-dev2015"
# SPLIT="llava_vqav2_mscoco_test-dev2015_for_test"

python eval/eval_for_api.py \
    --question-file $EVAL_DIR/vqav2/$SPLIT.jsonl \
    --image-folder $EVAL_DIR/vqav2/test2015 \
    --answers_path $EVAL_DIR/vqav2/answers/${SPLIT}_for_original_benchmark/$MODEL_NAME \
    --temperature 0 \
    --model-name $MODEL_NAME \
    --task vqav2 \
    --batch-size 1 \
    --original_benchmark "true" \
    --API_KEY "$API_KEY" \
    --thread_num $THREAD_NUM

output_file=$EVAL_DIR/vqav2/answers/${SPLIT}_for_original_benchmark/$MODEL_NAME/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((THREAD_NUM-1))); do
    cat $EVAL_DIR/vqav2/answers/${SPLIT}_for_original_benchmark/$MODEL_NAME/thread_${IDX}.jsonl >> "$output_file"
done

python eval/convert_vqav2_for_submission.py \
    --split ${SPLIT}_for_original_benchmark \
    --ckpt $MODEL_NAME \
    --dir $EVAL_DIR/vqav2 \
    --filter_answer "false"
