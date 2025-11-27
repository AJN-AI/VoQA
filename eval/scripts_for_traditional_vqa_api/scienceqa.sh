#!/bin/bash
MODEL_NAME="$1"
EVAL_DIR="$2"
API_KEY="$3"
THREAD_NUM="$4"

echo "$MODEL_NAME $EVAL_DIR $API_KEY $THREAD_NUM Start!"

python eval/eval_for_api.py \
    --question-file $EVAL_DIR/scienceqa/llava_test_CQM-A_selected_mm.jsonl \
    --image-folder $EVAL_DIR/scienceqa/images/test \
    --answers_path $EVAL_DIR/scienceqa/answers_for_original_benchmark/$MODEL_NAME \
    --single-pred-prompt \
    --temperature 0 \
    --model-name $MODEL_NAME \
    --task scienceqa \
    --batch-size 1 \
    --original_benchmark "true" \
    --API_KEY "$API_KEY" \
    --thread_num $THREAD_NUM

output_file=$EVAL_DIR/scienceqa/answers_for_original_benchmark/$MODEL_NAME/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((THREAD_NUM-1))); do
    cat $EVAL_DIR/scienceqa/answers_for_original_benchmark/$MODEL_NAME/thread_${IDX}.jsonl >> "$output_file"
done

python eval/eval_science_qa.py \
    --base-dir $EVAL_DIR/scienceqa \
    --result-file $EVAL_DIR/scienceqa/answers_for_original_benchmark/${MODEL_NAME}/merge.jsonl \
    --output-file $EVAL_DIR/scienceqa/answers_for_original_benchmark/${MODEL_NAME}/final_output.jsonl \
    --output-result $EVAL_DIR/scienceqa/answers_for_original_benchmark/${MODEL_NAME}/final_result.json \
    --filter_answer "false"
