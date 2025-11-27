#!/bin/bash
MODEL_NAME="$1"
EVAL_DIR="$2"
API_KEY="$3"
THREAD_NUM="$4"

echo "$MODEL_NAME $EVAL_DIR $API_KEY $THREAD_NUM Start!"

python eval/eval_for_api.py \
    --question-file $EVAL_DIR/textvqa/llava_textvqa_val_v051_ocr_new_id_without_ocr_reference.jsonl \
    --image-folder $EVAL_DIR/textvqa/train_images \
    --answers_path $EVAL_DIR/textvqa/answers_for_original_benchmark/$MODEL_NAME \
    --temperature 0 \
    --model-name $MODEL_NAME \
    --task textvqa \
    --batch-size 1 \
    --original_benchmark "true" \
    --API_KEY "$API_KEY" \
    --thread_num $THREAD_NUM 

output_file=$EVAL_DIR/textvqa/answers_for_original_benchmark/$MODEL_NAME/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((THREAD_NUM-1))); do
    cat $EVAL_DIR/textvqa/answers_for_original_benchmark/$MODEL_NAME/thread_${IDX}.jsonl >> "$output_file"
done

python eval/eval_textvqa.py \
    --annotation-file $EVAL_DIR/textvqa/TextVQA_0.5.1_val_new_id.json \
    --result-file $EVAL_DIR/textvqa/answers_for_original_benchmark/$MODEL_NAME/merge.jsonl \
    --filter_answer "false" \
    --original_benchmark "true" 