#!/bin/bash
MODEL_PATH="$1"
EVAL_DIR="$2"

echo "$MODEL_PATH $EVAL_DIR Start!"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_vqav2_mscoco_test-dev2015"

MODEL_NAME=$(basename ${MODEL_PATH})

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python eval/eval_main.py \
        --model-path $MODEL_PATH \
        --question-file $EVAL_DIR/vqav2/$SPLIT.jsonl \
        --image-folder $EVAL_DIR/vqav2/test2015 \
        --answers-file $EVAL_DIR/vqav2/answers/${SPLIT}_for_original_benchmark/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --model-name $MODEL_NAME \
        --task vqav2 \
        --batch-size 1 \
        --original_benchmark "true" &
done

wait

output_file=$EVAL_DIR/vqav2/answers/${SPLIT}_for_original_benchmark/$MODEL_NAME/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $EVAL_DIR/vqav2/answers/${SPLIT}_for_original_benchmark/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python eval/convert_vqav2_for_submission.py --split ${SPLIT}_for_original_benchmark --ckpt $MODEL_NAME --dir $EVAL_DIR/vqav2 --filter_answer "false"
