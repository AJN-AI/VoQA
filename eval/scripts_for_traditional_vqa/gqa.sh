#!/bin/bash
MODEL_PATH="$1"
EVAL_DIR="$2"

echo "$MODEL_PATH $EVAL_DIR Start!"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_gqa_testdev_balanced"
GQADIR="$EVAL_DIR/gqa"

MODEL_NAME=$(basename ${MODEL_PATH})

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python eval/eval_main.py \
        --model-path $MODEL_PATH \
        --question-file $EVAL_DIR/gqa/$SPLIT.jsonl \
        --image-folder $EVAL_DIR/gqa/images \
        --answers-file $EVAL_DIR/gqa/answers/${SPLIT}_for_original_benchmark/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --model-name $MODEL_NAME \
        --task gqa \
        --batch-size 1 \
        --original_benchmark "true" &
done

wait

output_file=$EVAL_DIR/gqa/answers/${SPLIT}_for_original_benchmark/$MODEL_NAME/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $EVAL_DIR/gqa/answers/${SPLIT}_for_original_benchmark/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python eval/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/answers/${SPLIT}_for_original_benchmark/$MODEL_NAME/testdev_balanced_predictions.json --filter_answer "false"

cd $GQADIR
python eval/eval.py --tier testdev_balanced --predictions answers/${SPLIT}_for_original_benchmark/$MODEL_NAME/testdev_balanced_predictions.json
