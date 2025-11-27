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

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_gqa_testdev_balanced"
GQADIR="$EVAL_DIR/gqa"

MODEL_NAME=$(basename ${MODEL_PATH})

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python eval/eval_main.py \
#         --model-path $MODEL_PATH \
#         --question-file $EVAL_DIR/gqa/$SPLIT.jsonl \
#         --image-folder $EVAL_DIR/gqa/$METHOD_FOLDER \
#         --answers-file $EVAL_DIR/gqa/answers/${SPLIT}_prompt${PROMPT_ID}/${MODEL_NAME}_${DIRECTION}/$METHOD_FOLDER/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --direction $DIRECTION \
#         --model-name $MODEL_NAME \
#         --task gqa \
#         --batch-size 1 \
#         --prompt "$PROMPT" \
#         --original_benchmark "false" &
# done

# wait

output_file=$EVAL_DIR/gqa/answers/${SPLIT}_prompt${PROMPT_ID}/${MODEL_NAME}_${DIRECTION}/$METHOD_FOLDER/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat $EVAL_DIR/gqa/answers/${SPLIT}_prompt${PROMPT_ID}/${MODEL_NAME}_${DIRECTION}/$METHOD_FOLDER/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

# if [ "$FILTER_ANSWER" = "false" ]; then
#     PREDICTION_FILE_NAME="testdev_balanced_predictions_without_filter.json"
#     echo "PREDICTION_FILE_NAME: $PREDICTION_FILE_NAME"
# else
#     PREDICTION_FILE_NAME="testdev_balanced_predictions_with_filter.json"
#     echo "PREDICTION_FILE_NAME: $PREDICTION_FILE_NAME"
# fi

python eval/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/answers/${SPLIT}_prompt${PROMPT_ID}/${MODEL_NAME}_${DIRECTION}/$METHOD_FOLDER/testdev_balanced_predictions.json \
    --filter_answer $FILTER_ANSWER \
    --split_word $SPLIT_WORD \
    --model_type $MODEL_TYPE

cd $GQADIR
python eval/eval.py --tier testdev_balanced --predictions answers/${SPLIT}_prompt${PROMPT_ID}/${MODEL_NAME}_${DIRECTION}/$METHOD_FOLDER/testdev_balanced_predictions.json