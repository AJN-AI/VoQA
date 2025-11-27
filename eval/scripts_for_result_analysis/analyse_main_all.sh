# Script for the model fine-tuned based on Question-Alignment SFT strategy, which calculates the accuracy in predicting problems in the picture.

# 1. The prompt id needs to correspond to the prompt used in reasoning. For SFT, default is 0
PROMPT_ID=0
# 2. VoQA dataset folder names (concatenation or watermark rendering)
METHOD_FOLDER='slide_with_hsv/alpha=100%/border_width=-1'
# 3. The model name, which should correspond to the saved json or jsonl.
MODEL_NAMES=(TinyLLaVA-Qwen2-0.5B-SigLIP-QRA, TinyLLaVA-Qwen2.5-3B-SigLIP-QRA, TinyLLaVA-Qwen2-0.5B-SigLIP-QRA-HELPER, TinyLLaVA-Qwen2-0.5B-SigLIP-QRA-CAT)
# 4. Concatenation directions (for watermark rendering, just keep 'no')
DIRECTION='no'
# 5. VoQA dataset path (root path)
EVAL_DIR="/path/to/VoQA_benchmark"
# 6. Whether the answers will be further carefully filtered (for zero-shot models, default is true; for training models, default is false)
FILTER_ANSWER="false"
# 7. Trigger token in the models, which is case-insensitive. Default is 'ASSISTANT'.
SPLIT_WORD="ASSISTANT"
# SPLIT_WORD="HELPER"
# SPLIT_WORD="CAT"

8. The type of the model relative to the VoQA dataset. It needs to correspond to the model name.
# MODEL_TYPE="qa-sft"
MODEL_TYPE="qra-sft"

# 9. The tasks in VoQA, you can choose in [scienceqa, textvqa, pope, gqa]
TASKS=(scienceqa textvqa pope gqa)

# 10. Special args for GQA only
GQA_SPLIT="llava_gqa_testdev_balanced"
GQA_TIER="testdev_balanced" # tier in GQA scripts


for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "########## Current Model Name: $MODEL_NAME ##########"

    for TASK in "${TASKS[@]}"; do
        echo "##### Current task: $TASK #####"

        if [ "$TASK" = "scienceqa" ]; then
            python scripts_for_result_analysis/eval_others.py \
                --task $TASK \
                --sqa_resoning_json "$EVAL_DIR/scienceqa/answers_prompt$PROMPT_ID/scienceqa_syn_image/${METHOD_FOLDER}/${MODEL_NAME}_${DIRECTION}.jsonl" \
                --sqa_prediction_json "$EVAL_DIR/scienceqa/answers_prompt$PROMPT_ID/scienceqa_syn_image/${METHOD_FOLDER}/${MODEL_NAME}_${DIRECTION}_output.jsonl" \
                --sqa_save_file "$EVAL_DIR/scienceqa/answers_prompt$PROMPT_ID/scienceqa_syn_image/${METHOD_FOLDER}/${MODEL_NAME}_${DIRECTION}_for_analysis.jsonl" \
                --analysis_file scripts_for_result_analysis/result_files/${MODEL_NAME}-sqa_result_analysis.jsonl \
                --split_word $SPLIT_WORD \
                --model_name $MODEL_NAME

        elif [ "$TASK" = "textvqa" ]; then
            python scripts_for_result_analysis/eval_others.py \
                --task $TASK \
                --textvqa_id_to_score_json "$EVAL_DIR/textvqa/answers_prompt$PROMPT_ID/textvqa_syn_image/${METHOD_FOLDER}/${MODEL_NAME}_${DIRECTION}_score_to_qids.json" \
                --textvqa_prediction_json "$EVAL_DIR/textvqa/answers_prompt$PROMPT_ID/textvqa_syn_image/${METHOD_FOLDER}/${MODEL_NAME}_${DIRECTION}.jsonl" \
                --textvqa_save_file "$EVAL_DIR/textvqa/answers_prompt$PROMPT_ID/textvqa_syn_image/${METHOD_FOLDER}/${MODEL_NAME}_${DIRECTION}_for_analysis.jsonl" \
                --analysis_file scripts_for_result_analysis/result_files/${MODEL_NAME}-textvqa_result_analysis.jsonl \
                --split_word $SPLIT_WORD \
                --annotation-file $EVAL_DIR/textvqa/TextVQA_0.5.1_val_new_id.json \
                --result-file $EVAL_DIR/textvqa/answers_prompt${PROMPT_ID}/textvqa_syn_image/$METHOD_FOLDER/${MODEL_NAME}_${DIRECTION}.jsonl \
                --filter_answer $FILTER_ANSWER \
                --model_type $MODEL_TYPE \
                --model_name $MODEL_NAME \
                --question_file $EVAL_DIR/textvqa/llava_textvqa_val_v051_ocr_new_id_without_ocr_reference.jsonl

        elif [ "$TASK" = "pope" ]; then
            python scripts_for_result_analysis/eval_pope.py \
                --annotation-dir $EVAL_DIR/pope/coco \
                --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
                --result-file $EVAL_DIR/pope/answers_prompt${PROMPT_ID}/pope_syn_image/$METHOD_FOLDER/${MODEL_NAME}_${DIRECTION}.jsonl \
                --filter_answer $FILTER_ANSWER \
                --split_word $SPLIT_WORD \
                --model_type $MODEL_TYPE \
                --analysis_file scripts_for_result_analysis/result_files/${MODEL_NAME}-pope_result_analysis.jsonl \
                --model_name $MODEL_NAME

        elif [ "$TASK" = "gqa" ]; then
            python scripts_for_result_analysis/eval_others.py \
                --task $TASK \
                --gqa_reference_json "$EVAL_DIR/gqa/${GQA_TIER}_questions.json" \
                --gqa_reasoning_json "$EVAL_DIR/gqa/answers/${GQA_SPLIT}_prompt${PROMPT_ID}/${MODEL_NAME}_${DIRECTION}/gqa_syn_image/$METHOD_FOLDER/merge.jsonl" \
                --gqa_prediction_json "$EVAL_DIR/gqa/answers/${GQA_SPLIT}_prompt${PROMPT_ID}/${MODEL_NAME}_${DIRECTION}/gqa_syn_image/$METHOD_FOLDER/${GQA_TIER}_predictions.json" \
                --analysis_file scripts_for_result_analysis/result_files/${MODEL_NAME}-gqa_result_analysis.jsonl \
                --split_word $SPLIT_WORD \
                --model_name $MODEL_NAME
        else
            echo "Unknown task name: $TASK. exit!"
        fi

    done

done
