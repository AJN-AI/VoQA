# Script for watermark rendering evaluation without any model reasoning, just for response filtering. 
# Please set all the parameters as follows:
# 1. Zero-shot model path (You can add other models here)
MODEL_PATHS=(tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B liuhaotian/llava-v1.5-7b OpenGVLab/InternVL2_5-1B Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5 deepseek-ai/deepseek-vl2-tiny Qwen/Qwen2.5-VL-3B-Instruct Qwen/Qwen2-VL-2B)
# 2. VoQA dataset folder names (concatenation or watermark rendering)
METHOD_FOLDERS=(watermark_rendering)
# 3. The prompt ids corresponding to the designed prompts in step 10 below (line 26).
PROMPT_IDS=(0 1 2 3 4 5 6 7)
# 4. VoQA evaluation tasks
TASKS=(scienceqa pope textvqa gqa vqav2) 
# 5. Concatenation directions (for watermark rendering, just keep 'no')
DIRECTION="no"
# 6. VoQA dataset path (root path)
EVAL_DIR="path/to/VoQA/evaluation/dataset"
# 7. Whether the answers will be further carefully filtered (for zero-shot models, default is true; for training models, default is false)
FILTER_ANSWER="true"
# 8. Trigger token in the models, which is case-insensitive. Default is ASSISTANT.
SPLIT_WORD="ASSISTANT"
# 9. The type of the model relative to the VoQA dataset, you can choose in ['zero-shot', 'baseline-sft', 'qa-sft', 'qra-sft'].
MODEL_TYPE="zero-shot"


for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    for METHOD_FOLDER in "${METHOD_FOLDERS[@]}"; do
        for PROMPT_ID in "${PROMPT_IDS[@]}"; do

            # 10. Prompt ID ——> prompt
            if [ "$PROMPT_ID" -eq 0 ]; then
                PROMPT="" # blank
            elif [ "$PROMPT_ID" -eq 1 ]; then
                PROMPT="There is a question in this image, you need to find the question and answer the question based on the visual information of the entire image."
            elif [ "$PROMPT_ID" -eq 2 ]; then
                PROMPT="There is a question in this image. You need to find the question and answer the question based on the visual information of the entire image. Please do not repeat the question and answer it directly."
            elif [ "$PROMPT_ID" -eq 3 ]; then
                PROMPT="Please answer the question in the image directly, and do not repeat the question. You need to find the answer based on the visual information of the entire image."
            elif [ "$PROMPT_ID" -eq 4 ]; then
                PROMPT="Please answer the question in the image directly, and do not repeat the question."                
            elif [ "$PROMPT_ID" -eq 5 ]; then
                PROMPT="Please find the question in the image, and answer it directly based on the whole image."                
            elif [ "$PROMPT_ID" -eq 6 ]; then
                PROMPT="Please find the question in the image, and answer it directly based on the whole image. Do not repeat the question."                
            elif [ "$PROMPT_ID" -eq 7 ]; then
                PROMPT="Please find the question in the image and answer it based on the image. Do not repeat the question you find in your answers."                
            else
                PROMPT="Unknown prompt!"
            fi

            for TASK in "${TASKS[@]}"; do

                echo "$MODEL_PATH $METHOD_FOLDER $TASK $DIRECTION $EVAL_DIR $FILTER_ANSWER $SPLIT_WORD $MODEL_TYPE"
                echo "prompt $PROMPT prompt_id $PROMPT_ID Start!"
                bash ./scripts_for_response_filtering/${TASK}_uni.sh \
                    "$MODEL_PATH" \
                    "${TASK}_${METHOD_FOLDER}_image" \
                    "$DIRECTION" \
                    "$PROMPT" \
                    "$PROMPT_ID" \
                    "$EVAL_DIR" \
                    "$FILTER_ANSWER" \
                    "$SPLIT_WORD" \
                    "$MODEL_TYPE"

            done
        done
    done
done
