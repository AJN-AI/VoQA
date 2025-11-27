# Script of the models using api, for traditional VQA evaluation, including model reasoning. 
# 1. Zero-shot model names (You can add other models here)
MODEL_NAMES=(gpt-4o)
# 2. Traditional VQA evaluation tasks
TASKS=(scienceqa pope textvqa gqa vqav2)
# 3. Traditional VQA dataset path (root path)
EVAL_DIR="path/to/VoQA/evaluation/dataset"
# 4. The api key for accessing the models. 
API_KEY="XXX"
# 5. The number of threads accessing the api simultaneously. 
THREAD_NUM=20

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for TASK in "${TASKS[@]}"; do

        echo "$MODEL_NAME $EVAL_DIR $TASK $API_KEY $THREAD_NUM Start!"

        bash ./scripts_for_traditional_vqa_api/${TASK}.sh \
            "$MODEL_NAME" \
            "$EVAL_DIR" \
            "$API_KEY" \
            "$THREAD_NUM"

    done
done
