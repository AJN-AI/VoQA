# Script for traditional VQA evaluation, including model inferencing. 
# Please set all the parameters as follows:
# 1. Zero-shot model path (You can add other models here)
MODEL_PATHS=(Zhang199/TinyLLaVA-Qwen2-0.5B-SigLIP tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B liuhaotian/llava-v1.5-7b Qwen/Qwen2.5-VL-3B-Instruct OpenGVLab/InternVL2_5-1B deepseek-ai/deepseek-vl2-tiny Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5)
# 2. Traditional VQA evaluation tasks
TASKS=(scienceqa pope textvqa gqa vqav2)
# 3. VQA dataset path (root path)
EVAL_DIR="path/to/VoQA/evaluation/dataset"
# 4. Initialize the conda environment, which is ready for step 5 below (about line 15).
# (Just for an example. Adjust the path according to the actual situation of your computer).
source /opt/miniconda3/etc/profile.d/conda.sh

for MODEL_PATH in "${MODEL_PATHS[@]}"; do

    # 5. Select the conda environment corresponding to the model, which avoids manually activating the conda environment.
    # (Just for examples. Adjust the conda environment name according to the actual situation of your computer).
    if [ "$MODEL_PATH" = "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B" ]; then
        conda activate tinyllava
    elif [ "$MODEL_PATH" = "liuhaotian/llava-v1.5-7b" ]; then
        conda activate llava
    elif [ "$MODEL_PATH" = "Qwen/Qwen2.5-VL-3B-Instruct" ]; then
        conda activate qwen
    elif [ "$MODEL_PATH" = "OpenGVLab/InternVL2_5-1B" ] ; then
        conda activate internvl
    elif [ "$MODEL_PATH" = "deepseek-ai/deepseek-vl2-tiny" ]; then
        conda activate deepseek
    elif [ "$MODEL_PATH" = "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5" ]; then
        conda activate blip3
    else
        echo "Unknown model path: $MODEL_PATH!"
    fi

    for TASK in "${TASKS[@]}"; do

        echo "$TASK $MODEL_PATH $EVAL_DIR Start!"
        bash ./scripts_for_traditional_vqa/${TASK}.sh \
            "$MODEL_PATH" \
            "$EVAL_DIR" 

    done
done
