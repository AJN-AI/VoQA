FINETUNE_DATA_PATH=/path/to/VoQA_training_json_file
FINETUNE_IMAGE_PATH=/path/to/VoQA_training_images

LLM_VERSION=Qwen/Qwen2-0.5B # llm path in huggingface
VT_VERSION=google/siglip-so400m-patch14-384 #vision tower path in huggingface

VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc

# VoQA SFT choices for CONV_VERSION:
# phi_baseline (for Baseline-SFT), phi_qa (for QA-SFT), phi_r_qra (for R-QRA-SFT), phi_qa_only (for QA-only),
# phi_qra (for QRA-SFT), phi_qra_cat (for QRA-SFT, CAT), phi_qra_helper (for QRA-SFT, HELPER)
CONV_VERSION=phi_baseline #chat template, other options for VoQA are: phi_baseline, phi_qa_only, phi_qa, phi_qra_cat, phi_qra_helper, phi_qra, phi_r_qra

VERSION=qwen2-0_5b_base #experiment name for recording different runnings
TRAIN_RECIPE=common_uni #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=2048 #max model length for llm

PRETRAINED_MODEL_PATH=/path/to/tiny-llava-qwen2-0.5B-Siglip-pretrain
OUTPUT_DIR=/path/to/VoQA_SFT_model_path

bash scripts/train/voqa/finetune_qwen2.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$PRETRAINED_MODEL_PATH" "$OUTPUT_DIR"
