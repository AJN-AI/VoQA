#!/bin/bash

# 1. Activate conda environment
## Initialize the conda environment (Just for an example. Adjust the path according to the actual situation of your computer)
source /opt/miniconda3/etc/profile.d/conda.sh 
conda init || echo "conda init failed!"
conda activate /path/to/VoQA_train_conda_environment || echo "conda activate failed!"
echo "Current conda environment: $(echo $CONDA_DEFAULT_ENV)"

# 2. Switch to VoQA training code folder 
cd /path/to/VoQA_training_code || { echo "Failed to change directory"; exit 1; }
echo "Current directory: $(pwd)"

# 3. Set CUDA_VISIBLE_DEVICES for training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "CUDA_VISIBLE_DEVICES set to $CUDA_VISIBLE_DEVICES"

export HF_ENDPOINT=https://hf-mirror.com

# 4. Start the training script at regular intervals
echo "Script started at $(date)"
sleep 0h
echo "Script resumed at $(date)"

# 5. Training
echo "Starting training..."
bash scripts/train/voqa/train_qwen2_base.sh # Please modify the specific training parameters in this script

echo "Training completed at $(date)"
echo "Script completed successfully."

exit 0