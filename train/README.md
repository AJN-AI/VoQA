## VoQA training code for TinyLLaVA models (Modified based on TinyLLaVA_Factory):

[ English | [中文](./README_zh.md) ]

This project contains fine-tuning templates for various strategies as well as corresponding scripts. The specific explanations of the relevant parameters have been reflected in each script. Please modify them respectively according to your own situation.

### Train model on VoQA dataset

### 1. Dataset Preparation

Download the evaluation datasets from 🤗Hugging Face: [AJN-AI/VoQA](https://huggingface.co/datasets/AJN-AI/VoQA)

Scripts for only downloading the test split are available at Hugging Face.

### 2. Get ready for training

**Step 1:** Modify the specific training parameters in [voqa_sft.sh](voqa_sft.sh) and [scripts/train/voqa/train_qwen2_base.sh](scripts/train/voqa/train_qwen2_base.sh)

**Step 2:** Run the following command for training:

```
bash ./voqa_sft.sh
```