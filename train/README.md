## VoQA training code for TinyLLaVA models (Modified based on TinyLLaVA_Factory):

[ English | [中文](./README_zh.md) ]

This project contains fine-tuning templates for various strategies as well as corresponding scripts. The specific explanations of the relevant parameters have been reflected in each script. Please modify them respectively according to your own situation.

### Train model on VoQA dataset
Firstly, please modify the specific training parameters in *voqa_sft.sh* and *scripts/train/voqa/train_qwen2_base.sh*.

Then, just run the following command:

```
bash ./voqa_sft.sh
```