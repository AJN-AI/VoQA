## TinyLLaVA系列模型的VoQA训练代码（基于TinyLLaVA_Factory修改）：

该项目包含各种策略的微调模板以及相应的脚本，相关参数的具体说明已在各脚本中体现，请根据自身情况分别修改即可。

### 在VoQA数据集上训练模型

### 1. 数据集准备

从 🤗Hugging Face 上下载评测数据集: [AJN-AI/VoQA](https://huggingface.co/datasets/AJN-AI/VoQA)

在Hugging Face上还提供了可以仅下载测试集分割的脚本。

### 2. 准备训练

**第1步：** 请修改 [voqa_sft.sh](voqa_sft.sh) 和 [scripts/train/voqa/train_qwen2_base.sh](scripts/train/voqa/train_qwen2_base.sh) 中的具体训练参数。

**第2步：** 执行如下命令开始训练：

```
bash ./voqa_sft.sh
```