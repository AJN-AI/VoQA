## TinyLLaVA系列模型的VoQA训练代码（基于TinyLLaVA_Factory修改）：

该项目包含各种策略的微调模板以及相应的脚本，相关参数的具体说明已在各脚本中体现，请根据自身情况分别修改即可。

### 在VoQA数据集上训练模型
首先，请修改*voqa_sft.sh*和*scripts/train/voqa/train_qwen2_base.sh*中的具体训练参数。

然后，执行如下命令：

```
bash ./voqa_sft.sh
```