## VoQA 评测基准

VoQA 评测基准（VoQA Benchmark）是一个针对仅视觉输入的视觉问答任务（VoQA）的综合基准，它为开源和闭源模型提供了统一的评测框架。该项目集成了各种数据集和模型的评估逻辑，允许**通过简单的参数修改进行高效的批量评估**。

## 关键特征

* 在 VoQA 数据集上评估开源模型
* 基于 api 的 VoQA 数据集闭源模型评估
* 在传统 VQA 数据集上评估开源模型
* 基于 api 的传统 VQA 数据集闭源模型评估
* 基于现有推理结果的回答筛选和准确率计算
* 评估基于问题对齐策略微调的模型对问题的对齐精度（QAA）

## 快速开始

### 1. 数据集准备

**第1步：** 从 🤗Hugging Face 上下载评测数据集: [AJN-AI/VoQA](https://huggingface.co/datasets/AJN-AI/VoQA)

在Hugging Face上还提供了可以仅下载测试集分割的脚本。

**第2步：** 修改主脚本中的 `EVAL_DIR` 参数，使其指向评测集的根目录

### 2. 模型和环境准备

**对于开源模型**，项目中包括以下预配置好的模型：

* TinyLLaVA\_Factory: TinyLLaVA-Phi-2-SigLIP-3.1B, TinyLLaVA-Qwen2-0.5B-SigLIP, TinyLLaVA-Qwen2.5-3B-SigLIP
* LLaVA: llava-v1.5-7b, llava-1.5-7b-hf
* Qwen: Qwen2.5-VL-3B-Instruct, Qwen2-VL-2B
* InternVL: InternVL2\_5-1B, InternVL3-1B
* DeepSeek\_VL2: deepseek-vl2-tiny
* LAVIS\_xgen\_mm: xgen-mm-phi3-mini-instruct-interleave-r-v1.5 (BLIP-3)

每个系列的模型都有相应的 conda 环境配置。以`TinyLLaVA_Factory`为例：

首先，切换到`TinyLLaVA_Factory`目录:

```Shell
cd models/TinyLLaVA_Factory
```

然后，执行 `README.md`中的命令:

```Shell
conda create -n tinyllava_factory python=3.10 -y
conda activate tinyllava_factory
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install flash-attn --no-build-isolation
```

**对于所有模型，您需要在您的环境中安装一些必要的包。在根目录下执行如下命令：**
```Shell
pip install -r requirements.txt
```

### 3. 评测过程

在准备好数据集、模型和 conda 环境之后，您只需要修改主脚本中的几个参数。每个脚本中都提供了对每个参数的描述，您可以通过参考现有示例进行相应的修改。

#### 开源模型评测

* 传统 VQA 数据集: 修改 `scripts_for_traditional_vqa/eval_traditional_vqa_main.sh`

```Bash
bash scripts_for_traditional_vqa/eval_traditional_vqa_main.sh
```

* VoQA 拼接数据集: 修改 `scripts_for_voqa/eval_concatenation_for_zero_shot.sh`

```Bash
bash scripts_for_voqa/eval_concatenation_for_zero_shot.sh
```

* VoQA 水印数据集: 修改 `scripts_for_voqa/eval_watermark_for_zero_shot.sh`

```Bash
bash scripts_for_voqa/eval_watermark_for_zero_shot.sh
```

#### 闭源模型评测

* 传统 VQA 数据集: 修改 `scripts_for_traditional_vqa_api/eval_for_traditional_vqa.sh`

```Bash
bash scripts_for_traditional_vqa_api/eval_for_traditional_vqa.sh
```

* VoQA 拼接数据集: 修改 `scripts_for_voqa_api/eval_concatenation_for_voqa.sh`

```Bash
bash scripts_for_voqa_api/eval_concatenation_for_voqa.sh
```

* VoQA 水印数据集: 修改 `scripts_for_voqa_api/eval_watermark_for_voqa.sh`

```Bash
bash scripts_for_voqa_api/eval_watermark_for_voqa.sh
```

## 项目结构

```Plain
VoQA-code/
├── eval/                                       # 核心的评测逻辑
│   ├── api_for_submit.py                       # 使用api发送一次请求
│   ├── convert_gqa_for_eval.py                 # GQA 的评测函数
│   ├── convert_vqav2_for_submission.py         # VQAv2 的评测函数
│   ├── eval_for_api.py                         # 闭源模型评测的主函数
│   ├── eval_main.py                            # 开源模型评测的主函数
│   ├── eval_pope.py                            # POPE 的评测函数
│   ├── eval_science_qa.py                      # SQA 的评测函数
│   ├── eval_textvqa.py                         # TextVQA 的评测函数
│   ├── load_models.py                          # 控制不同开源模型加载的脚本
│   ├── m4c_evaluator.py                        # TextVQA and VQAv2 的评测函数
│   ├── models_inference.py                     # 控制不同开源模型推理的脚本
│   └── process_answer.py                       # 筛选回答的脚本
├── models/                                     # 不同模型项目的文件夹
├── scripts_for_response_filtering/             # 包含回答过滤的脚本
├── scripts_for_result_analysis/                # 基于问题对齐微调模型结果分析脚本
├── scripts_for_traditional_vqa/                # 传统 VQA 任务的评测脚本
├── scripts_for_traditional_vqa_api/            # 传统 VQA 任务基于 API 的评测脚本
├── scripts_for_voqa/                           # VoQA 任务的评测脚本
└── scripts_for_voqa_api/                       # VoQA 任务基于 API 的评测脚本
```

## 添加新的模型

您可以参考如下步骤添加一个新的模型：

**第1步：** 创建一个新模型的文件夹；

**第2步：** 为您的模型配置相应的 conda 环境；

**第3步：** 在 `eval/load_models.py` 和 `eval/models_inference.py` 中添加新模型对应的分支；

**第4步：** 实现模型加载和模型推理的函数。

## 其他功能

除了基础的评测代码，我们还实现了两个额外的功能：**回答筛选**和**基于问题对齐微调模型结果分析**：

### 回答筛选

评测代码会自动保存模型的完整推理结果，如果您希望修改回答筛选部分的处理逻辑，请更新`eval/process_answer.py`中的逻辑，然后修改对应脚本中的参数：

* `scripts_for_response_filtering/eval_concatenation_with_filter_main.sh` 用于 VoQA 拼接数据集

```Bash
bash scripts_for_response_filtering/eval_concatenation_with_filter_main.sh
```

* `scripts_for_response_filtering/eval_watermark_with_filter_main.sh` 用于 VoQA 水印数据集

```Bash
bash scripts_for_response_filtering/eval_watermark_with_filter_main.sh
```

### 基于问题对齐微调模型结果分析

要分析使用问题对齐微调策略微调的模型的问题对齐精度（QAA），请使用`scripts_for_result_analysis/analyse_main_all.sh`并根据需要修改其参数。

```Bash
bash scripts_for_result_analysis/analyse_main_all.sh
```