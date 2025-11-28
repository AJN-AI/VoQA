<h2 align="center"> <a href="http://arxiv.org/abs/2505.14227">VoQA: Visual-only Question Answering</a><h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2505.14227-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2505.14227)[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](./LICENSE)

[ [English](./README.md) | 中文 ]

##

仅视觉输入的视觉问答任务（VoQA）是一项新颖的任务，其中单个图像包含场景和问题，要求模型仅从视觉信息进行推理。以下是*传统 VQA 任务*和*VoQA 任务*的区别：

<p align="center">
    <img src="./assets/VQA and VoQA.jpg" width="600">
</p>

*VoQA 任务*对现有的大型视觉语言模型（LVLMs）提出了挑战，与传统的VQA任务相比，即使采用提示工程或者OCR系统辅助，模型也会出现显著的性能下降。

<p align="center">
    <img src="./assets/VoQA Benchmark.jpg" width="900">
</p>

为了解决这个问题，我们研究了问题对齐微调策略，旨在指导模型在推理之前对齐视觉问题。下面是*Baseline-SFT (VQA, VoQA)* 、 *QA-SFT (VoQA)* 和 *QRA-SFT (VoQA)* 的区别：

<p align="center">
    <img src="./assets/four SFT.jpg" width="750">
</p>

## 快速开始

* **VoQA数据集:** 详见🤗Hugging Face: [AJN-AI/VoQA](https://huggingface.co/datasets/AJN-AI/VoQA)

* **VoQA评测基准:** 详见`./eval`目录，具体使用方法请查看[./eval/README.md](./eval/README.md)

* **VoQA训练代码（TinyLLaVA系列模型）:** 详见`./train`目录，具体使用方法请查看[./train/README.md](./train/README.md)

## ❤️ 致谢
* 我们整个的评测代码逻辑是基于 [TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory) 项目完成的。很棒的工作!
* `./models` 部分的代码是基于如下这些项目完成的: [TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory)，[LLaVA](https://github.com/haotian-liu/LLaVA)，[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)，[InternVL](https://github.com/OpenGVLab/InternVL/tree/v2.5)，[DeepSeek-VL2](https://github.com/deepseek-ai/DeepSeek-VL2)，[BLIP-3](https://github.com/salesforce/LAVIS/tree/xgen-mm)。很棒的工作!
* 我们项目中使用的数据来源于 [ShareGPT4V](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V) 项目。很棒的工作!

## 许可

请参阅[LICENSE](./LICENSE)文件，了解提供此代码的许可协议的详细信息。
对于模型和数据集，请参考原始资源页面并遵循相应的许可。

## 引用

如果您认为我们的论文、代码或数据集对您的研究有帮助，请考虑留下星星:star:，并引用我们的论文：

```bibtex
@article{jiang2025voqa,
  title={VoQA: Visual-only Question Answering},
  author={Jiang, Luyang and An, Jianing and Luo, Jie and Wu, Wenjun and Huang, Lei},
  journal={arXiv preprint arXiv:2505.14227},
  year={2025}
}
```