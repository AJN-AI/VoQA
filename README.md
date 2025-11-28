<h2 align="center"> <a href="http://arxiv.org/abs/2505.14227">VoQA: Visual-only Question Answering</a><h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2505.14227-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2505.14227)[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](./LICENSE)

[ English | [中文](./README_zh.md) ]

##

Visual-only Question Answering (VoQA) is a novel task where a single image contains both the scene and the question, requiring models to reason solely from visual information. Here is the difference between *Traditional VQA Task* and *VoQA Task*:

<p align="center">
    <img src="./assets/VQA and VoQA.jpg" width="600">
</p>

The *VoQA task* poses substantial challenges for current LVLMs, which show clear performance drops even with prompt engineering or OCR assistance compared to traditional VQA.

<p align="center">
    <img src="./assets/VoQA Benchmark.jpg" width="900">
</p>

To address this, we investigate question-alignment fine-tuning strategies designed to guide models toward interpreting the visual question prior to reasoning. Here is the difference between *Baseline-SFT (VQA, VoQA)*, *QA-SFT (VoQA)* and *QRA-SFT (VoQA)*:

<p align="center">
    <img src="./assets/four SFT.jpg" width="900">
</p>

## Quick start

* **VoQA dataset:** See 🤗 Hugging Face: [AJN-AI/VoQA](https://huggingface.co/datasets/AJN-AI/VoQA)

* **VoQA Benchmark:** For details, please refer to the `./eval` directory. For specific usage methods, please check [./eval/README.md](./eval/README.md).

* **VoQA training code (TinyLLaVA models):** For details, please refer to the `./train` directory. For specific usage methods, please check [./train/README.md](./train/README.md).

## ❤️ Acknowledgement
* Our whole evaluation codebase is built upon the [TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory) project. Great work!
* `./models` part of our evaluation codebase is built upon the projects as follows: [TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory), [LLaVA](https://github.com/haotian-liu/LLaVA), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [InternVL](https://github.com/OpenGVLab/InternVL/tree/v2.5), [DeepSeek-VL2](https://github.com/deepseek-ai/DeepSeek-VL2), [BLIP-3](https://github.com/salesforce/LAVIS/tree/xgen-mm). Great works!
* Our project uses data from the [ShareGPT4V](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V) project. Great work!

## License

See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.
For models and datasets, please refer to the original resource page and follow the corresponding License.

## Citation

If you find our paper, code or datasets helpful in your research, please consider giving a star :star: and kindly cite as:
```bibtex
@article{jiang2025voqa,
  title={VoQA: Visual-only Question Answering},
  author={Jiang, Luyang and An, Jianing and Luo, Jie and Wu, Wenjun and Huang, Lei},
  journal={arXiv preprint arXiv:2505.14227},
  year={2025}
}
```