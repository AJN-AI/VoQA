# VoQA Benchamark

## Overview

VoQA Benchmark is a comprehensive benchmark for Visual-only Question Answering (VoQA) that provides a unified evaluation framework for both open-source and closed-source models. This project integrates evaluation logic for various datasets and models, allowing for **efficient batch evaluation through simple parameter modifications**.

## Key Features

* Evaluation of open-source models on the VoQA dataset
* API-based evaluation of closed-source models on the VoQA dataset
* Evaluation of open-source models on traditional VQA datasets
* API-based evaluation of closed-source models on traditional VQA datasets
* Response filtering and accuracy calculation based on existing inference results
* Evaluation of Question-Alignment Fine-Tuning models for Question Alignment Accuracy (QAA)

## Quick Start

### 1. Dataset Preparation

1. Download the evaluation datasets: [Hidden here]

Scripts for only downloading the test split are available.

2. Modify the `EVAL_DIR` parameter in the main scripts to point to your dataset root directory

### 2. Model and Environment Setup (Open-source Models Only)

The project includes the following pre-configured models:

* TinyLLaVA\_Factory: TinyLLaVA-Phi-2-SigLIP-3.1B, TinyLLaVA-Qwen2-0.5B-SigLIP, TinyLLaVA-Qwen2.5-3B-SigLIP
* LLaVA: llava-v1.5-7b, llava-1.5-7b-hf
* Qwen: Qwen2.5-VL-3B-Instruct, Qwen2-VL-2B
* InternVL: InternVL2\_5-1B, InternVL3-1B
* DeepSeek\_VL2: deepseek-vl2-tiny
* LAVIS\_xgen\_mm: xgen-mm-phi3-mini-instruct-interleave-r-v1.5 (BLIP-3)

Each model series has its corresponding conda environment configuration. Take `TinyLLaVA_Factory` as an example:

First, switch the directory to `TinyLLaVA_Factory`:

```Shell
cd models/TinyLLaVA_Factory
```

Then execute the commands in `README.md`:

```Shell
conda create -n tinyllava_factory python=3.10 -y
conda activate tinyllava_factory
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install flash-attn --no-build-isolation
```

### 3. Evaluation Process

After preparing for the datasets, models, and conda environments, you just need to modify a few parameters in the main script. The descriptions of each parameter have been provided in each script and can be modified by referring to the existing examples.

#### Open-source Model Evaluation

* Traditional VQA Datasets: Modify `scripts_for_traditional_vqa/eval_traditional_vqa_main.sh`

```Bash
bash scripts_for_traditional_vqa/eval_traditional_vqa_main.sh
```

* VoQA Concatenation Dataset: Modify `scripts_for_voqa/eval_concatenation_for_zero_shot.sh`

```Bash
bash scripts_for_voqa/eval_concatenation_for_zero_shot.sh
```

* VoQA Watermark Dataset: Modify `scripts_for_voqa/eval_watermark_for_zero_shot.sh`

```Bash
bash scripts_for_voqa/eval_watermark_for_zero_shot.sh
```

#### Closed-source Model Evaluation

* Traditional VQA Datasets: Modify `scripts_for_traditional_vqa_api/eval_for_traditional_vqa.sh`

```Bash
bash scripts_for_traditional_vqa_api/eval_for_traditional_vqa.sh
```

* VoQA Concatenation Dataset: Modify `scripts_for_voqa_api/eval_concatenation_for_voqa.sh`

```Bash
bash scripts_for_voqa_api/eval_concatenation_for_voqa.sh
```

* VoQA Watermark Dataset: Modify `scripts_for_voqa_api/eval_watermark_for_voqa.sh`

```Bash
bash scripts_for_voqa_api/eval_watermark_for_voqa.sh
```

## Project Structure

```Plain
eval
├── eval/                                       # Core evaluation logic
│   ├── api_for_submit.py                       # Submit a single request using the api
│   ├── convert_gqa_for_eval.py                 # Evaluation functions for GQA
│   ├── convert_vqav2_for_submission.py         # Evaluation functions for VQAv2
│   ├── eval_for_api.py                         # The main function for evaluating the close-source models
│   ├── eval_main.py                            # The main function for evaluating the open-source models
│   ├── eval_pope.py                            # Evaluation functions for POPE
│   ├── eval_science_qa.py                      # Evaluation functions for SQA
│   ├── eval_textvqa.py                         # Evaluation functions for TextVQA
│   ├── load_models.py                          # Scripts for controlling the loading of different open-source models
│   ├── m4c_evaluator.py                        # Evaluation functions for TextVQA and VQAv2
│   ├── models_inference.py                     # Scripts for controlling different open-source models for reasoning
│   └── process_answer.py
├── models/                                     # Model project folders
├── scripts_for_response_filtering/             # Response filtering scripts
├── scripts_for_result_analysis/                # Question-Alignment Fine-Tuning models result analysis
├── scripts_for_traditional_vqa/                # Traditional VQA evaluation scripts
├── scripts_for_traditional_vqa_api/            # Traditional VQA API evaluation scripts
├── scripts_for_voqa/                           # VoQA evaluation scripts
└── scripts_for_voqa_api/                       # VoQA API evaluation scripts
```

## Dataset Structure

### VoQA Benchmark

The VoQA evaluation dataset includes the following tasks:

* GQA
* POPE
* ScienceQA
* TextVQA
* VQAv2

Each task contains three data formats:

* Concatenation method (with resizing)
* Concatenation method (without resizing)
* Watermark rendering

```Plain
VoQA_eval/                                      # VoQA evaluation dataset
├── gqa/                                        ########## GQA task ##########
│   ├── eval/                                   # 'eval' folder in original GQA dataset
│   ├── gqa_concat_with_resizing_image/         # concatenation method with resizing dataset folder of GQA task
│   │   ├── 2012597/                            # image id in original GQA task
│   │   │   ├── d.jpg                           # bottom concatenation with resizing image
│   │   │   ├── l.jpg                           # left concatenation with resizing image
│   │   │   ├── r.jpg                           # right concatenation with resizing image
│   │   │   └── u.jpg                           # top concatenation with resizing image
│   │   ├── 2012660/
│   │   ├── ...
│   │   └── 202287013/
│   ├── gqa_concat_without_resizing_image/      # concatenation method without resizing dataset folder of GQA task
│   │   └── ...                                 # the same structure to concatenation method with resizing
│   ├── gqa_watermark_rendering_image/          # watermark rendering dataset folder of GQA task
│   │   ├── 2012597.jpg                         # watermark rendering image
│   │   ├── 2012660.jpg
│   │   ├── ...
│   │   └── 202287013.jpg
│   ├── images/                                 # 'images' folder in original GQA dataset (Used for traditional VQA evaluation)
│   ├── llava_gqa_testdev_balanced.jsonl        # jsonl in original GQA dataset
│   └── testdev_balanced_questions.json         # json in original GQA dataset
├── pope/                                       ########## POPE task ##########
│   ├── coco/                                   # 'coco' folder in original POPE dataset
│   ├── llava_pope_test.jsonl                   # jsonl in original POPE dataset
│   ├── pope_concat_with_resizing_image/
│   ├── pope_concat_without_resizing_image/
│   ├── pope_watermark_rendering_image/
│   └── val2014/                                # 'val2014' folder in original POPE dataset (Used for traditional VQA evaluation)
├── scienceqa/                                  ########## SQA task ##########
│   ├── images/                                 # 'images' folder in original SQA dataset (Used for traditional VQA evaluation)
│   ├── llava_test_CQM-A_selected_mm.jsonl      # from jsonl in original SQA dataset, but selected part of images
│   ├── pid_splits_selected.json                # from json in original SQA dataset, but selected part of images
│   ├── problems.json                           # json in original SQA dataset
│   ├── scienceqa_concat_with_resizing_image/
│   ├── scienceqa_concat_without_resizing_image/
│   └── scienceqa_watermark_rendering_image/
├── textvqa/                                                            ########## TextVQA task ##########
│   ├── TextVQA_0.5.1_val_new_id.json                                   # from jsonl in original TextVQA dataset, but make the image id non-repetitive
│   ├── llava_textvqa_val_v051_ocr_new_id_without_ocr_reference.jsonl   # from jsonl in original TextVQA dataset, but make the image id non-repetitive and remove the ocr references
│   ├── textvqa_concat_with_resizing_image/
│   ├── textvqa_concat_without_resizing_image/
│   ├── textvqa_watermark_rendering_image/
│   └── train_images/                           # 'train_images' folder in original TextVQA dataset (Used for traditional VQA evaluation)
└── vqav2/                                      ########## VQAv2 task ##########
    ├── llava_vqav2_mscoco_test-dev2015.jsonl   # jsonl in original VQAv2 dataset
    ├── test2015/                               # 'test2015' folder in original VQAv2 dataset (Used for traditional VQA evaluation)
    ├── vqav2_concat_with_resizing_image/
    ├── vqav2_concat_without_resizing_image/
    └── vqav2_watermark_rendering_image/
```

### VoQA Training Dataset

```Plain
VoQA_train/                                      # VoQA train dataset         
├── concat_with_resizing_image/                  # concatenation method with resizing image folder of train dataset
│   ├── 000000000009/                            # image id in VoQA train dataset                        
│   │   ├── prompt_1_d.jpg                       # bottom concatenation with resizing image of prompt 1
│   │   ├── prompt_1_l.jpg                       # left concatenation with resizing image of prompt 1
│   │   ├── prompt_1_r.jpg                       # right concatenation with resizing image of prompt 1
│   │   ├── prompt_1_u.jpg                       # top concatenation with resizing image of prompt 1
│   │   ├── prompt_2_d.jpg
│   │   ├── prompt_2_l.jpg
│   │   ├── ...
│   │   └── prompt_4_u.jpg
│   ├── 000000000009_2/
│   ├── ...
│   └── VG_100K_8-2/
├── concat_without_resizing_image/               # concatenation method without resizing image folder of train dataset
│   └── ...                                      # the same structure to concatenation method with resizing
├── watermark_rendering_image/                   # watermark rendering image folder of train dataset
│   ├── 000000000009/                            # image id in VoQA train dataset                        
│   │   ├── prompt_1.jpg                         # watermark rendering image of prompt 1
│   │   ├── prompt_2.jpg
│   │   ├── prompt_3.jpg
│   │   └── prompt_4.jpg
│   ├── 000000000009_2/
│   ├── ...
│   └── VG_100K_8-2/
└── Voqa_3_3M.jsonl                              # jsonl for VoQA training
```

## Adding New Models

To add a new model for evaluation, follow these steps:

**Step 1:** Create a new model folder

**Step 2:** Configure the conda environment for your model

**Step 3:** Add new branches in `eval/load_models.py` and `eval/models_inference.py`

**Step 4:** Implement model loading and inference functions

## Additional functions

In addition to the basic evaluation code, we have also implemented two additional functions: **Response Filtering** and **Question-Alignment Fine-Tuning Result Analysis**.

### Response Filtering

If you have saved inference results and want to modify the answer processing logic, update the logic in `eval/process_answer.py`. Then modify parameters in:

* `scripts_for_response_filtering/eval_concatenation_with_filter_main.sh` for VoQA concatenation dataset

```Bash
bash scripts_for_response_filtering/eval_concatenation_with_filter_main.sh
```

* `scripts_for_response_filtering/eval_watermark_with_filter_main.sh` for VoQA watermark dataset

```Bash
bash scripts_for_response_filtering/eval_watermark_with_filter_main.sh
```

### Question-Alignment Fine-Tuning Result Analysis

To analyze the Question Alignment Accuracy (QAA) of models fine-tuned with the Question-Alignment Fine-Tuning strategies, use `scripts_for_result_analysis/analyse_main_all.sh` and modify its parameters as needed.

```Bash
bash scripts_for_result_analysis/analyse_main_all.sh
```

## License

[To be added]

## Citation

[To be added]
