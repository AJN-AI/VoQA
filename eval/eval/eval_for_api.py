import argparse
import time
import torch
from torch.utils.data import Dataset, DataLoader
import os
import re
import random
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
import math
from process_answer import str2bool
from api_for_submit import post_data
from concurrent.futures import ThreadPoolExecutor

# Ensure that the model path can be imported correctly
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

def split_questions(questions, n):
    split_size = math.ceil(len(questions) / n)  # integer division
    return [questions[i:i+split_size] for i in range(0, len(questions), split_size)]

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, direction, qs, task, single_pred_prompt, model_name, model_config, original_benchmark):
        self.questions = questions
        self.image_folder = image_folder
        self.direction = direction
        self.qs = qs
        self.task = task
        self.single_pred_prompt = single_pred_prompt
        self.model_name = model_name
        self.model_config = model_config
        self.original_benchmark = original_benchmark

    def __getitem__(self, index):
        line = self.questions[index]
        if self.task == "scienceqa":
            idx = line["id"]
            cur_prompt = line['conversations'][0]['value']
            if self.original_benchmark:
                qs = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n'
            cur_prompt = cur_prompt + '\n'

            if self.single_pred_prompt:
                cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."
        else:
            idx = line["question_id"]
            cur_prompt = line["text"]
            if self.original_benchmark:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + cur_prompt
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' 
            cur_prompt = cur_prompt + '\n'
        if self.original_benchmark:
            # original benchmark
            image_path = os.path.join(self.image_folder, line["image"])
        else:
            # Concat Image
            if self.direction != 'no':
                image_path = os.path.join(self.image_folder, str(idx), f'{self.direction}.jpg')
            # Watermark Image
            else:
                image_path = os.path.join(self.image_folder, f'{idx}.jpg')

        return idx, qs, cur_prompt, image_path

    def __len__(self):
        return len(self.questions)


def custom_collate_fn(batch):    
    idx, qs, cur_prompt, image_path = zip(*batch)
    return {
        "idx": list(idx),
        "qs": list(qs),
        "cur_prompt": list(cur_prompt),
        "image_path": list(image_path)
    }


def create_data_loader(questions, args, num_workers=4):
    if args.model_name != 'InternVL2_5-1B':
        assert args.batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, args.image_folder, args.direction, args.prompt, args.task, args.single_pred_prompt, args.model_name, args.model_config, args.original_benchmark)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=num_workers, shuffle=False, collate_fn=custom_collate_fn, drop_last=False)
    return data_loader


# Processing related to Multithreading. High-concurrency task submission
def submit_batch_ordered(questions_lst, args):
    with ThreadPoolExecutor(max_workers=args.thread_num) as executor:
        for i in range(args.thread_num):
            executor.submit(preprocess_and_inference, questions_lst[i], i, args)


# Processing logic of each thread
def preprocess_and_inference(questions, thread_id, args):
    answers_file = os.path.expanduser(os.path.join(args.answers_path, f"thread_{thread_id}.jsonl"))
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    data_loader = create_data_loader(questions, args)
    # print("Tokenizer's eos token: ", tokenizer.eos_token)

    # Inference
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        # print(batch)
        idx_lst, qs_lst, cur_prompt_lst, image_path_lst = batch["idx"], batch["qs"], batch["cur_prompt"], batch["image_path"]

        # You can modify your model inference functions here. 
        output_lst = []
        whole_output_lst = []
        for image_path, qs in zip(image_path_lst, qs_lst):
            # Single-upload inference, The functions called here need to be modified according to the actual situation
            output, whole_output = post_data(args.API_KEY, args.model_name, image_path, qs)
            output_lst.append(output)
            whole_output_lst.append(whole_output)

        # save batch outputs
        for idx, cur_prompt, outputs, whole_output in zip(idx_lst, cur_prompt_lst, output_lst, whole_output_lst):
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "answer_id": ans_id,
                                    "model_id": args.model_name,
                                    "metadata": {},
                                    "whole_output": whole_output}) + "\n")
        if i % 10 == 0:
            ans_file.flush()

    ans_file.close()
    print(f"The thread {thread_id} has been completed!")


# Choose diffirent models to evaluate 
def eval_model(args):
    # Load data
    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    with open(os.path.expanduser(args.question_file), "r", encoding="utf-8") as f:
        questions = [json.loads(line.strip()) for line in f if line.strip()]

    # The first step is to divide the problem based on the number of threads
    questions_lst = split_questions(questions, args.thread_num)

    # The second step is that each multi-thread loads data independently, uploads it to the website respectively for reasoning, and saves the results
    submit_batch_ordered(questions_lst, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ##### tinyllava #####
    # for all task
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--image_aspect_ratio", type=str, default='pad')
    # only for vqav2, gqa, pope, textvqa
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    # only for sqa
    parser.add_argument("--answer-prompter", action="store_true") # only for sqa
    parser.add_argument("--single-pred-prompt", action="store_true") # only for sqa

    ##### internvl-2_5 #####
    parser.add_argument("--batch-size", type=int, default=1, help="inference batch size")

    ##### llava #####
    parser.add_argument("--model-config", type=json.loads)

    # contrast experiment
    # parser.add_argument("--filter_answer", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--original_benchmark", type=str2bool, nargs="?", const=True, default=True)

    parser.add_argument("--direction", type=str, default=None, choices=['no', 'l', 'r', 'u', 'd'], help="Specify a direction ('l', 'r', 'u', 'd', 'no'), 'no' is used in watermark datasets.")
    parser.add_argument("--prompt", type=str, default='', help="Prompts that needs to be added behind the original question.")
    parser.add_argument("--model-name", type=str, default='gpt-4o')
    parser.add_argument("--task", type=str, help="dataset name")

    # for api
    parser.add_argument("--thread_num", type=int, default=20)
    parser.add_argument("--API_KEY", type=str, default='sk-iK8szMzbPC7F9Kmoj4SX8YntvXTjBVX1W8Ig1pwOp2D2Hrbs')
    parser.add_argument("--answers_path", type=str)

    args = parser.parse_args()

    eval_model(args)
