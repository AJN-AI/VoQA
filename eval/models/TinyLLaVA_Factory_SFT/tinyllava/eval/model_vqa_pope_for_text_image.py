import argparse
import time

import torch
import os
import re
import json
from tqdm import tqdm
import shortuuid

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *

from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, text_image_folder, text_processor, image_processor, text_image_processor):
        self.questions = questions
        self.image_folder = image_folder
        self.text_image_folder = text_image_folder  # 新增文本图片目录
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.text_image_processor = text_image_processor

    def __getitem__(self, index):
        line = self.questions[index]
        question_id = line['question_id']
        image_file = line["image"]
        qs = line["text"]

        # 读取普通图片
        image_path = os.path.join(self.image_folder, image_file)
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_processor(image)

        # 读取文本图片
        text_image_path = os.path.join(self.text_image_folder, f"{question_id}.jpg")
        text_image = Image.open(text_image_path).convert('RGB')
        text_image_tensor = self.text_image_processor(text_image)

        qs = '\n'
        msg = Message()
        msg.add_message(qs)
        result = self.text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']
        return input_ids, image_tensor, text_image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, text_image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    text_image_tensors = torch.stack(text_image_tensors, dim=0)  # 新增文本图片输入
    return input_ids, image_tensors, text_image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, text_image_folder, text_processor, image_processor, text_image_processor, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, text_image_folder, text_processor, image_processor, text_image_processor)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model, tokenizer, image_processor, context_len = load_pretrained_model_with_text_image(model_path)
    
    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    #model.config.image_aspect_ratio = 'pad'
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")


    data_loader = create_data_loader(questions, args.image_folder, args.text_image_folder, text_processor, image_processor, process_text_image)
    # print("Tokenizer's eos token: ", tokenizer.eos_token)
    model.to(device='cuda')
    model.to(torch.bfloat16)

    for (input_ids, image_tensor, text_image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]
        # keywords = [tokenizer.eos_token]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                text_images=text_image_tensor.to(dtype=torch.bfloat16, device='cuda', non_blocking=True),
                inputs=input_ids,
                images=image_tensor.to(dtype=torch.bfloat16, device='cuda', non_blocking=True),
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                # stopping_criteria=[stopping_criteria],
                image_sizes=image_sizes,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # print("Printing outputs")
        # print(outputs)
        # time.sleep(5)

        if args.for_benchmark:
            match = re.search(r"ASSISTANT\s*:\s*(.*)", outputs)
            if match:
                result = match.group(1)
                result.strip()
                outputs = result


        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": args.model_base,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--text-image-folder", type=str, default='')
    parser.add_argument("--conv-mode", type=str, default="llama")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    parser.add_argument("--for_benchmark", type=bool, default=True)
    args = parser.parse_args()
    

    eval_model(args)
