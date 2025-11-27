import argparse
import time

import torch
import os
import re
import random
import json
from tqdm import tqdm
import shortuuid

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *

from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def load_tinyllava_model(args):
    # if args.model_name == 'TinyLLaVA-Phi-2-SigLIP-3.1B' or args.model_name == 'TinyLLaVA-Qwen2-0.5B-SigLIP':
    args.conv_mode = 'phi'
    print(f"{args.model_name} change args.conv_mode to {args.conv_mode}.")
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)
    model.to(device='cuda')
    
    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)
    return model, tokenizer, text_processor, image_processor, context_len


# input images and prompts, output tokens
def tinyllava_inference(image_path_lst, qs_lst, image_processor, text_processor, tokenizer, model, args):
    image = Image.open(image_path_lst[0]).convert('RGB')
    image_tensor = image_processor(image)

    # qs = DEFAULT_IMAGE_TOKEN + '\n' + qs_lst[0]
    msg = Message()
    msg.add_message(qs_lst[0])
    # print(msg.messages)
    result = text_processor(msg.messages, mode='eval')
    input_ids = result['input_ids']
    
    if args.task != "scienceqa":
        image_tensor = image_tensor.unsqueeze(0)
        input_ids = input_ids.unsqueeze(0)
        # keywords = [tokenizer.eos_token]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.bfloat16, device='cuda', non_blocking=True),
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p, #
                num_beams=args.num_beams,  #
                max_new_tokens=args.max_new_tokens,
                # stopping_criteria=[stopping_criteria], #
                image_sizes=image.size,
                use_cache=True
            )
    else:
        input_ids = input_ids.unsqueeze(0).cuda()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=1024,
                image_sizes=[image.size],
                use_cache=True,
            )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return [outputs]

if __name__ == "__main__":
    disable_torch_init()
    model_path = os.path.expanduser('Zhang199/TinyLLaVA-Qwen2-0.5B-SigLIP')
    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)