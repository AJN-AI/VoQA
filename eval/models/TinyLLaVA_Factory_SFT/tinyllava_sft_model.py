import argparse
import re
import requests
from PIL import Image
from io import BytesIO

import torch
from transformers import PreTrainedModel

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def load_tinyllava_sft_model(args):
    if args.model_name in ['TinyLLaVA-Qwen2-0.5B-SigLIP-QRA', 'TinyLLaVA-Qwen2.5-3B-SigLIP-QRA', \
                             'TinyLLaVA-Qwen2-0.5B-SigLIP-QRA-HELPER', 'TinyLLaVA-Qwen2-0.5B-SigLIP-QRA-CAT']:
        args.conv_mode = 'phi_empty_inference'
    else:
        args.conv_mode = 'phi'
    print(f"{args.model_name} change args.conv_mode to {args.conv_mode}.")
    # Model
    disable_torch_init()

    if args.model_path is not None:
        model, tokenizer, image_processor, context_len = load_pretrained_model_uni(args.model_path)
    else:
        assert args.model is not None, 'model_path or model must be provided'
        model = args.model
        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048
        tokenizer = model.tokenizer
        image_processor = model.vision_tower._image_processor

    # qs = '\n'
    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)
    model.cuda()
    return model, image_processor, text_processor, tokenizer, context_len

def tinyllava_sft_inference(image_path_lst, qs_lst, image_processor, text_processor, tokenizer, model, args):
    msg = Message()
    msg.add_message(qs_lst[0])
    result = text_processor(msg.messages, mode='eval')
    input_ids = result['input_ids']
    prompt = result['prompt']
    # print(f'prompt: {prompt}')
    # print(f'input_ids: {input_ids}')
    input_ids = input_ids.unsqueeze(0).cuda()
        
    # image_files = image_parser(args)
    images = load_images(image_path_lst)[0]
    images_tensor = image_processor(images)
    images_tensor = images_tensor.unsqueeze(0).cuda()
    images_tensor.to(torch.bfloat16)

    stop_str = text_processor.template.separator.apply()[1]
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )[0]

    outputs = outputs.strip()

    return [outputs]

