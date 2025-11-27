import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

def load_llava_hf_model(args):
    # model_id = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path, 
        torch_dtype=torch.float16, 
        device_map='auto',
        low_cpu_mem_usage=True,
    ).to(0)
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    return model, processor

def llava_hf_inference(image_path_lst, qs_lst, model, processor, args):
    # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image") 
    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": qs_lst[0]},
            {"type": "image"},
            ],
        },
    ]
    with torch.inference_mode():
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        # print('prompt:', prompt)

        # image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
        # raw_image = Image.open(requests.get(image_file, stream=True).raw)
        raw_image = Image.open(image_path_lst[0])
        inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        # final_output = processor.decode(output[0][2:], skip_special_tokens=True)
        final_output = processor.decode(output[0][:], skip_special_tokens=True)
        # print('final_output:', final_output)
        return [final_output]


if __name__ == '__main__':
    model_path = ''
    model, processor = load_llava_hf_model(model_path)
    image_path = ''
    qs = ''
    image_path_list = []
    image_path_list.append(image_path)
    qs_list = []
    qs_list.append(qs)
    llava_hf_inference(image_path_list, qs_list, model, processor)
    
