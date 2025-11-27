from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, LogitsProcessor
import torch
import json
import PIL
import textwrap
import os


def load_blip3_model(args):
    # model_name_or_path = "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5"
    model = AutoModelForVision2Seq.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False, legacy=False)
    image_processor = AutoImageProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = model.update_special_tokens(tokenizer)
    return model, tokenizer, image_processor


def apply_prompt_template(prompt):
    s = (
                '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
                "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
                f'<|user|>\n{prompt}<|end|>\n<|assistant|>\n'
            )
    return s 


def blip3_inference(image_path_lst, qs_lst, model, tokenizer, image_processor, args):
    data = [
        {
            "image_path": image_path_lst,
            "question": qs_lst
        }
    ]

    model = model.to('cuda')
    model.eval()
    tokenizer.padding_side = "left"
    tokenizer.eos_token = '<|end|>'

    prediction_lst = []
    for sample in data:
        image_list = []
        image_sizes = []
        for fn in sample['image_path']:
            img = PIL.Image.open(fn)
            image_list.append(image_processor([img], image_aspect_ratio='anyres')["pixel_values"].to(dtype=torch.bfloat16).cuda())
            image_sizes.append(img.size)
        inputs = {
            "pixel_values": [image_list]
        }
        for query in sample['question']:
            prompt = apply_prompt_template(query)
            language_inputs = tokenizer([prompt], return_tensors="pt")
            inputs.update(language_inputs)
            # To cuda
            for name, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs[name] = value.cuda()
            with torch.inference_mode():
                generated_text = model.generate(**inputs, image_size=[image_sizes],
                                                pad_token_id=tokenizer.pad_token_id,
                                                eos_token_id=tokenizer.eos_token_id,
                                                temperature=0.05,
                                                do_sample=False, max_new_tokens=1024, top_p=None, num_beams=1,
                                                )
                prediction = tokenizer.decode(generated_text[0], skip_special_tokens=True).split("<|end|>")[0]
            prediction_lst.append(prediction)
        #     print("User: ", query)
        #     print("Assistant: ", textwrap.fill(prediction, width=100))
        # print("-"*120)

    return prediction_lst

