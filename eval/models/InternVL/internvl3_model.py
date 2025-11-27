from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

def load_internvl3_model(args):
    torch_device = "cuda"
    model_checkpoint = args.model_path
    processor = AutoProcessor.from_pretrained(model_checkpoint)
    model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map=torch_device, torch_dtype=torch.bfloat16)

    return processor, model

def internvl3_inference(image_path_lst, qs_lst, model, processor, args):

    messages = [
        {
            "role": "user",
            "content": [
                # {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
                # {"type": "text", "text": "Please describe the image explicitly."},
                {"type": "image", "url": image_path_lst[0]},
                {"type": "text", "text": qs_lst[0]},
            ],
        }
    ]

    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

    generate_ids = model.generate(**inputs, max_new_tokens=200)
    decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    # print(f"decoded_output: {decoded_output}")
    return [decoded_output]

