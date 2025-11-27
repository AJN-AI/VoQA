from torch.cuda.amp import autocast
from packaging import version
import pathlib

import tokenizers
import transformers


from tinyllava.train.tinyllava_trainer import LLaVATrainer
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.data.dataset import make_supervised_data_module

import pdb

from tinyllava.data.text_preprocess import *
from tinyllava.data.text_image_preprocess import process_text_image
from tinyllava.data.image_preprocess import ImagePreprocess

import os
import os
# 在代码最开始设置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio

    model_args = {}
    model_args['llm'] = _load_llm_settings(model_arguments)
    model_args['vision_tower'] = _load_vision_settings(model_arguments)
    model_args['connector'] = _load_connector_settings(model_arguments) 
    model_args['text_vision_tower'] = _load_text_vision_tower_settings(model_arguments)
    model_args['text_vision_connector'] = _load_text_vision_connector_settings(model_arguments)
    return model_args

def _load_llm_settings(model_arguments):
    llm_args = {}
    llm_args['model_name_or_path'] = model_arguments.model_name_or_path
    llm_args['cache_dir'] = model_arguments.cache_dir
    llm_args['attn_implementation'] = model_arguments.attn_implementation # flash_attention_2 only supports torch.float16 and torch.bfloat16 dtypes
    return llm_args

def _load_vision_settings(model_arguments):
    vision_args = {}
    vision_args['model_name_or_path'] = model_arguments.vision_tower.split(':')[-1]
    if model_arguments.vision_tower2 != '':
        vision_args['model_name_or_path2'] = model_arguments.vision_tower2.split(':')[-1]
    return vision_args

def _load_text_vision_tower_settings(model_arguments):
    text_vision_tower_args = {}
    text_vision_tower_args['model_name_or_path'] = model_arguments.text_vision_tower.split(':')[-1]
    return text_vision_tower_args

def _load_connector_settings(model_arguments):
    connector_args = {}
    connector_args['connector_type'] = model_arguments.connector_type
    return connector_args

def _load_text_vision_connector_settings(model_arguments):
    text_vision_connector_args = {}
    text_vision_connector_args['text_vision_connector_type'] = model_arguments.text_vision_connector_type
    return text_vision_connector_args


def train():
    
    # load argument
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()

    # print(f'text vision connector: {model_arguments.text_vision_connector_type}')
    
    logger_setting(getattr(training_arguments, 'output_dir', None))

    # print(training_arguments.training_recipe)
    # print(training_arguments.output_dir)
    # print(f'conv version:{data_arguments.conv_version}')

    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments) 
    # model_args contain arguements for huggingface model .from_pretrained function
    model_args = load_settings(model_arguments, data_arguments, training_arguments)
    # 如果是预训练模型，更新加载的模型板块的地址为保存的预训练板块
    model_args = training_recipe.add_args(model_args)

    # print(f'text vision connector: {model_args.text_vision_connector_type}')
    # print(f"text vision connector: {model_args['text_vision_connector']}")

    model_config = TinyLlavaConfig()
    model_config.load_from_config(model_arguments)
    model = TinyLlavaForConditionalGenerationWithTextImage(model_config)
    
    # load pretrained checkpoint
    if training_arguments.pretrained_model_path is not None:
        model = training_recipe.load(model, model_args)
    else:
        model.load_llm(**model_args['llm'])
        model.load_vision_tower(**model_args['vision_tower'])
        model.load_text_vision_tower(**model_args['text_vision_tower'])
        model.load_connector(**model_args['connector'])
        model.load_text_vision_connector(**model_args['text_vision_connector'])

    model = training_recipe(model)
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    tokenizer = model.tokenizer
    data_arguments.image_processor = model.vision_tower._image_processor
    data_arguments.is_multimodal = True
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_arguments)
    log_trainable_params(model)  # not work well with zero3
    trainer = LLaVATrainer(model=model, #does not require model.to(device), huggingface/deepspeed does it for you?
                           tokenizer=tokenizer,
                           args=training_arguments,
                           **data_module)

    # vision_tower_params = sum(p.numel() for p in model.vision_tower.parameters())
    # text_vision_tower_params = sum(p.numel() for p in model.text_vision_tower.parameters())
    # llm_params = sum(p.numel() for p in model.language_model.parameters())
    # total_params = vision_tower_params + text_vision_tower_params + llm_params

    # print(f"视觉模型参数量: {vision_tower_params:,}")
    # print(f"文本视觉模型参数量: {text_vision_tower_params:,}") 
    # print(f"语言模型参数量: {llm_params:,}")
    # print(f"总参数量: {total_params:,}")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device=device)

    # allocated_memory = torch.cuda.memory_allocated(device)
    # reserved_memory = torch.cuda.memory_reserved(device)

    # # 打印内存使用情况（以 MB 为单位）
    # print(f"Allocated memory: {allocated_memory / (1024 ** 2):.2f} MB")
    # print(f"Reserved memory: {reserved_memory / (1024 ** 2):.2f} MB")

    # 继续训练
    trainer.train()
    
    training_recipe.save(model, trainer)


def inference():
    # load argument
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()
    
    logger_setting(getattr(training_arguments, 'output_dir', None))

    # 加载预训练的模型
    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments)
    # model_args contain arguements for huggingface model .from_pretrained function
    model_args = load_settings(model_arguments, data_arguments, training_arguments)
    model_args = training_recipe.add_args(model_args)
    model_config = TinyLlavaConfig()
    model_config.load_from_config(model_arguments)
    model = TinyLlavaForConditionalGenerationWithTextImage(model_config)

    model.eval()

    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    tokenizer = model.tokenizer
    data_arguments.image_processor = model.vision_tower._image_processor
    data_arguments.is_multimodal = True
    # tokenizer = model.tokenizer
    # data_arguments.image_processor = model.vision_tower._image_processor
    # data_arguments.is_multimodal = True
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_arguments)

    if training_arguments.pretrained_model_path is not None:
        model = training_recipe.load(model, model_args)
        # model = TinyLlavaForConditionalGenerationWithTextImage.from_pretrained(training_arguments.pretrained_model_path,low_cpu_mem_usage=True,torch_dtype=torch.bfloat16)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 需要把精度设置成half --  phi2
    # 需要把精度设置成 bf16 -- qwen2
    model = model.to(device).to(torch.bfloat16)

    # 生成模型生成需要的 input_ids
    #qs = "\n <image><image>"
    # stage 2: OCR task
    qs = '\n'

    text_processor = TextPreprocess(tokenizer, data_arguments.conv_version)
    data_args = model.config

    msg = Message()
    msg.add_message(qs)
    # print(msg.messages)

    result = text_processor(msg.messages, mode='eval')
    input_ids = result['input_ids']
    prompt = result['prompt']
    # print(f'prompt: {prompt}')
    input_ids = input_ids.unsqueeze(0).cuda()

    # 处理输入一般图片和文本图片
    image_path = '/path/to/data/example/000000569613-1/000000569613.jpg'
    text_image_path = '/path/to/example/prompt_3.jpg'
    image = Image.open(image_path).convert('RGB')
    text_image = Image.open(text_image_path).convert('RGB')

    image_tensor = ImagePreprocess(image_processor=data_arguments.image_processor, data_args=data_arguments)(image)
    image_tensor = image_tensor.unsqueeze(0).to(torch.bfloat16)  # 转换为 bf16
    text_image_tensor = process_text_image(text_image)
    text_image_tensor = text_image_tensor.unsqueeze(0).to(torch.bfloat16)  # 转换为 bf16
    # text_image_tensor = None

    # INFERENCE CODE
    stop_str = text_processor.template.separator.apply()[1]
    # stop_str = '<|endoftext|>'
    keywords = [stop_str]
    # keywords = []
    # print(f'stop str:{stop_str}')
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    temperature = 0.5
    top_p = None
    num_beams = 1
    max_new_tokens = 512
    sep = ','

    # with autocast(dtype=torch.float16):
    with torch.inference_mode():
        output_ids = model.generate(
            text_images=text_image_tensor,
            inputs=input_ids,
            images=image_tensor,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    # print(f'outputs: {output_ids}')

    output_ids = torch.tensor([[      198,  15191,   6139,    419,   2311,   5267,  16141,    279,
          3405,   1667,    264,   3175,   3409,    476,  17133,     13,  35560,
          3846,   2821,     25,  14927,    468,     13,    434,   1330,   3179,
        151643,       198,   3838,    374,    279,   2265,    315,    419,
          2311,     30,  35560,   3846,   2821,     25,  32939,   8257,  44773,
         52256,     25,  10967,   2379,    444,   2221,     11,   2585,   2379,
           444,   2221,     11,    323,    279,  56016,  10479,  19173,  47938,
        151643,       198,   3838,    943,    315,   2311,    374,    419,
            30,  35560,   3846,   2821,     25,   9965,    609,   4149, 151643,
              198,   3872,    419,   2311,   5435,    311,   9965,    609,
          4149,     30,  35560,   3846,   2821,     25,   7414, 151643,   
           198,   3872,    419,   2311,   5435,    311,   9965,    609,   4149,
            30,  35560,   3846,   2821,     25,   2308, 151643]])

    # output_ids = torch.tensor([[ 220]])
    
    outputs = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )[0]

    # print(f'outputs: {outputs}')
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    print('answer is: ')
    print(outputs)

if __name__ == "__main__":
    
    # train()
    inference()
    
