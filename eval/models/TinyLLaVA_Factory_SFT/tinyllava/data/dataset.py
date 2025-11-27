import copy
from dataclasses import dataclass
import json
from typing import Dict,  Sequence, TYPE_CHECKING
from PIL import Image, ImageFile
import os

from .text_preprocess import TextPreprocess
from .text_image_preprocess import process_text_image
from .image_preprocess import ImagePreprocess
from ..utils.arguments import DataArguments
from ..utils.constants import *


import transformers
import torch
from torch.utils.data import Dataset

import pdb
import shutil

ImageFile.LOAD_TRUNCATED_IMAGES = True


class LazySupervisedDatasetWithTextImage(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDatasetWithTextImage, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)
        self.text_image_preprocess = process_text_image

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def convert_image_id_to_path(image_id):
        prefix = image_id[:5]
        image_path = f"{prefix}/{image_id}.jpg"
        return image_path

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        
        data_dict = self.text_preprocess(copy.deepcopy(sources["conversations"]))

        text_image_folder = self.data_args.text_image_folder

        if self.data_args.conv_version == 'pretrain':
            data_dict['text_image'] = None
        elif self.data_args.conv_version == 'phi_stage3':
            rounds = len(sources["conversations"]) // 2
            text_images = []
            for question_idx in range(rounds):
                text_image_file_id = str(sources['id'])
                text_image_file_id = text_image_file_id.replace('/','_')
                text_image_file = f"{text_image_file_id}/prompt_{question_idx+1}.jpg"
                text_image = Image.open(os.path.join(text_image_folder, text_image_file)).convert("RGB")
                text_image = self.text_image_preprocess(text_image)
                text_images.append(text_image)
            data_dict['text_image'] = torch.stack(text_images)
        else:
            if self.data_args.conv_version == 'phi_stage2':
                text_image_file = f"{sources['id']}/caption.jpg"
            else:
                text_image_file = f"{sources['id']}/prompt.jpg"
                
            text_image = Image.open(os.path.join(text_image_folder, text_image_file)).convert("RGB")
            text_image = self.text_image_preprocess(text_image)
            data_dict['text_image'] = text_image
        
        if 'image' in sources:
            # if self.data_args.conv_version == 'phi_stage3':
            #     image_file = f"{sources['id']}.jpg"
            # else:
            #     image_file = self.list_data_dict[i]['image']
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image = self.image_preprocess(image)
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            # print(f'{i}:{sources}')
            crop_size = getattr(self.data_args.image_processor, 'crop_size', getattr(self.data_args.image_processor, 'size'))
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict
 


class LazySupervisedDatasetUni(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                    tokenizer: transformers.PreTrainedTokenizer,
                    data_args: DataArguments):
        super(LazySupervisedDatasetUni, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        data_dict = self.text_preprocess(copy.deepcopy(sources["conversations"]))

        if self.data_args.conv_version == 'pretrain':
            if 'image' in sources:
                image_file = self.list_data_dict[i]['image']
                image_folder = self.data_args.image_folder
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                image = self.image_preprocess(image)
                data_dict['image'] = image
            elif self.data_args.is_multimodal:
                # image does not exist in the data, but the model is multimodal
                # print(f'{i}:{sources}')
                crop_size = getattr(self.data_args.image_processor, 'crop_size', getattr(self.data_args.image_processor, 'size'))
                data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])

            return data_dict
        
        if self.data_args.conv_version == 'pretrain_stage2':
            image_file = f"{sources['id']}/caption.jpg"
            image_folder = self.data_args.image_folder
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image = self.image_preprocess(image)
            data_dict['image'] = image 

            return data_dict


        if self.data_args.conv_version == 'phi_uni' or self.data_args.conv_version == 'phi_uni_a' or self.data_args.conv_version == 'phi_uni_qa':
            # Finetune Stage
            # At this stage, the images used for training are synthesized by 
            # combining question text and original images, which can be done 
            # through methods like concatenation or watermarking.
            id = str(self.list_data_dict[i]['id'])
            id = id.replace('/','_')

            syn_image_files = self.list_data_dict[i]['syn_images']
            syn_images = []
            syn_image_folder = os.path.join(self.data_args.image_folder, id)
            for syn_image_file in syn_image_files:
                # print(f'syn_image_file: {syn_image_file}\n')
                syn_image = Image.open(os.path.join(syn_image_folder, syn_image_file)).convert('RGB')
                syn_image = self.image_preprocess(syn_image)
                # print(f'syn image: {syn_image.shape}')
                syn_images.append(syn_image)
            
            data_dict['image'] = torch.stack(syn_images)
            # print(f"syn_images_for each data: {data_dict['image'].shape}")

            return data_dict



@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, :self.tokenizer.model_max_length]
        # FIXME: This is a hack for handling phi and stablelm, as they have the same eos, pad and unk. We want the model
        # FIXME: to predict the eos in the input ids, but we also use the id of eos to pad sequence, so we use a temp
        # FIXME: eos id first, and convert them back.
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
            
            # print(f'datacollator中的image tensor: {images}')


        if 'text_image' in instances[0]:
            text_images = [instance['text_image'] for instance in instances]

            # print(f'text images type: {type(text_images)}')
            # print(f'text images element type: {type(text_images[0])}')

            if all(x is not None and x.shape == text_images[0].shape for x in text_images):
                batch['text_images'] = torch.stack(text_images)
            else:
                batch['text_images'] = text_images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDatasetWithTextImage(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def make_supervised_data_module_for_uni_model(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDatasetUni(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
