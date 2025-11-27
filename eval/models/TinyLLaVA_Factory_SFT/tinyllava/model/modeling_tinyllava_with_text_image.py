import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import sys

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import ast

import torch
import torch.utils.checkpoint
from torch import nn

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from . import LLMFactory, ConnectorFactory, VisionTowerFactory, TextVisionConnectorFactory
from .configuration_tinyllava import TinyLlavaConfig
from ..utils.constants import *
# from tinyllava.utils.data_utils import get_value_from_kwargs

from .vary_b import build_vary_vit_b

def get_value_from_kwargs(kwargs, name):
    if name in kwargs:
        return kwargs.pop(name)
    else:
        return None
    


class TinyLlavaPreTrainedModel(PreTrainedModel):
    config_class = TinyLlavaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        return self.language_model._supports_sdpa


class TinyLlavaForConditionalGenerationWithTextImage(TinyLlavaPreTrainedModel):
    def __init__(self, config: TinyLlavaConfig):
        
        super().__init__(config)

        self.language_model = LLMFactory(config.llm_model_name_or_path)[0](config.text_config)
        self.vision_tower = VisionTowerFactory(config.vision_model_name_or_path)(config.vision_config)
        self.connector = ConnectorFactory(config.connector_type)(config)
        self.text_vision_connector = TextVisionConnectorFactory(config.text_vision_connector_type)(config)
        self.text_vision_tower = build_vary_vit_b()

        (Tokenizer, post_load) = LLMFactory(config.llm_model_name_or_path)[1]
        self.tokenizer = post_load(Tokenizer.from_pretrained(
            config.tokenizer_name_or_path,
            cache_dir = config.cache_dir,
            model_max_length = config.tokenizer_model_max_length,
            padding_side = config.tokenizer_padding_side,
            use_fast = config.tokenizer_use_fast,
        ))
        self.post_init()

    
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        text_images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal_with_text_image(
                text_images,
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        
        # return self.language_model.forward(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     labels=labels,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict
        # )
        # 获取 logits
        outputs = self.language_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )

        logits = outputs.logits  # (batch_size, seq_len, vocab_size)
        loss = outputs.loss if labels is not None else None  # 计算 loss（如果有 labels）

        # 取概率最大的 token 作为预测值
        predicted_tokens = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)

        # 如果有 labels，则进行错位对齐
        aligned_pred_tokens, aligned_labels = None, None
        if labels is not None:
            aligned_pred_tokens = predicted_tokens[:, :-1]  # 去掉最后一个预测值
            aligned_labels = labels[:, 1:]  # 去掉第一个 label，使其和预测错位对齐

        valid_indices = aligned_labels != -100
        filtered_labels = aligned_labels[valid_indices]
        filtered_pred_tokens = aligned_pred_tokens[valid_indices]
        
        print(f'labels: {filtered_labels.tolist()}')
        print(f'pred_tokens: {filtered_pred_tokens.tolist()}')
    
        return outputs


    @torch.no_grad()
    def generate(
        self,
        text_images: Optional[torch.Tensor] = None,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal_with_text_image(
                text_images,
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(inputs)

        return self.language_model.generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        
    def encode_images(self, images):
        # print('In function encode_images')
        kwargs = {}
        kwargs['vision_feature_layer'] = self.config.vision_feature_layer
        kwargs['vision_feature_select_strategy'] = self.config.vision_feature_select_strategy
        image_dtype = next(self.vision_tower.parameters()).dtype  # 获取 vision_tower 的参数精度
        images = images.to(device=self.device, dtype=image_dtype)  # 将 image 的精度设置为与 vision_tower 一致
        image_features = self.vision_tower(images, **kwargs)
        image_features = self.connector(image_features)
        return image_features
    
    # Encode the text images
    def encode_text_images(self, text_images):
        text_image_type = next(self.text_vision_connector.parameters()).dtype
        text_images = text_images.to(device=self.device, dtype=text_image_type)
        text_image_features = self.text_vision_tower(text_images)
        text_image_features = text_image_features.flatten(2).permute(0, 2, 1)
        text_image_features = self.text_vision_connector(text_image_features)
        return text_image_features
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = self.language_model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
        
    
    def prepare_inputs_labels_for_multimodal_with_text_image(
        self, text_images, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):

        vision_tower = self.vision_tower
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # Encode Images
        image_features = self.encode_images(images) # 
        
        # print(f'text_images: {text_images}')

        if text_images is None or (isinstance(text_images, list) and all(img is None for img in text_images)):
            pass
        else:
            # Encode Text Images 
            image_counts = []
            all_images = []

            # 遍历每个样本的图片
            if isinstance(text_images, torch.Tensor):
                if len(text_images.shape) == 5:  # [B, N, 3, 1024, 1024]
                    batch_size = text_images.size(0)
                    for i in range(batch_size):
                        num_images = text_images[i].size(0)
                        image_counts.append(num_images)
                        all_images.extend([img for img in text_images[i]])
                else:  # [B, 3, 1024, 1024]
                    image_counts.extend([1] * text_images.size(0))
                    all_images.extend([img for img in text_images])
            else:
                for sample_images in text_images:
                    if len(sample_images.shape) == 3:  # 单张图片 [3, 1024, 1024]
                        image_counts.append(1)
                        all_images.append(sample_images)
                    else:  # 多张图片 [N, 3, 1024, 1024]
                        num_images = sample_images.size(0)
                        image_counts.append(num_images)
                        all_images.extend([img for img in sample_images])

            # 堆叠所有图片
            stacked_images = torch.stack(all_images)
            # 创建图片索引映射，用于追踪每张图片属于哪个样本
            sample_indices = []
            for sample_idx, count in enumerate(image_counts):
                sample_indices.extend([sample_idx] * count)
            sample_indices = torch.tensor(sample_indices)
        
            stacked_text_images_features = self.encode_text_images(stacked_images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        
        if input_ids == None:
            # For inference
            ...

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        def match_input_embeds_labels(input_embeds, labels, IGNORE_INDEX):
            """
            Adjust the length of the IGNORE_INDEX part in labels to match the length of input_embeds.
            The total length of labels remains unchanged.

            Parameters:
            - input_embeds: torch.Tensor, input embedding tensor with shape (N, D)
            - labels: torch.Tensor, original label tensor with shape (M,)
            - IGNORE_INDEX: int, a special marker in labels for the question part

            Returns:
            - adjusted_labels: torch.Tensor, adjusted labels with the same shape as the original labels
            - answer_labels: torch.Tensor, extracted answer part from labels (excluding IGNORE_INDEX)
            """
            # Get the length of input_embeds
            input_length = input_embeds.size(0)
            
            # Check if labels contains any IGNORE_INDEX
            ignore_mask = labels == IGNORE_INDEX
            if not ignore_mask.any():
                # If no IGNORE_INDEX found, prepend IGNORE_INDEX tokens
                ignore_tokens = torch.full((input_length,), IGNORE_INDEX, dtype=labels.dtype, device=labels.device)
                adjusted_labels = torch.cat([ignore_tokens, labels])
                answer_labels = labels
                return adjusted_labels, answer_labels
            
            # Initialize adjusted_labels as a copy of the original labels
            adjusted_labels = labels.clone()
            
            # Count the number of IGNORE_INDEX tokens
            num_ignore = ignore_mask.sum().item()
            
            if num_ignore < input_length:
                # If IGNORE_INDEX count is less than input_embeds length, pad with additional IGNORE_INDEX tokens
                extra_ignore = input_length - num_ignore
                ignore_indices = ignore_mask.nonzero(as_tuple=True)[0]
                extended_ignore = torch.full((extra_ignore,), IGNORE_INDEX, dtype=labels.dtype, device=labels.device)
                adjusted_labels = torch.cat([adjusted_labels[:ignore_indices[-1] + 1], extended_ignore, adjusted_labels[ignore_indices[-1] + 1:]])
            elif num_ignore > input_length:
                # If IGNORE_INDEX count exceeds input_embeds length, trim the excess
                ignore_indices = ignore_mask.nonzero(as_tuple=True)[0]
                excess = num_ignore - input_length
                indices_to_remove = ignore_indices[-excess:]
                adjusted_labels[indices_to_remove] = -1  # Mark as invalid for further processing
            
            # Extract the answer part from labels
            answer_labels = labels[~ignore_mask]

            return adjusted_labels, answer_labels

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        cur_text_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

            if text_images is None or (isinstance(text_images, list) and all(img is None for img in text_images)):
                # stage 1 预训练对齐 vision tower
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds = cur_image_features
                # 把labels的长度和输入文本图像的token长度做匹配
                cur_labels, answer_labels = match_input_embeds_labels(cur_input_embeds, labels[batch_idx], IGNORE_INDEX)
                # 添加 answer部分
                cur_input_embeds_2 = self.language_model.get_input_embeddings()(answer_labels)
                cur_input_embeds = torch.cat([cur_input_embeds, cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(cur_labels)
                cur_image_idx += 1
                continue

            if num_images == 0: # 用于 stage2 预训练对齐 或者 无一般图片输入的情况，input_ids中没有 <image>
                # print(f'treat correctly the training data for pretrain-stage-2')
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = stacked_text_images_features[batch_idx]
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                # 把labels的长度和输入文本图像的token长度做匹配
                cur_labels, answer_labels = match_input_embeds_labels(cur_input_embeds, labels[batch_idx], IGNORE_INDEX)
                # 因为现在的输入只包含问题部分，所以手动添加回答部分的input_embeddings
                cur_input_embeds_2 = self.language_model.get_input_embeddings()(answer_labels)
                cur_input_embeds = torch.cat([cur_input_embeds, cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(cur_labels)
                cur_image_idx += 1
                continue
            
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])

            cur_new_input_embeds = []
            cur_new_labels = []
            if len(image_token_indices) == 3: 
                # 一组数据只包含单张 image的情况，所以 len(image_token_indices) 应该为3, cur_labels_noim 应该只包含两个元素
                cur_image_features = image_features[batch_idx]
                # 判断text_images是否为空tensor
                if text_images.shape[0] == 0: # stage 1 预训练 用于对齐 一般VT
                    cur_input_embeds = cur_image_features
                else: # 正常单张一般图像输入加问题图片，并且问题图片没有用 <image>表示
                    # print(f'treat correctly the training data for pretrain-stage-2')
                    cur_input_embeds_1 = stacked_text_images_features[batch_idx]
                    cur_input_embeds = torch.cat([cur_image_features, cur_input_embeds_1], dim=0)
                
                cur_labels_1, answer_labels = match_input_embeds_labels(cur_input_embeds, cur_labels_noim[0], IGNORE_INDEX) # 把问题部分进行对齐，同时添加 image features, 理论上应该全部为 IGNORE INDEX
                cur_new_labels.append(cur_labels_1)
                cur_new_input_embeds.append(cur_input_embeds)
                cur_input_embeds_2 = self.language_model.get_input_embeddings()(cur_input_ids_noim[1]) # 把回答部分添加到input_embeds中，并且更新 cur_new_labels
                
                cur_new_input_embeds.append(cur_input_embeds_2)
                cur_new_labels.append(cur_labels_noim[1])
            else:
                # 多轮问答的情况，包含一张image和多个text_image
                # 一般图像和文本图像都用 <image>代替了，但是处理的时候默认只有第一个<image>是一般图像，剩下的<image>都是文本图像
                # print('treat correctly the training data for finetune')
                split_sizes = [x.shape[0] for x in cur_labels_noim]
                cur_input_embeds = self.language_model.get_input_embeddings()(torch.cat(cur_input_ids_noim))
                cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            
                for i in range(num_images + 1):
                    cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                    cur_new_labels.append(cur_labels_noim[i])
                    if i < num_images:
                        if i == 0: # 第一张图片是一般图片
                            cur_image_features = image_features[cur_image_idx]
                            cur_image_idx += 1
                            cur_new_input_embeds.append(cur_image_features)
                            cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        else: # 后面的图片是问题的文本图片
                            assert sample_indices[cur_text_image_idx] == batch_idx, f"Text image index mismatch: sample_indices[{cur_text_image_idx}]={sample_indices[cur_text_image_idx]} != batch_idx={batch_idx}"
                            cur_text_image_features = stacked_text_images_features[cur_text_image_idx]
                            cur_text_image_idx += 1
                            cur_new_input_embeds.append(cur_text_image_features)
                            cur_new_labels.append(torch.full((cur_text_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            cur_len_labels = cur_new_labels.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len_labels > 0:
                    new_labels_padded[i, -cur_len_labels:] = cur_new_labels
                    attention_mask[i, -cur_len_labels:] = True
                    position_ids[i, -cur_len_labels:] = torch.arange(0, cur_len_labels, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len_labels > 0:
                    new_labels_padded[i, :cur_len_labels] = cur_new_labels
                    attention_mask[i, :cur_len_labels] = True
                    position_ids[i, :cur_len_labels] = torch.arange(0, cur_len_labels, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    
    
    def load_llm(self, **kwargs):
        language_model_name = get_value_from_kwargs(kwargs, 'model_name_or_path')
        pretrained_llm_path = get_value_from_kwargs(kwargs, 'pretrained_llm_path')
        if pretrained_llm_path is not None:
            language_model_name = pretrained_llm_path
        if language_model_name is not None:
            self.language_model = self.language_model.from_pretrained(
                language_model_name, **kwargs
            )
        print('loading language model from ', language_model_name)
        self.language_model.requires_grad_(False)
        
        self.config.text_config.torch_dtype = kwargs.get('torch_dtype', None)
        self.config.pad_token = getattr(self.tokenizer, 'pad_token', None)
        self.config.pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
        #self.config.tokenizer_padding_side = getattr(self.tokenizer, 'padding_side', None)
        #self.config.tokenizer_model_max_length =  getattr(self.tokenizer, 'model_max_length', None)
        
        
    def load_vision_tower(self, **kwargs):
        vision_tower_name = get_value_from_kwargs(kwargs, 'model_name_or_path')
        self.vision_tower.load_model(vision_tower_name, **kwargs)
        self.vision_tower = self.vision_tower.half()

    def load_text_vision_tower(self, **kwargs):
        text_vision_tower_name = get_value_from_kwargs(kwargs, 'model_name_or_path')
        pretrained_text_vision_tower_path = get_value_from_kwargs(kwargs, 'pretrained_text_vision_tower_path')
        if pretrained_text_vision_tower_path is not None:
            text_vision_tower_name = pretrained_text_vision_tower_path
            text_vision_tower_name = os.path.join(text_vision_tower_name, 'pytorch_model.bin')
        vision_tower_high_weights = torch.load(text_vision_tower_name)
        self.text_vision_tower.load_state_dict(vision_tower_high_weights)
        print(f'Loading text vision tower from {text_vision_tower_name}')
        self.text_vision_tower.requires_grad_(False)
        
    def load_connector(self, **kwargs):
        self.connector.load_model(**kwargs)

    def load_text_vision_connector(self, **kwargs):
        self.text_vision_connector.load_model(**kwargs)