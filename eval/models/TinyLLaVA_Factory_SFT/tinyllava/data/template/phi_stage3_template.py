from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

from .formatter import EmptyFormatter, StringFormatter
from .base import Template
from .formatter import Formatter
from . import register_template

from ...utils.constants import *

from transformers import PreTrainedTokenizer
import torch
import copy
    
system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

@register_template('phi_stage3')
@dataclass
class PhiStage3Template(Template):
    format_image_token: "Formatter" = StringFormatter(slot="<image>\n{{content}}")
    format_user: "Formatter" = StringFormatter(slot="USER" + ": " + "<image>" + "{{content}}" + " ")
    format_assistant: "Formatter" = StringFormatter(slot="ASSISTANT" + ": " + "{{content}}" + "<|endoftext|>")
    system: "Formatter" = EmptyFormatter(slot=system+" ")
    separator: "Formatter" = EmptyFormatter(slot=[' ASSISTANT: ', '<|endoftext|>'])

    # 只在问题和回答上求损失
    # def _make_masks(self, labels, tokenizer, sep, eos_token_length, rounds):
    #     cur_len = 0
    #     first_round = True

    #     for rou in rounds:
    #         if not rou.strip():
    #             break

    #         # 处理系统提示词（仅第一个有效轮次前的内容）
    #         if first_round:
    #             # 检查是否包含完整的用户标签
    #             if "USER: " not in rou:
    #                 # 整个轮次视为系统提示词
    #                 system_len = len(self.tokenizer_image_token(rou, tokenizer))
    #                 labels[cur_len : cur_len + system_len] = IGNORE_INDEX
    #                 cur_len += system_len
    #                 continue

    #             # 分割系统提示和对话内容（确保使用 "USER: " 作为分隔符）
    #             system_part = rou.split("USER: ")[0]

    #             # print(f'system part: {system_part}')

    #             system_len = len(self.tokenizer_image_token(system_part, tokenizer)) - 1
    #             # print(f'original label: {labels[cur_len : cur_len + system_len]}')
    #             labels[cur_len : cur_len + system_len] = IGNORE_INDEX
    #             # print(f'mask: {cur_len} - {cur_len + system_len}')
    #             cur_len += system_len

    #             # 截取剩余对话内容作为当前轮次
    #             rou = rou[len(system_part):].lstrip()
    #             # print(f'rou: {rou}')
    #             first_round = False

    #         # 分割用户和助手内容（使用 sep="ASSISTANT: "）
    #         parts = rou.split(sep)
    #         if len(parts) != 2:
    #             break  # 格式错误，终止处理

    #         user_part, assistant_part = parts

    #         # 处理用户部分（忽略 "USER: " 标签）
    #         user_prompt = "USER: "  # 明确标签格式
    #         user_prompt_tokens = self.tokenizer_image_token(user_prompt, tokenizer)
    #         user_prompt_len = len(user_prompt_tokens)

    #         # print(f'user_prompt_len: {user_prompt_len}')

    #         # 检查用户部分是否以 "USER: " 开头
    #         if not user_part.startswith(user_prompt):
    #             break  # 格式错误，终止处理

    #         # 截取实际用户内容（去掉 "USER: "）
    #         user_content = user_part[len(user_prompt):]

    #         # print(f'user content: {user_content}')

    #         user_content_tokens = self.tokenizer_image_token(user_content, tokenizer)
    #         user_total_len = user_prompt_len + len(user_content_tokens)

    #         # 忽略 "USER: " 部分，保留用户内容
    #         # print(f'original label user: {labels[cur_len : cur_len + user_prompt_len]}')
    #         labels[cur_len : cur_len + user_prompt_len] = IGNORE_INDEX
    #         # print(f'mask: {cur_len} - {cur_len + user_prompt_len}')
    #         cur_len += user_total_len  # 用户部分总长度（含 "USER: "）

    #         # 处理助手部分（包含 sep 并忽略 "ASSISTANT: "）
    #         assistant_part_with_sep = sep + assistant_part  # 添加 "ASSISTANT: " 到助手内容前
    #         assistant_tokens = self.tokenizer_image_token(assistant_part_with_sep, tokenizer)
    #         assistant_label_len = len(self.tokenizer_image_token(sep, tokenizer)) - 1 # 计算 "ASSISTANT: " 的 token 长度

    #         # 忽略助手标签部分
    #         labels[cur_len : cur_len + assistant_label_len] = IGNORE_INDEX
    #         cur_len += len(assistant_tokens) + eos_token_length  # 包含 EOS token

    #     # 剩余部分全部忽略
    #     labels[cur_len:] = IGNORE_INDEX
    #     return labels, cur_len
    # 忽略除 "ASSISTANT: " 以外的提示词，在 问题 + "ASSISTANT: " + 回答上求loss
    # 相当于给模型一个引导，先让模型看懂图片里的问题，然后再生成 "ASSISTANT: "来生成回答，类似于一个思考的过程
    def _make_masks(self, labels, tokenizer, sep, eos_token_length, rounds):
        cur_len = 0
        first_round = True

        for rou in rounds:
            if not rou.strip():
                break

            # 处理系统提示词（仅第一个有效轮次前的内容）
            if first_round:
                # 检查是否包含完整的用户标签
                if "USER: " not in rou:
                    # 整个轮次视为系统提示词
                    system_len = len(self.tokenizer_image_token(rou, tokenizer))
                    labels[cur_len : cur_len + system_len] = IGNORE_INDEX
                    cur_len += system_len
                    continue

                # 分割系统提示和对话内容（确保使用 "USER: " 作为分隔符）
                system_part = rou.split("USER: ")[0]

                # print(f'system part: {system_part}')

                system_len = len(self.tokenizer_image_token(system_part, tokenizer)) - 1
                # print(f'original label: {labels[cur_len : cur_len + system_len]}')
                labels[cur_len : cur_len + system_len] = IGNORE_INDEX
                # print(f'mask: {cur_len} - {cur_len + system_len}')
                cur_len += system_len

                # 截取剩余对话内容作为当前轮次
                rou = rou[len(system_part):].lstrip()
                # print(f'rou: {rou}')
                first_round = False

            # 分割用户和助手内容（使用 sep="ASSISTANT: "）
            parts = rou.split(sep)
            if len(parts) != 2:
                break  # 格式错误，终止处理

            user_part, assistant_part = parts

            # 处理用户部分（忽略 "USER: " 标签）
            user_prompt = "USER: "  # 明确标签格式
            user_prompt_tokens = self.tokenizer_image_token(user_prompt, tokenizer)
            user_prompt_len = len(user_prompt_tokens)

            # print(f'user_prompt_len: {user_prompt_len}')

            # 检查用户部分是否以 "USER: " 开头
            if not user_part.startswith(user_prompt):
                break  # 格式错误，终止处理

            # 截取实际用户内容（去掉 "USER: "）
            user_content = user_part[len(user_prompt):]

            # print(f'user content: {user_content}')

            user_content_tokens = self.tokenizer_image_token(user_content, tokenizer)
            user_total_len = user_prompt_len + len(user_content_tokens)

            # 忽略 "USER: " 部分，保留用户内容
            # print(f'original label user: {labels[cur_len : cur_len + user_prompt_len]}')
            labels[cur_len : cur_len + user_prompt_len] = IGNORE_INDEX
            # print(f'mask: {cur_len} - {cur_len + user_prompt_len}')
            cur_len += user_total_len  # 用户部分总长度（含 "USER: "）

            # 处理助手部分（包含 sep 并忽略 "ASSISTANT: "）
            assistant_part_with_sep = sep + assistant_part  # 添加 "ASSISTANT: " 到助手内容前
            assistant_tokens = self.tokenizer_image_token(assistant_part_with_sep, tokenizer)
            assistant_label_len = len(self.tokenizer_image_token(sep, tokenizer)) - 1 # 计算 "ASSISTANT: " 的 token 长度

            # 忽略助手标签部分
            # labels[cur_len : cur_len + assistant_label_len] = IGNORE_INDEX
            cur_len += len(assistant_tokens) + eos_token_length  # 包含 EOS token

        # 剩余部分全部忽略
        labels[cur_len:] = IGNORE_INDEX
        return labels, cur_len