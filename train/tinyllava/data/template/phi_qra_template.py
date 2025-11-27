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

@register_template('phi_qra')
@dataclass
class PhiQRATemplate(Template):

    # This template calculte loss at both questions and answers, except the instructions added.
    
    format_image_token: "Formatter" = StringFormatter(slot="<image>\n{{content}}")
    format_user: "Formatter" = StringFormatter(slot="USER" + ": " + "<image>\n" + "{{content}}" + " ")
    format_assistant: "Formatter" = StringFormatter(slot="ASSISTANT" + ": " + "{{content}}" + "<|endoftext|>")
    system: "Formatter" = EmptyFormatter(slot=system+" ")
    separator: "Formatter" = EmptyFormatter(slot=[' ASSISTANT: ', '<|endoftext|>'])


    def _prompt(
            self,
            question_list, answer_list,
        ):
            msg = ""
            for i, (question, answer) in enumerate(zip(question_list, answer_list)):
                if i == 0:
                    msg += self.system.apply()
                if DEFAULT_IMAGE_TOKEN in question:
                    question = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    # question = self.format_image_token.apply(content=question).strip()
                msg += self.format_user.apply(content=question)
                msg += self.format_assistant.apply(content=answer)
            return msg

    def _make_masks(self, labels, tokenizer, sep, eos_token_length, rounds):
        cur_len = 0
        first_round = True
        # print("rounds:", rounds)

        for rou in rounds:
            if not rou.strip():
                break

            if first_round:
                if "USER: <image>\n" not in rou:
                    system_len = len(self.tokenizer_image_token(rou, tokenizer))
                    labels[cur_len : cur_len + system_len] = IGNORE_INDEX
                    cur_len += system_len
                    continue

                system_part = rou.split("USER: <image>\n")[0]

                # print(f'system part: {system_part}')

                system_len = len(self.tokenizer_image_token(system_part, tokenizer)) - 1
                # print(f'original label: {labels[cur_len : cur_len + system_len]}')
                labels[cur_len : cur_len + system_len] = IGNORE_INDEX
                # print(f'mask: {cur_len} - {cur_len + system_len}')
                cur_len += system_len

                rou = rou[len(system_part):].lstrip()
                # print(f'rou: {rou}')
                first_round = False

            parts = rou.split(sep)
            if len(parts) != 2:
                break

            user_part, assistant_part = parts

            user_prompt = "USER: " + "<image>\n"
            user_prompt_tokens = self.tokenizer_image_token(user_prompt, tokenizer)
            user_prompt_len = len(user_prompt_tokens)

            # print(f'user_prompt_len: {user_prompt_len}')

            if not user_part.startswith(user_prompt):
                break

            user_content = user_part[len(user_prompt):]

            # print(f'user content: {user_content}')

            user_content_tokens = self.tokenizer_image_token(user_content, tokenizer)
            user_total_len = user_prompt_len + len(user_content_tokens)

            # print(f'original label user: {labels[cur_len : cur_len + user_prompt_len]}')
            labels[cur_len : cur_len + user_prompt_len] = IGNORE_INDEX
            # print(f'mask: {cur_len} - {cur_len + user_prompt_len}')
            cur_len += user_total_len

            assistant_part_with_sep = sep + assistant_part
            assistant_tokens = self.tokenizer_image_token(assistant_part_with_sep, tokenizer)
            assistant_label_len = len(self.tokenizer_image_token(sep, tokenizer)) - 1

            # labels[cur_len : cur_len + assistant_label_len] = IGNORE_INDEX
            cur_len += len(assistant_tokens) + eos_token_length

        labels[cur_len:] = IGNORE_INDEX
        # print("labels:", labels)
        return labels, cur_len