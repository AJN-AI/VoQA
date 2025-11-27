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

# QA-only-SFT
@register_template('phi_qa_only')
@dataclass
class PhiUniQAOnlyTemplate(Template):

    # This template calculte loss at both questions and answers, except the instructions added.
    
    format_image_token: "Formatter" = StringFormatter(slot="<image>\n{{content}}")
    format_user: "Formatter" = StringFormatter(slot="USER" + ": " + "<image>\n" +  "{{content}}" + " ")
    format_assistant: "Formatter" = StringFormatter(slot="{{content}}" + "<|endoftext|>")
    system: "Formatter" = EmptyFormatter(slot=system+" ")
    separator: "Formatter" = EmptyFormatter(slot=['', '<|endoftext|>'])
    # separator: "Formatter" = EmptyFormatter(slot=['<|endoftext|>'])


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
        # print('rounds:', rounds)

        for rou in rounds:
            if not rou.strip():
                break

            user_sep = "USER: " + "<image>\n"
            parts = rou.split(user_sep)
            if len(parts) != 2:
                break

            user_part, assistant_part = parts

            user_prompt_tokens = self.tokenizer_image_token(user_part, tokenizer)
            sep_tokens = self.tokenizer_image_token(user_sep, tokenizer)
            user_prompt_len = len(user_prompt_tokens) + len(sep_tokens)
            # print(f'user_prompt: "{user_part + user_sep}"')

            # print(f'original label user: {labels[cur_len : cur_len + user_prompt_len]}')
            labels[cur_len : cur_len + user_prompt_len] = IGNORE_INDEX
            # print(f'mask: {cur_len} - {cur_len + user_prompt_len}')
            cur_len += user_prompt_len

            question_answer_tokens = self.tokenizer_image_token(assistant_part, tokenizer)
            cur_len += len(question_answer_tokens) + eos_token_length

        cur_len -= 1
        labels[cur_len:] = IGNORE_INDEX
        return labels, cur_len