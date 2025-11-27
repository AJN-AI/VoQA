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

@register_template('phi_r_qra')
@dataclass
class PhiRQRATemplate(Template):

    # This template calculte loss at both questions and answers, except the instructions added.
    
    format_image_token: "Formatter" = StringFormatter(slot="<image>\n{{content}}")
    format_user: "Formatter" = StringFormatter(slot="USER" + ": " + "<image>\n ASSISTANT: " + "{{content}}" + " ")
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


    @classmethod    
    def tokenizer_image_token(cls, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        def _insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
        # prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>\n ASSISTANT: ')]
        trigger_input_ids = tokenizer("\n ASSISTANT: ").input_ids

        input_ids = []
        offset = 0
        
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in _insert_separator(prompt_chunks, ([image_token_index] + trigger_input_ids) * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    def _make_masks(self, labels, tokenizer, sep, eos_token_length, rounds):
        cur_len = 0

        for rou in rounds:
            if not rou.strip():
                break

            parts = rou.split(sep)
            if len(parts) != 3:
                break

            user_part, question_part, answer_part = parts
            # print("split:", user_part, question_part, answer_part)

            user_prompt_tokens = self.tokenizer_image_token(user_part, tokenizer)
            sep_tokens = self.tokenizer_image_token(sep, tokenizer)
            total_tokens = self.tokenizer_image_token(user_part + sep, tokenizer)
            user_prompt_len = len(total_tokens)
            # if len(user_prompt_tokens) + len(sep_tokens) != len(total_tokens):
            #     print("lens-1:", len(user_prompt_tokens) + len(sep_tokens), len(total_tokens))

            # print(f'user_prompt: "{user_part + sep}"')

            # print(f'original label user: {labels[cur_len : cur_len + user_prompt_len]}')
            labels[cur_len : cur_len + user_prompt_len] = IGNORE_INDEX
            # print(f'mask: {cur_len} - {cur_len + user_prompt_len}')
            cur_len += user_prompt_len

            loss_part = question_part + sep + answer_part
            # question_tokens = self.tokenizer_image_token(question_part, tokenizer)
            # answer_tokens = self.tokenizer_image_token(answer_part, tokenizer)
            # sep_and_answer_tokens = self.tokenizer_image_token(sep + answer_part, tokenizer)
            loss_len = len(self.tokenizer_image_token(loss_part, tokenizer))
            # if loss_len != len(question_tokens) + len(sep_and_answer_tokens):
            #     print("lens-2:", loss_len, len(question_tokens) + len(sep_tokens) + len(answer_tokens))

            cur_len += loss_len + eos_token_length

        labels[cur_len:] = IGNORE_INDEX

        return labels, cur_len