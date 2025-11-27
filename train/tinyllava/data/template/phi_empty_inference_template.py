from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

from .formatter import EmptyFormatter, StringFormatter
from .base import Template
from .formatter import Formatter
from . import register_template

from ...utils.constants import *

from transformers import PreTrainedTokenizer
import torch
    
system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

@register_template('phi_empty_inference')
@dataclass
class PhiEmptyInferenceTemplate(Template):
    format_image_token: "Formatter" = StringFormatter(slot="<image>\n{{content}}")
    format_user: "Formatter" = StringFormatter(slot="USER" + ": " + "<image>" + "{{content}}" + " ")
    format_assistant: "Formatter" = StringFormatter(slot="" + "{{content}}" + "<|endoftext|>")
    system: "Formatter" = EmptyFormatter(slot=system+" ")
    separator: "Formatter" = EmptyFormatter(slot=['', '<|endoftext|>'])

    def _prompt(
        self,
        question_list, answer_list,
    ):
        msg = ""
        for i, (question, answer) in enumerate(zip(question_list, answer_list)):
            if i == 0:
                msg += self.system.apply()
            if DEFAULT_IMAGE_TOKEN in question:
                # question = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
                question = self.format_image_token.apply(content=question).strip()
            msg += self.format_user.apply(content=question)
            # msg += self.format_assistant.apply(content=answer)
        return msg


    def _make_masks(self, labels, tokenizer, sep, eos_token_length, rounds):
       
        cur_len = 0
        for rou in rounds:
            if rou == "":
                break
                
            parts = rou.split(sep)
            if len(parts) != 2:
                break
                
            round_len = len(self.tokenizer_image_token(rou, tokenizer)) + eos_token_length
            
            cur_len += round_len
               
        labels[cur_len:] = IGNORE_INDEX
        
        return labels, cur_len

