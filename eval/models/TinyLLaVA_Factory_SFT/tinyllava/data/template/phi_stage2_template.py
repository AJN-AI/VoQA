from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

from .formatter import EmptyFormatter, StringFormatter
from .base import Template
from .formatter import Formatter
from . import register_template
import copy

from transformers import PreTrainedTokenizer
import torch
    
system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

@register_template('phi_stage2')
@dataclass
class PhiStage2Template(Template):
    format_image_token: "Formatter" = EmptyFormatter(slot="")
    format_user: "Formatter" = EmptyFormatter(slot="")
    format_assistant: "Formatter" = StringFormatter(slot="{{content}}" + "<|endoftext|>")
    system: "Formatter" = EmptyFormatter(slot="")
    separator: "Formatter" = EmptyFormatter(slot=['', '<|endoftext|>'])

    def make_labels(self, input_ids, prompt, tokenizer):
        labels = copy.deepcopy(input_ids)
        return labels







