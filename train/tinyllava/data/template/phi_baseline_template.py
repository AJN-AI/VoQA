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

@register_template('phi_baseline')
@dataclass
class PhiBaselineTemplate(Template):

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
