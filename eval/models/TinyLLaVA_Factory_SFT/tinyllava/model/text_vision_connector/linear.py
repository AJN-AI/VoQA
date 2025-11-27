import torch.nn as nn

from . import register_text_vision_connector
from .base import Text_Vision_Connector



@register_text_vision_connector('linear')    
class LinearTextVisionConnector(Text_Vision_Connector):
    def __init__(self, config):
        super().__init__()
        self._text_vision_connector =  nn.Linear(config.text_vision_hidden_size, config.hidden_size)

        
    # @property
    # def config(self):
    #     return {"connector_type": 'linear',
    #             "in_hidden_size": self.in_hidden_size, 
    #             "out_hidden_size": self.out_hidden_size
    #            }
