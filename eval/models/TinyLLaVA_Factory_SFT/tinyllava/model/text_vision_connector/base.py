import os

import torch
import torch.nn as nn


class Text_Vision_Connector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self._text_vision_connector = None

    def load_model(self, **kwargs):
        pretrained_text_vision_connector_path = kwargs.get('pretrained_text_vision_connector_path', None)
        if pretrained_text_vision_connector_path is not None:
            pretrained_text_vision_connector_path = os.path.join(pretrained_text_vision_connector_path, 'pytorch_model.bin')
            text_vision_connector_weights = torch.load(pretrained_text_vision_connector_path, map_location='cpu')
            
            # 检查权重键名是否包含_text_vision_connector
            has_prefix = any('_text_vision_connector' in key for key in text_vision_connector_weights.keys())
            
            if has_prefix:
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
                self._text_vision_connector.load_state_dict(get_w(text_vision_connector_weights, '_text_vision_connector'))
            else:
                self._text_vision_connector.load_state_dict(text_vision_connector_weights)
                
            print(f'Loading text_vision_connector from {pretrained_text_vision_connector_path}...')

        for p in self._text_vision_connector.parameters():
            p.requires_grad = False
   
    def forward(self, x):
        return self._text_vision_connector(x)
        

  
