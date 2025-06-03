"""This files contains training loss implementation.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

Ref:
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py
"""
from typing import Mapping, Text, Tuple

import torch
import torch.nn as nn
    

class ARLoss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.target_vocab_size = config.model.vq_model.codebook_size
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        shift_logits = logits[..., :-1, :].permute(0, 2, 1).contiguous() # NLC->NCL
        shift_labels = labels.contiguous()
        shift_logits = shift_logits.view(shift_logits.shape[0], self.target_vocab_size, -1)
        shift_labels = shift_labels.view(shift_labels.shape[0], -1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = self.criterion(shift_logits, shift_labels)
        correct_tokens = (torch.argmax(shift_logits, dim=1) == shift_labels).sum(dim=1) / shift_labels.size(1)
        return loss, {"loss": loss, "correct_tokens": correct_tokens.mean()}
