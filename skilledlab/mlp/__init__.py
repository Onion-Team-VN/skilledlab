"""
---
title: Multipayer Perceptron for Image Recognition 
---
"""

from typing import List, Optional

import torch
from torch import nn

from labml_helpers.module import Module

class MlpBase(Module):
    """
    Base layers for multilayer perceptron (MLP)
    """

    def __init__(self, layer_sizes: list) -> None:
        """
        * `layer_sizes` is the list contains the size of each layers in the MLP including the last layer 
        for the mnist classification problem the last layer should be equal to 10
        """
        super().__init__()
        layers = []
        for j in range(len(layer_sizes)-1):
            # act = nn.ReLU if j < len(layer_sizes)-2 else nn.Identity
            layers += [nn.Linear(layer_sizes[j], layer_sizes[j+1]), nn.ReLU()]
        self.base = nn.Sequential(*layers)

    def forward(self,x: torch.Tensor):
        """
        * `x` has shape [batch_size, height, width]
        so it should be flatten to [batch_size, height*width] to be fed into MLP 
        """
        x = x.view(x.shape[0], -1)
        # return self.act(self.fc(x))
        return self.base(x)