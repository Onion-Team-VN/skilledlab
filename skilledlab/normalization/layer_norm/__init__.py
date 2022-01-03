from typing import List, Optional

import torch
from torch import nn

from labml_helpers.module import Module

class ConvBLock(Module):

    def __init__(self,in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.ln = nn.LayerNorm(out_channels)
        # First activation function (ReLU)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.act(self.ln(self.conv(x)))
    

class CnnNetBase(Module):
    
    def __init__(self,n_channels: List[int],
                img_channels: int = 1, first_kernel_size: int = 7) -> None:
        super().__init__()
        # Initial convolution layer maps from `img_channels` to number of channels in the first
        self.conv = nn.Conv2d(img_channels, n_channels[0],
                              kernel_size=first_kernel_size, stride=2, padding=first_kernel_size // 2)
        # Batch norm after initial convolution
        self.ln = nn.LayerNorm(n_channels[0])
        # Maxpool 
         # List of blocks
        blocks = []
        # Number of channels from previous layer (or block)
        prev_channels = n_channels[0]
        # Loop through each feature map size
        for i, channels in enumerate(n_channels):
            blocks.append(ConvBLock(prev_channels,channels,stride=1))
            prev_channels = channels
        # Stack the blocks
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor):
        """
        * `x` has shape `[batch_size, img_channels, height, width]`
        """
        # Initial convolution and batch normalization
        x = self.ln(self.conv(x))
        # Convolutional Blocks 
        x = self.blocks(x)
        # Change `x` from shape `[batch_size, channels, h, w]` to `[batch_size, channels, h * w]`
        x = x.view(x.shape[0], x.shape[1], -1)
        # Global average pooling
        return x.mean(dim=-1)