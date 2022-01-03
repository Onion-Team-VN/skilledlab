from typing import List, Optional

import torch
from torch import nn
from labml_helpers.module import Module

class ConvBLock(Module):

    def __init__(self,in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        # First activation function (ReLU)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.act(self.conv(x))
        
    
class SkipConvBlock(Module):
    def __init__(self, n_layers: int, in_channels: int, out_channels: int,kernel_size: int) -> None:
        super().__init__()
        #The last convolution will have a different number of inputs and output channels, so we still need that index
        l = n_layers - 1
        #this is just simple helper values 
        f = (kernel_size, kernel_size)
        pad = (kernel_size-1)//2
        #Defining the layers used, altering the construction of the last layer using the same `if i == l` list comprehension. We are going to combine convolutions via their channels, so the in and out channels change for the last layer.  
        self.layers = nn.ModuleList([nn.Conv2d(in_channels*l, out_channels, kernel_size=f, padding=pad) if i == l else nn.Conv2d(in_channels, in_channels, kernel_size=f, padding=pad) for i in range(n_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(out_channels) if i == l else nn.BatchNorm2d(in_channels) for i in range(n_layers)])
        self.act = nn.ReLU()
    
    def forward(self, x):
        activations = []
        
        for layer, bn in zip(self.layers[:-1], self.bns[:-1]):
            x = self.act(bn(layer(x)))
            activations.append( x )
        #Which is the concatination of all the activations here. Our tensors are organized as (B, C, W, H), which is the default in PyTorch. But you can change that, and sometimes people use (B, W, H, C). In that situation the C channel is at index 3 instead of 1. So you would change `cat=3` in that scenario. This is also how you would adapt this code to work with RNNs
        x = torch.cat(activations, dim=1)
        
        return self.act(self.bns[-1](self.layers[-1](x)))


class infoShareBlock(Module):
    def __init__(self,n_filters: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(n_filters, n_filters, (1,1), padding=0)
        self.bn = nn.BatchNorm2d(n_filters)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.act(self.bn(self.conv(x)))


class CnnNetBase(Module):
    
    def __init__(self,n_channels: List[int],
                img_channels: int = 1, first_kernel_size: int = 7) -> None:
        super().__init__()
        # Initial convolution layer maps from `img_channels` to number of channels in the first
        self.conv = nn.Conv2d(img_channels, n_channels[0],
                              kernel_size=first_kernel_size, stride=2, padding=first_kernel_size // 2)
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
        x = self.conv(x)
        # Convolutional Blocks 
        x = self.blocks(x)
        # Change `x` from shape `[batch_size, channels, h, w]` to `[batch_size, channels, h * w]`
        x = x.view(x.shape[0], x.shape[1], -1)
        # Global average pooling
        return x.mean(dim=-1)


class CnnSkipBase(Module):
    def __init__(self,n_channels: List[int],
                img_channels: int = 1, first_kernel_size: int = 7) -> None:
        super().__init__()
        # Initial convolution layer maps from `img_channels` to number of channels in the first
        self.conv = nn.Conv2d(img_channels, n_channels[0],
                              kernel_size=first_kernel_size, stride=2, padding=first_kernel_size // 2)
        
        # List of blocks
        blocks = []
        # Number of channels from previous layer (or block)
        prev_channels = n_channels[0]
        # Loop through each feature map size
        # Number of channels from previous layer (or block)
        prev_channels = n_channels[0]
        # Loop through each feature map size
        for i, channels in enumerate(n_channels):
            blocks.append(SkipConvBlock(n_layers=3,in_channels=prev_channels,out_channels=channels,kernel_size=3))
            prev_channels = channels
        # Stack the blocks
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor):
        """
        * `x` has shape `[batch_size, img_channels, height, width]`
        """
        # Initial convolution and batch normalization
        x = self.conv(x)
        # Convolutional Blocks 
        x = self.blocks(x)
        # Change `x` from shape `[batch_size, channels, h, w]` to `[batch_size, channels, h * w]`
        x = x.view(x.shape[0], x.shape[1], -1)
        # Global average pooling
        return x.mean(dim=-1)



class CnnSkipInfoShareBase(Module):
    def __init__(self,n_channels: List[int],
                img_channels: int = 1, first_kernel_size: int = 7) -> None:
        super().__init__()
        # Initial convolution layer maps from `img_channels` to number of channels in the first
        self.conv = nn.Conv2d(img_channels, n_channels[0],
                              kernel_size=first_kernel_size, stride=2, padding=first_kernel_size // 2)
        
        # List of blocks
        blocks = []
        # Number of channels from previous layer (or block)
        prev_channels = n_channels[0]
        # Loop through each feature map size
        # Number of channels from previous layer (or block)
        prev_channels = n_channels[0]
        # Loop through each feature map size
        for i, channels in enumerate(n_channels):
            blocks.append(SkipConvBlock(n_layers=3,in_channels=prev_channels,out_channels=channels,kernel_size=3))
            blocks.append(infoShareBlock(channels))
            prev_channels = channels
        # Stack the blocks
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor):
        """
        * `x` has shape `[batch_size, img_channels, height, width]`
        """
        # Initial convolution and batch normalization
        x = self.conv(x)
        # Convolutional Blocks 
        x = self.blocks(x)
        # Change `x` from shape `[batch_size, channels, h, w]` to `[batch_size, channels, h * w]`
        x = x.view(x.shape[0], x.shape[1], -1)
        # Global average pooling
        return x.mean(dim=-1)


