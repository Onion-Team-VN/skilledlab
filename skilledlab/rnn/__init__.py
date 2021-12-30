"""
title: Simple recuurent neural network 
"""

from typing import Optional, Tuple

import torch 
from torch import nn 

class LastTimeStep(nn.Module):
    """
    A class for extracting the hidden activations of the last time step following 
    the output of a PyTorch RNN module. 
    """
    def __init__(self, bidirectional=False):
        super(LastTimeStep, self).__init__()
        if bidirectional:
            self.num_driections = 2
        else:
            self.num_driections = 1    
    # def forward(self, x: Tuple):
    #     print(x[0].size())
    #     print(x[1].size())
    #     #Result is either a tupe (out, h_t)
    #     #or a tuple (out, (h_t, c_t))
    #     last_step = x[1] #this will be h_t
    #     print(last_step.size())
    #     if(type(last_step) == Tuple):#unless it's a tuple, 
    #         last_step = last_step[0]#then h_t is the first item in the tuple
    #     batch_size = last_step.shape[2] #per docs, shape is: '(num_layers * num_directions, batch, hidden_size)'
    #     #reshaping so that everything is separate 
    #     last_step = last_step.view(self.rnn_layers, self.num_driections, batch_size, -1)
    #     #We want the last layer's results
    #     last_step = last_step[self.rnn_layers-1] 
    #     print(last_step.size())
    #     #Re order so batch comes first
    #     # last_step = last_step.permute(2, 0, 1)
    #     #Finally, flatten the last two dimensions into one
    #     return last_step.reshape(batch_size, -1)
    def forward(self, x: Tuple):
        last_step = x[0]
        batch_size = last_step.shape[1]
        seq_len = last_step.shape[0]
        last_step = last_step.view(seq_len,batch_size,self.num_driections,-1)
        last_step = torch.mean(last_step,2)
        last_step = last_step[0]
        return last_step.reshape(batch_size, -1)


class SimpleRnnBase(nn.Module):
    """
    Simple RNN network 
    """
    def __init__(self, 
        vocab_size: int, 
        embed_size: int=200, 
        hidden_nodes: int=128,
        num_layers: int = 3,
        ) -> None:
        super().__init__()
        self.base = nn.Sequential(
            nn.Embedding(vocab_size, embed_size),
            nn.RNN(
                embed_size, 
                hidden_nodes,
                num_layers, 
                batch_first=True,
                bidirectional= True),
            LastTimeStep(True)
        )
    
    def forward(self, x:torch.Tensor):
        return self.base(x)