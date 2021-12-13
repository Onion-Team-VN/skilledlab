"""
---
title: Train a MLP on MNIST
summary: >
  Train a MLP on MNIST 
---
"""
from typing import List, Optional

from torch import nn

from labml import experiment
from labml.configs import option

from skilledlab.experiments.mnist import MNISTConfigs
from skilledlab.mlp import MlpBase

class Configs(MNISTConfigs):
    layer_sizes: list = [784,128,64,32,10] 


@option(Configs.model)
def _mlp(c: Configs):
    """
    ### Create model 
    """
    base = MlpBase(c.layer_sizes)
    # Linear layer for classification 
    classification = nn.Linear(c.layer_sizes[-1],10)

    # Stack them 
    model = nn.Sequential(base,classification)
    # Move the model to the device 
    return model.to(c.device)

def main():
    # Create experiment 
    experiment.create(name='mlp',comment='mnist',writers={'screen'})
    # Create configuration 
    conf = Configs()
    # Load configuration 
    experiment.configs(conf, {
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 0.001,
        'epochs': 10
    })
    # Set model for saving/loading
    experiment.add_pytorch_models({'model': conf.model})
    # Start the experiment and run the training loop
    with experiment.start():
        conf.run()

#
if __name__ == '__main__':
    main()