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
from torchvision import transforms

from skilledlab.experiments.mnist import MNISTConfigs
from skilledlab.cnn import CnnNetBase

class Configs(MNISTConfigs):
    # Number of channels for each feature map size
    n_channels: List[int] = [16, 16, 16, 16, 16]
    # Kernel size of the initial convolution layer
    first_kernel_size: int = 1


@option(Configs.model)
def _cnnnet(c: Configs):
    """
    ### Create model
    """
    base = CnnNetBase(c.n_channels,img_channels=1)
    # Linear layer for classification
    classification = nn.Linear(c.n_channels[-1], 10)
    # Stack them
    model = nn.Sequential(base, classification)
    # Move the model to the device
    return model.to(c.device)

@option(Configs.dataset_transforms)
def mnist_transforms():
    return transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.98, 1.02)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def main():
    # Create experiment 
    experiment.create(name='cnn',comment='mnist_cnn_data_augmentation',writers={'screen'})
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
