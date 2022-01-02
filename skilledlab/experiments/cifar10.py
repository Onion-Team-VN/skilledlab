"""
---
title: CIFAR10 Experiment
summary: >
  This is a reusable trainer for CIFAR10 dataset
---

# CIFAR10 Experiment
"""
from typing import List

import torch.nn as nn

from labml import lab
from labml.configs import option
from labml_helpers.datasets.cifar10 import CIFAR10Configs as CIFAR10DatasetConfigs
from labml_helpers.module import Module
from skilledlab.experiments.mnist import MNISTConfigs

class CIFAR10Configs(CIFAR10DatasetConfigs, MNISTConfigs):
    """
    ## Configurations

    This extends from CIFAR 10 dataset configurations from
     [`labml_helpers`](https://github.com/labmlai/labml/tree/master/helpers)
     and [`MNISTConfigs`](mnist.html).
    """
    # Use CIFAR10 dataset by default
    dataset_name: str = 'CIFAR10'

@option(CIFAR10Configs.train_dataset)
def cifar10_train_augmented():
    """
    ### Augmented CIFAR 10 train dataset
    """
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import transforms
    return CIFAR10(str(lab.get_data_path()),
                   train=True,
                   download=True,
                   transform=transforms.Compose([
                       # Pad and crop
                       transforms.RandomCrop(32, padding=4),
                       # Random horizontal flip
                       transforms.RandomHorizontalFlip(),
                       #
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ]))


@option(CIFAR10Configs.valid_dataset)
def cifar10_valid_no_augment():
    """
    ### Non-augmented CIFAR 10 validation dataset
    """
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import transforms
    return CIFAR10(str(lab.get_data_path()),
                   train=False,
                   download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ]))
