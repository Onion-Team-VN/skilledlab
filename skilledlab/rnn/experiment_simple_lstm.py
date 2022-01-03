"""
---
title: Train a RNN on AG news 
https://www.kaggle.com/amananandrai/ag-news-classification-dataset
---
"""

from typing import Optional
from torch import nn
from labml import experiment
from labml.configs import option
from skilledlab.experiments.nlp_classification import NLPClassificationConfigs
from skilledlab.rnn import SimpleLstmBase

class Configs(NLPClassificationConfigs):
    embed_size: int = 100 
    hidden_nodes: int = 64
    batch_size: int = 64 

@option(Configs.model)
def _simple_rnn(c: Configs):
    """
    ### Create model 
    """
    base = SimpleLstmBase(c.n_tokens, c.embed_size, c.hidden_nodes)
    classification = nn.Linear(c.hidden_nodes,4)
    # Stack them 
    model = nn.Sequential(base,classification)
    # Move the model to the device 
    return model.to(c.device)

def main():
    # Create experiment 
    experiment.create(name='simple_rnn_classify',comment='agnews',writers={'screen'})
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
