import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from labml import experiment
from labml.configs import option
from labml_helpers.module import Module
from skilledlab.experiments.mnist import MNISTConfigs

class Model(Module):
    """
    ### Model definition
    """

    def __init__(self):
        super().__init__()
        # Note that we omit the bias parameter
        self.conv1 = nn.Conv2d(1, 20, 5, 1, bias=False)
        # Batch normalization with 20 channels (output of convolution layer).
        # The input to this layer will have shape `[batch_size, 20, height(24), width(24)]`
        #
        self.conv2 = nn.Conv2d(20, 50, 5, 1, bias=False)
        # Batch normalization with 50 channels.
        # The input to this layer will have shape `[batch_size, 50, height(8), width(8)]`
        #
        self.fc1 = nn.Linear(4 * 4 * 50, 500, bias=False)
        # Batch normalization with 500 channels (output of fully connected layer).
        # The input to this layer will have shape `[batch_size, 500]`
        #
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

@option(MNISTConfigs.model)
def model(c: MNISTConfigs):
    """
    ### Create model

    We use [`MNISTConfigs`](../../experiments/mnist.html#MNISTConfigs) configurations
    and set a new function to calculate the model.
    """
    return Model().to(c.device)


def main():
    # Create experiment
    experiment.create(name='mnist_batch_norm')
    # Create configurations
    conf = MNISTConfigs()
    # Load configurations
    experiment.configs(conf, {
        'optimizer.optimizer': 'SGD',
        'optimizer.learning_rate': 0.001,
    })
    # Start the experiment and run the training loop
    with experiment.start():
        conf.run()


#
if __name__ == '__main__':
    main()