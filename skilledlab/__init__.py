from typing import Dict, Tuple, Any
import torch 
class WeightDecay:
    """
    ## L2 Weight decay
    """

    def __init__(self, weight_decay: float = 0., weight_decouple: bool = True, absolute: bool = False):
        """
        ### Initialize weight decay

        * `weight_decay` is the decay coefficient
        * `weight_decouple` is a flag indicating whether to add the weight decay to the gradient or directly
        decay from the parameter. If added to the  gradient it will go through the normal optimizer update.
        * `absolute` this flag indicates whether the weight decay coefficient is absolute. This is applicable
        when the decay is performed directly on the parameter. If this is false the actual decay is
        `weight_decay`
        * `learning_rate`.
        """
        # Check hyper-parameters
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.absolute = absolute
        self.weight_decouple = weight_decouple
        self.weight_decay = weight_decay

    def defaults(self):
        """
        Return defaults for parameter groups
        """
        return dict(weight_decay=self.weight_decay)

    def __call__(self, param: torch.nn.Parameter, grad: torch.Tensor, group: Dict[str, any]):
        """
        ### Perform weight decay and return the gradient
        """

        # If we are doing the decay on the parameter directly
        if self.weight_decouple:
            # If the weight decay coefficient is absolute
            if self.absolute:
                param.data.mul_(1.0 - group['weight_decay'])
            # Otherwise,
            else:
                param.data.mul_(1.0 - group['lr'] * group['weight_decay'])
            # Return the unmodified gradient
            return grad
        else:
            if group['weight_decay'] != 0:
                # Add the weight decay to the gradient and return the modified gradient
                return grad.add(param.data, alpha=group['weight_decay'])
            else:
                return grad