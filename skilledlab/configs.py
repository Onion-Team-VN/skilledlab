from typing import List, Callable, overload, Union, Tuple
from skilledlab.internal.configs.base import Configs as _Configs
from skilledlab.internal.configs.config_item import ConfigItem
from skilledlab.utils.errors import ConfigsError

class BaseConfigs(_Configs):
    r"""
    You should sub-class this class to create your own configurations
    """
    pass
