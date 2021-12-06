from pathlib import PurePath
from typing import Dict, List, Optional, Callable, Union, Tuple

class Tracker:
    def __init__(self) -> None:
        self.__global_step: Optional[int] = None
    
    def __assert_name(self) -> None:
        pass

    def set_global_step(self, global_step: Optional[int]):
        self.__global_step = global_step


_internal: Optional[Tracker] = None


def tracker_singleton() -> Tracker:
    global _internal
    if _internal is None:
        _internal = Tracker()

    return _internal
