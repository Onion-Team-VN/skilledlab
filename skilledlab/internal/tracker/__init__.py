from pathlib import PurePath
from typing import Dict, List, Optional, Callable, Union, Tuple

from skilledlab.internal import util
from skilledlab.internal.lab import lab_singleton, LabYamlNotfoundError
from skilledlab.internal.util import strings
from .indicator import Indicator
from .indicator.factory import load_indicator_from_dict, create_default_indicator
from .indicator.numeric import Scalar
from .namespace import Namespace


class Tracker:
    indicators: Dict[str, Indicator]
    dot_indicators: Dict[str, Indicator]
    namespaces: List[Namespace]

    def __init__(self) -> None:
        self.__start_global_step: Optional[int] = None
        self.__global_step: Optional[int] = None

        self.indicators = {}
        self.dot_indicators = {}
        self.is_indicators_updated = True
        self.__indicators_file = None
        self.reset_store()
    
    def __assert_name(self, name: str, value: any):
        if name.endswith("."):
            if name in self.dot_indicators:
                assert self.dot_indicators[name].equals(value)

        assert name not in self.indicators, f"{name} already used"
    
    def reset_store(self):
        self.indicators = {}
        self.dot_indicators = {}
        self.__indicators_file = None
        self.namespaces = []
        self.is_indicators_updated = True
        try:
            for ind in lab_singleton().indicators:
                self.add_indicator(load_indicator_from_dict(ind))
        except LabYamlNotfoundError:
            pass
    
    def add_indicator(self, indicator: Indicator):
        self.dot_indicators[indicator.name] = indicator
        self.is_indicators_updated = True

    def _create_indicator(self, key: str, value: any):
        if key in self.indicators:
            return

        ind_key, ind_score = strings.find_best_pattern(key, self.dot_indicators.keys())
        print(self.dot_indicators)
        if ind_key is None:
            raise ValueError(f"Cannot find matching indicator for {key}")
        if ind_score == 0:
            is_print = self.dot_indicators[ind_key].is_print
            self.indicators[key] = create_default_indicator(key, value, is_print)
        else:
            self.indicators[key] = self.dot_indicators[ind_key].copy(key)
        self.is_indicators_updated = True
    
    def store(self, key: str, value: any):
        if value is None:
            return

        if key.endswith('.'):
            key = '.'.join([key[:-1]] + [ns.name for ns in self.namespaces])

        self._create_indicator(key, value)
        self.indicators[key].collect_value(value)
    
    def namespace_enter(self, ns: Namespace):
        self.namespaces.append(ns)

    def namespace_exit(self, ns: Namespace):
        if len(self.namespaces) == 0:
            raise RuntimeError("Impossible")

        if ns is not self.namespaces[-1]:
            raise RuntimeError("Impossible")

        self.namespaces.pop(-1)


    def save_indicators(self, file: Optional[PurePath] = None):
        if not self.is_indicators_updated:
            return
        self.is_indicators_updated = False
        if file is None:
            if self.__indicators_file is None:
                return
            file = self.__indicators_file
        else:
            self.__indicators_file = file

        wildcards = {k: ind.to_dict() for k, ind in self.dot_indicators.items()}
        inds = {k: ind.to_dict() for k, ind in self.indicators.items()}
        with open(str(file), "w") as file:
            file.write(util.yaml_dump({'wildcards': wildcards,
                                       'indicators': inds}))

        for w in self.__writers:
            w.save_indicators(self.dot_indicators, self.indicators)


    def set_global_step(self, global_step: Optional[int]):
        self.__global_step = global_step

    def add_global_step(self, increment_global_step: int = 1):
        if self.__global_step is None:
            self.__global_step = self.global_step

        self.__global_step += increment_global_step
    
    def set_start_global_step(self, global_step: Optional[int]):
        self.__start_global_step = global_step


_internal: Optional[Tracker] = None


def tracker_singleton() -> Tracker:
    global _internal
    if _internal is None:
        _internal = Tracker()

    return _internal
