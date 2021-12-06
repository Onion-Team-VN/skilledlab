import json
import os
import pathlib
import time
from typing import Optional, List, Set, Dict, Union, TYPE_CHECKING, Any

from skilledlab import logger
from skilledlab.internal.configs.base import Configs
from skilledlab.internal.experiment.experiment_run import Run
from skilledlab.internal.lab import lab_singleton
from skilledlab.internal.tracker import tracker_singleton as tracker
from skilledlab.logger import Text
from skilledlab.utils import get_caller_file 
from skilledlab.utils.notice import skilledlab_notice


class ModelSaver:
    """
    An abstract class defining model saver/loader.

    The implementation should keep a reference to the model and load and save the model
    parameters.
    """
    def save(self, checkpoint_path: pathlib.Path) -> any:
        """
        Saves the model in the given checkpoint path

        Arguments:
            checkpoint_path (pathlib.Path): The path to save the model at

        Returns any meta info, such as the individual filename(s)
        """
        raise NotImplementedError()

    def load(self, checkpoint_path: pathlib.Path, info: any) -> None:
        """
        Loads the model from the given checkpoint path

        Arguments:
            checkpoint_path (pathlib.Path): The path to load the model from
            info (any): The returned meta data when saving
        """
        raise NotImplementedError()

class CheckpointSaver:
    model_savers: Dict[str, ModelSaver]

    def __init__(self, path: pathlib.PurePath):
        self.path = path
        self.model_savers = {}
        self.__no_savers_warned = False

    def add_savers(self, models: Dict[str, ModelSaver]):
        """
        ## Set variable for saving and loading
        """
        if experiment_singleton().is_started:
            raise RuntimeError('Cannot register models with the experiment after experiment has started.'
                               'Register models before calling experiment.start')

        self.model_savers.update(models)

    def save(self, global_step):
        """
        ## Save model as a set of numpy arrays
        """

        if not self.model_savers:
            if not self.__no_savers_warned:
                skilledlab_notice(["No models were registered for saving\n",
                              "You can register models with ",
                              ('experiment.add_pytorch_models', Text.value)])
                self.__no_savers_warned = True
            return

        checkpoints_path = pathlib.Path(self.path)
        if not checkpoints_path.exists():
            checkpoints_path.mkdir()

        checkpoint_path = checkpoints_path / str(global_step)
        assert not checkpoint_path.exists()

        checkpoint_path.mkdir()

        info = {}
        for name, saver in self.model_savers.items():
            info[name] = saver.save(checkpoint_path)

        # Save header
        with open(str(checkpoint_path / "info.json"), "w") as f:
            f.write(json.dumps(info))

    def load(self, checkpoint_path: pathlib.Path, models: List[str] = None):
        """
        ## Load model as a set of numpy arrays
        """

        if not self.model_savers:
            if not self.__no_savers_warned:
                skilledlab_notice(["No models were registered for loading or saving\n",
                              "You can register models with ",
                              ('experiment.add_pytorch_models', Text.value)])
                self.__no_savers_warned = True
            return

        if not models:
            models = list(self.model_savers.keys())

        with open(str(checkpoint_path / "info.json"), "r") as f:
            info = json.loads(f.readline())

        to_load = []
        not_loaded = []
        missing = []
        for name in models:
            if name not in info:
                missing.append(name)
            else:
                to_load.append(name)
        for name in info:
            if name not in models:
                not_loaded.append(name)

        # Load each model
        for name in to_load:
            saver = self.model_savers[name]
            saver.load(checkpoint_path, info[name])

        if missing:
            skilledlab_notice([(f'{missing} ', Text.highlight),
                          ('model(s) could not be found.\n'),
                          (f'{to_load} ', Text.none),
                          ('models were loaded.', Text.none)
                          ], is_danger=True)
        if not_loaded:
            skilledlab_notice([(f'{not_loaded} ', Text.none),
                          ('models were not loaded.\n', Text.none),
                          'Models to be loaded should be specified with: ',
                          ('experiment.add_pytorch_models', Text.value)])


class Experiment:
    r"""
    Each experiment has different configurations or algorithms.
    An experiment can have multiple runs.

    Keyword Arguments:
        name (str, optional): name of the experiment
        python_file (str, optional): path of the Python file that
            created the experiment
        comment (str, optional): a short description of the experiment
        writers (Set[str], optional): list of writers to write stat to
        ignore_callers: (Set[str], optional): list of files to ignore when
            automatically determining ``python_file``
        tags (Set[str], optional): Set of tags for experiment
    """
    run: Run
    checkpoint_saver: CheckpointSaver
    distributed_rank: int

    def __init__(self, *,
                 uuid: str,
                 name: Optional[str],
                 python_file: Optional[str],
                 comment: Optional[str],
                 writers: Set[str],
                 ignore_callers: Set[str],
                 tags: Optional[Set[str]],
                 is_evaluate: bool):
        self.experiment_path = lab_singleton().experiments / name
        
        self.run = Run.create(
            uuid=uuid,
            experiment_path=self.experiment_path,
            python_file=python_file,
            trial_time=time.localtime(),
            name=name,
            comment=comment,
            tags=list(tags))

        self.checkpoint_saver = CheckpointSaver(self.run.checkpoint_path)
        
        self.distributed_rank = 0

    def save_checkpoint(self):
        if self.is_evaluate:
            return
        if self.distributed_rank != 0:
            return

        self.checkpoint_saver.save(tracker().global_step)
    
    

def experiment_singleton() -> Experiment:
    global _internal

    if _internal is None:
        raise RuntimeError('Experiment not created. '
                           'Create an experiment first with `experiment.create`'
                           ' or `experiment.record`')

    return _internal


def create_experiment(*,
                      uuid: str,
                      name: Optional[str],
                      python_file: Optional[str],
                      comment: Optional[str],
                      writers: Set[str],
                      ignore_callers: Set[str],
                      tags: Optional[Set[str]],
                      is_evaluate: bool):
    global _internal

    _internal = Experiment(uuid=uuid,
                           name=name,
                           python_file=python_file,
                           comment=comment,
                           writers=writers,
                           ignore_callers=ignore_callers,
                           tags=tags,
                           is_evaluate=is_evaluate)





