import json
import os
import pathlib
import time
from typing import Optional, List, Set, Dict, Union, TYPE_CHECKING, Any

from skilledlab import logger
from skilledlab.internal.configs.base import Configs
from skilledlab.internal.configs.processor import ConfigProcessor
from skilledlab.internal.experiment.experiment_run import Run
from skilledlab.internal.experiment import experiment_run
from skilledlab.internal.experiment.watcher import ExperimentWatcher
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
        if python_file is None:
            python_file = get_caller_file(ignore_callers)

        lab_singleton().set_path(python_file)

        if name is None:
            file_path = pathlib.PurePath(python_file)
            name = file_path.stem
        
        if comment is None:
            comment = ''
        if global_params_singleton().comment is not None:
            comment = global_params_singleton().comment
        self.experiment_path = lab_singleton().experiments / name
        if tags is None:
            tags = set(name.split('_'))
        self.run = Run.create(
            uuid=uuid,
            experiment_path=self.experiment_path,
            python_file=python_file,
            trial_time=time.localtime(),
            name=name,
            comment=comment,
            tags=list(tags))

        self.run.commit_message = ''
        self.writers = writers
        self.checkpoint_saver = CheckpointSaver(self.run.checkpoint_path)
        self.is_evaluate = is_evaluate
        self.distributed_rank = 0
    
    def __print_info(self):
        """
        🖨 Print the experiment info and check git repo status
        """

        logger.log()
        logger.log([
            (self.run.name, Text.title),
            ': ',
            (str(self.run.uuid), Text.meta)
        ])

        if self.run.comment != '':
            logger.log(['\t', (self.run.comment, Text.highlight)])

        commit_message = self.run.commit_message.strip().replace('\n', '¶ ').replace('\r', '')
        logger.log([
            "\t"
            "[dirty]" if self.run.is_dirty else "[clean]",
            ": ",
            (f"\"{commit_message}\"", Text.highlight)
        ])

        if self.run.load_run is not None:
            logger.log([
                "\t"
                "loaded from",
                ": ",
                (f"{self.run.load_run}", Text.meta2),
            ])

    def _load_checkpoint(self, checkpoint_path: pathlib.Path):
        self.checkpoint_saver.load(checkpoint_path)

    def save_checkpoint(self):
        if self.is_evaluate:
            return
        if self.distributed_rank != 0:
            return

        self.checkpoint_saver.save(tracker().global_step)
    

    def calc_configs(self,
                     configs: Union[Configs, Dict[str, any]],
                     configs_override: Optional[Dict[str, any]]):
        if configs_override is None:
            configs_override = {}
        if global_params_singleton().configs is not None:
            configs_override.update(global_params_singleton().configs)

        self.configs_processor = ConfigProcessor(configs, configs_override)

        if self.distributed_rank == 0:
            logger.log()
    
    def __start_from_checkpoint(self, run_uuid: str, checkpoint: Optional[int]):
        checkpoint_path, global_step = experiment_run.get_run_checkpoint(
            run_uuid,
            checkpoint)

        if global_step is None:
            return 0
        else:
            self._load_checkpoint(checkpoint_path)
            self.run.load_run = run_uuid

        return global_step

    def _save_pid(self):
        if not self.run.pids_path.exists():
            self.run.pids_path.mkdir()

        pid_path = self.run.pids_path / f'{self.distributed_rank}.pid'
        assert not pid_path.exists(), str(pid_path)

        with open(str(pid_path), 'w') as f:
            f.write(f'{os.getpid()}')

    def _start_tracker(self):
        tracker().reset_writers()

        if self.is_evaluate:
            return

        if self.distributed_rank != 0:
            return

        if 'screen' in self.writers:
            from skilledlab.internal.tracker.writers import screen
            tracker().add_writer(screen.ScreenWriter())

        if 'file' in self.writers:
            raise NotImplementedError()

            self.web_api = None
    
    def start(self, *,
              run_uuid: Optional[str] = None,
              checkpoint: Optional[int] = None):
        if run_uuid is not None:
            if checkpoint is None:
                checkpoint = -1
            global_step = self.__start_from_checkpoint(run_uuid, checkpoint)
        else:
            global_step = 0

        self.run.start_step = global_step

        self._start_tracker()
        tracker().set_start_global_step(global_step)

        if self.distributed_rank == 0:
            self.__print_info()

        if not self.is_evaluate:
            if self.distributed_rank == 0:
                from skilledlab.internal.computer.configs import computer_singleton
                computer_singleton().add_project(lab_singleton().path)
                self.run.save_info()
            self._save_pid()

            if self.distributed_rank == 0:
                tracker().save_indicators(self.run.indicators_path)

        self.is_started = True
        return ExperimentWatcher(self)
    
    def finish(self, status: str, details: any = None):
        if not self.is_evaluate:
            with open(str(self.run.run_log_path), 'a') as f:
                end_time = time.time()
                data = json.dumps({'status': status,
                                   'rank': self.distributed_rank,
                                   'details': details,
                                   'time': end_time}, indent=None)
                f.write(data + '\n')

        tracker().finish_loop()


class GlobalParams:
    def __init__(self):
        self.configs = None
        self.comment = None

_global_params: Optional[GlobalParams] = None
_internal: Optional[Experiment] = None


def global_params_singleton() -> GlobalParams:
    global _global_params

    if _global_params is None:
        _global_params = GlobalParams()

    return _global_params


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





