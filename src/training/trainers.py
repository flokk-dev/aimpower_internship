"""
Creator: Flokk
Date: 09/04/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

import time
from tqdm import tqdm

# IMPORT: project
import utils

from .trainer import Trainer
from .learner import ClassicLearner, ReinforcementLearner


class ClassicTrainer(Trainer):
    """
    Represents a ClassicTrainer.

    Attributes
    ----------
        _config : Dict[str, Any]
            configuration needed to adjust the program behaviour
        _path : str
            path to the training's logs
        _learner : Learner
            ...
        _dashboard : Dashboard
            ...

    Methods
    ----------
        _launch
            Launches the training
        _run_epoch
            Runs an epoch
        _checkpoint
            Runs a checkpoint procedure
    """
    def __init__(
            self,
            config: Dict[str, Any]
    ):
        """
        Instantiates a ClassicTrainer.

        Parameters
        ----------
            config : Dict[str, Any]
                configuration needed to adjust the program behaviour
        """
        # Mother class
        super(ClassicTrainer, self).__init__(config)

    def _launch(
            self
    ):
        """ Launches the training. """
        time.sleep(1)
        self._learner.components.model.train()

        p_bar: tqdm = tqdm(total=self._config["num_epochs"], desc="training in progress")
        for epoch in range(self._config["num_epochs"]):
            self._learner.components.model.train()

            # Learns
            self._run_epoch(p_bar)

            # Updates
            self._dashboard.upload_values(self._learner.components.lr_scheduler.get_last_lr()[0])
            if (epoch + 1) % 10 == 0:
                self._checkpoint(epoch + 1)

            p_bar.update(1)

        # End
        time.sleep(10)
        self._dashboard.shutdown()

    def _run_epoch(
            self,
            p_bar: tqdm
    ):
        """
        Runs an epoch.

        Parameters
        ----------
            p_bar : tqdm
                the training's progress bar
        """
        num_batch: int = len(self._learner.components.data_loader)

        epoch_loss: list = list()
        for batch_idx, batch in enumerate(self._learner.components.data_loader):
            p_bar.set_postfix(batch=f"{batch_idx}/{num_batch}", gpu=utils.gpu_utilization())

            # Learns on batch
            epoch_loss.append(self._learner(batch))

        # Stores the results
        self._dashboard.update_loss(epoch_loss)

    def __call__(self, dataset_path: str):
        """
        Parameters
        ----------
            dataset_path : str
                path to the dataset
        """
        # Learner
        self._learner = ClassicLearner(self._config, dataset_path)

        # Mother class
        super().__call__()


class ReinforcementTrainer(Trainer):
    """
    Represents a ReinforcementTrainer.

    Attributes
    ----------
        _config : Dict[str, Any]
            configuration needed to adjust the program behaviour
        _path : str
            path to the training's logs
        _learner : Learner
            ...
        _dashboard : Dashboard
            ...

    Methods
    ----------
        _launch
            Launches the training
        _run_epoch
            Runs an epoch
        _checkpoint
            Runs a checkpoint procedure
    """
    def __init__(
            self,
            config: Dict[str, Any]
    ):
        """
        Instantiates a ReinforcementTrainer.

        Parameters
        ----------
            config : Dict[str, Any]
                configuration needed to adjust the program behaviour
        """
        # Mother class
        super(ReinforcementTrainer, self).__init__(config)

    def _launch(
            self
    ):
        """ Launches the training. """
        # Not implemented
        raise NotImplementedError()

    def _run_epoch(
            self,
            p_bar: tqdm
    ):
        """
        Runs an epoch.

        Parameters
        ----------
            p_bar : tqdm
                the training's progress bar
        """
        # Not implemented
        raise NotImplementedError()

    def __call__(self, dataset_path: str):
        """
        Parameters
        ----------
            dataset_path : str
                path to the dataset
        """
        # Learner
        self._learner = ReinforcementLearner(self._config, dataset_path)

        # Mother class
        super().__call__()
