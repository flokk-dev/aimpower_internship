"""
Creator: Flokk
Date: 09/04/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

import os
import json
from tqdm import tqdm

# IMPORT: deep learning
import torch

# IMPORT: project
import paths
import utils

from .learner import Learner
from src.training.learner.components.pipeline import Pipeline
from .dashboard import Dashboard


class Trainer:
    """
    Represents a Trainer.

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
        Instantiates a Trainer.

        Parameters
        ----------
            config : Dict[str, Any]
                configuration needed to adjust the program behaviour
        """
        # Attributes
        self._config: Dict[str, Any] = config

        # Creates training's repository
        self._path = os.path.join(paths.MODELS_PATH, utils.get_datetime())
        if not os.path.exists(self._path):
            os.makedirs(self._path)
            os.mkdir(os.path.join(self._path, "images"))

        with open(os.path.join(self._path, "config.json"), 'w') as file_content:
            json.dump(self._config, file_content)

        # Components
        self._learner: Learner = None
        self._pipeline: Pipeline = None
        self._dashboard: Dashboard = None

    def _launch(
            self
    ):
        """
        Launches the training.

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
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

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()

    def _checkpoint(
            self,
            epoch: int
    ):
        """
        Runs a checkpoint procedure.

        Parameters
        ----------
            epoch : int
                the current epoch idx
        """
        # Saves pipeline and generates images
        tensors: Dict[str, torch.Tensor] = self._learner.components.pipeline(
            prompt=self._config["validation_prompts"],
            inference=True,
            return_dict=True
        )

        # Uploads and saves qualitative results
        for key, tensor in tensors.items():
            # Creates destination directory
            key_path = os.path.join(self._path, "images", key)
            if not os.path.exists(key_path):
                os.mkdir(key_path)

            # Uploads checkpoint images to WandB
            self._dashboard.upload_images(key, tensor)

            # Saves checkpoint image on disk
            utils.save_plt(tensor, os.path.join(key_path, f"epoch_{epoch}.png"))

    def __call__(self):
        # Dashboard
        self._dashboard = Dashboard(train_id=os.path.basename(self._path))

        # Launches the training procedure
        self._launch()
