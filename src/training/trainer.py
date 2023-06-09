"""
Creator: Flokk
Date: 09/04/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

import os
import time
import json
from tqdm import tqdm

# IMPORT: deep learning
import torch

# IMPORT: project
import paths
import utils

from .learner import Learner, \
    DiffusionLearner, GuidedDiffusionLearner, StableDiffusionLearner

from .pipeline import Pipeline, \
    DiffusionPipeline, GuidedDiffusionPipeline, StableDiffusionPipeline, LoRADiffusionPipeline

from src.training.dashboard import Dashboard


class Trainer:
    """
    Represents a Trainer, which will be modified depending on the use case.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
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
            Saves noise_scheduler's weights
    """
    _LEARNERS = {
        "diffusion": DiffusionLearner,
        "guided diffusion": GuidedDiffusionLearner,
        "stable diffusion": StableDiffusionLearner
    }

    _PIPELINES = {
        "diffusion": DiffusionPipeline,
        "guided diffusion": GuidedDiffusionPipeline,
        "stable diffusion": StableDiffusionPipeline,
        "lora diffusion": LoRADiffusionPipeline
    }

    def __init__(
            self,
            params: Dict[str, Any]
    ):
        """
        Instantiates a Trainer.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
        """
        # Attributes
        self._params: Dict[str, Any] = params

        # Creates training's repository
        self._path = os.path.join(paths.MODELS_PATH, utils.get_datetime())
        if not os.path.exists(self._path):
            os.makedirs(self._path)
            os.mkdir(os.path.join(self._path, "images"))

        with open(os.path.join(self._path, "config.json"), 'w') as file_content:
            json.dump(self._params, file_content)

        # Components
        self._learner: Learner = None
        self._pipeline: Pipeline = None

        self._dashboard: Dashboard = None

    def _verify_parameters(
            self,
            params: Dict[str, Any]
    ):
        """
        Verifies if the training's configuration is correct.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()

    def _launch(
            self
    ):
        """ Launches the training. """
        time.sleep(1)

        p_bar: tqdm = tqdm(total=self._params["num_epochs"], desc="training in progress")
        for epoch in range(self._params["num_epochs"]):
            self._learner.components.model.train()

            # Learns
            self._learner.components.model.train()
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

    def _checkpoint(
            self,
            epoch: int
    ):
        """
        Saves noise_scheduler's weights.

        Parameters
        ----------
            epoch : int
                the current epoch idx
        """
        # Saves pipeline and generates images
        tensors: Dict[str, torch.Tensor] = self._pipeline.checkpoint(
            components=self._learner.components,
            save_path=self._path
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

    def __call__(self, dataset_path: str):
        """
        Parameters
        ----------
            dataset_path : str
                path to the dataset
        """
        # Learner
        self._learner = self._LEARNERS[self._params["types"]["learner"]](
            self._params, dataset_path, self._params["num_epochs"]
        )

        # Pipeline
        self._pipeline = self._PIPELINES[self._params["types"]["pipeline"]](
            self._params
        )

        # Dashboard
        self._dashboard = Dashboard(train_id=os.path.basename(self._path))

        # Launches the training procedure
        self._launch()
