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
from torch.utils.data import DataLoader

# IMPORT: project
import paths
import utils

from src.loading import Loader

from src.training.pipeline import Pipeline
from src.training.dashboard import Dashboard


class Trainer:
    """
    Represents a Trainer, which will be modified depending on the use case.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _data_loader : Dict[str: DataLoader]
            loader containing training's data

    Methods
    ----------
        _launch
            Launches the training
        _run_epoch
            Runs an epoch
        _checkpoint
            Saves noise_scheduler's weights
    """
    _PIPELINES = dict()

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
        self._verify_parameters(params)
        self._params: Dict[str, Any] = params

        # Creates training's repository
        self._path = os.path.join(paths.MODELS_PATH, utils.get_datetime())
        if not os.path.exists(self._path):
            os.makedirs(self._path)
            os.mkdir(os.path.join(self._path, "images"))

        with open(os.path.join(self._path, "config.json"), 'w') as file_content:
            json.dump(self._params, file_content)

        # Components
        self._data_loader: DataLoader = None
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
            # Clears cache
            torch.cuda.empty_cache()

            # Learns
            self._pipeline.components.model.train()
            self._run_epoch(p_bar)

            # Updates
            self._dashboard.upload_values(self._pipeline.components.lr_scheduler.get_last_lr()[0])
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
        num_batch: int = len(self._data_loader)

        epoch_loss: list = list()
        for batch_idx, batch in enumerate(self._data_loader):
            p_bar.set_postfix(batch=f"{batch_idx}/{num_batch}", gpu=utils.gpu_utilization())

            # Learns on batch
            epoch_loss.append(self._pipeline.learn(batch))

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
        # Saves pipeline
        # self._pipeline().save_pretrained(os.path.join(self._path, "pipeline"))

        # Generates checkpoint images
        tensors: Dict[str, torch.Tensor] = self._pipeline.inference()

        # Uploads and saves qualitative results
        for key, tensor in tensors.items():
            # Creates destination directory
            key_path = os.path.join(self._path, "images", key)
            if not os.path.exists(key_path):
                os.makedirs(key_path)

            # Uploads checkpoint images to WandB
            self._dashboard.upload_inference(key, tensor)

            # Saves checkpoint image on disk
            utils.save_plt(tensor, os.path.join(key_path, f"epoch_{epoch}.png"))

    def __call__(self, dataset_path: str):
        """
        Parameters
        ----------
            dataset_path : str
                path to the dataset
        """
        # Loading
        self._data_loader = Loader(self._params["loader"])(dataset_path)

        # Pipeline
        self._pipeline = self._PIPELINES[self._params["pipeline"]["pipeline_type"]](
            self._params["pipeline"], len(self._data_loader), self._params["num_epochs"]
        )

        # Dashboard
        self._dashboard = Dashboard(self._params, train_id=os.path.basename(self._path))

        # Launches the training procedure
        self._launch()
