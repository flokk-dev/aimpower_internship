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
from PIL import Image

# IMPORT: deep learning
import torch
from torch.utils.data import DataLoader

# IMPORT: project
import paths
import utils

from src.loading import Loader, InfoLoader
from .learner import Learner, GuidedLearner
from .dashboard import Dashboard


class Trainer:
    """
    Represents a Trainer, that will be derived depending on the use case.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _data_loaders : Dict[str: DataLoader]
            training and validation data loaders

    Methods
    ----------
        _init_pipeline
            Initializes the training's pipeline
        _launch
            Launches the training
        _run_epoch
            Runs an epoch
        _learn_on_batch
            Learns using data within a batch
    """
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self, params: Dict[str, Any]):
        """
        Instantiates a Trainer.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
        """
        # Attributes
        self._params: Dict[str, Any] = params

        self._path = os.path.join(paths.MODELS_PATH, utils.get_datetime())
        if not os.path.exists(self._path):
            os.makedirs(self._path)
            os.mkdir(os.path.join(self._path, "images"))

        with open(os.path.join(self._path, "config.json"), 'w') as file_content:
            json.dump(self._params, file_content)

        # Components
        self._data_loaders: Dict[str: DataLoader] = None
        self._learner: Learner = None

        self._dashboard: Dashboard = None

    def _launch(self):
        """ Launches the training. """
        time.sleep(1)

        p_bar: tqdm = tqdm(total=self._params["num_epochs"], desc="training in progress")
        for epoch in range(self._params["num_epochs"]):
            # Clears cache
            torch.cuda.empty_cache()

            # Learns
            self._run_epoch(p_bar, step="train")
            self._run_epoch(p_bar, step="valid")

            # Updates
            self._dashboard.upload_values(self._learner.scheduler.get_last_lr()[0])
            if (epoch+1) % 10 == 0:
                self._checkpoint(epoch+1)

            p_bar.update(1)

        # End
        time.sleep(10)
        self._dashboard.shutdown()

    def _run_epoch(self, p_bar: tqdm, step: str):
        """
        Runs an epoch.

        Parameters
        ----------
            p_bar : tqdm
                the training's progress bar
            step : str
                training step
        """
        num_batch: int = len(self._data_loaders[step])
        learning_allowed: bool = step == "train"

        epoch_loss: list = list()
        for batch_idx, batch in enumerate(self._data_loaders[step]):
            p_bar.set_postfix(batch=f"{batch_idx}/{num_batch}")

            # Learns on batch
            loss, images = self._learner(batch, learn=learning_allowed)

            # Stores the loss value
            epoch_loss.append(loss)

            # Uploads images generated using the batch on WandB
            batch_modulo = num_batch // 2 if num_batch // 2 > 0 else 1
            if batch_idx % batch_modulo == 0:
                # self._dashboard.upload_images(images, step=step)
                pass

        # Stores the results
        self._dashboard.update_loss(epoch_loss, step)

    def _checkpoint(self, epoch: int):
        """
        Saves pipeline's weights.

        Parameters
        ----------
            epoch : int
                the current epoch idx
        """
        # Saves pipeline
        self._learner.pipeline.save_pretrained(os.path.join(self._path, "pipeline"))

        # Generates checkpoint images
        images: List[Image.Image] = self._learner.pipeline(
            batch_size=8, generator=torch.manual_seed(0)
        ).images

        # Uploads checkpoint images to WandB
        self._dashboard.upload_inference(images)

        # Saves checkpoint image on disk
        utils.save_image_as_plt(images, os.path.join(self._path, "images", f"epoch_{epoch}.png"))

    def __call__(self, dataset_path: str, weights_path: str):
        """
        Parameters
        ----------
            dataset_path : str
                path to the dataset
            weights_path : str
                path to the pipeline's weights
        """
        # Loading
        loader_class = Loader if self._params["train_type"] == "basic" else InfoLoader
        self._data_loaders = loader_class(self._params)(dataset_path)

        # Learner
        learner_class = Learner if self._params["train_type"] == "basic" else GuidedLearner
        self._learner = learner_class(self._params, len(self._data_loaders["train"]), weights_path)

        # Dashboard
        self._dashboard = Dashboard(self._params, train_id=os.path.basename(self._path))

        # Launches the training procedure
        self._launch()
