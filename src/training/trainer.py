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

import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, DDIMPipeline, DDIMScheduler

# IMPORT: project
import paths
import utils

from src.loading import Loader
from .models import UNet
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
        _pipeline : Union[DDPMPipeline, DDIMPipeline]
            diffusion pipeline
        _optimizer : torch.optim.Optimizer
            pipeline's optimizer
        _scheduler : torch.nn.Module
            optimizer's scheduler
        _loss : Loss
            training's loss function

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
            os.makedirs(os.path.join(self._path, "images"))

        with open(os.path.join(self._path, "config.json"), 'w') as file_content:
            json.dump(self._params, file_content)

        # Loading
        self._data_loaders: Dict[str: DataLoader] = None

        # Pipeline
        self._pipeline: Union[DDPMPipeline, DDIMPipeline] = None

        # Optimizer and learning rate
        self._optimizer: diffusers.optimization.Optimizer = None
        self._scheduler: torch.nn.Module = None

        # Loss
        self._loss: torch.nn.Module = torch.nn.MSELoss().to(self._DEVICE)

        # Dashboard
        self._dashboard: Dashboard = Dashboard(params, train_id=os.path.basename(self._path))

    def _init_pipeline(self, weights_path: str) -> Union[DDPMPipeline, DDIMPipeline]:
        """
        Initializes the training's pipeline.

        Parameters
        ----------
            weights_path : str
                path to the model's weights

        Returns
        ----------
            Union[DDPMPipeline, DDIMPipeline]
                training's pipeline
        """
        # Get pipeline class
        pipeline_class = DDPMPipeline if self._params["pipeline_id"] == "DDPM" else DDIMPipeline
        scheduler_class = DDPMScheduler if self._params["pipeline_id"] == "DDPM" else DDIMScheduler

        # If pretrained load weights
        if weights_path is not None:
            return pipeline_class.from_pretrained(weights_path)

        # Else instantiate from scratch
        return pipeline_class(
            unet=UNet(self._params),
            scheduler=scheduler_class(num_train_timesteps=1000),
        ).to(self._DEVICE)

    def _launch(self):
        """ Launches the training. """
        time.sleep(1)

        p_bar: tqdm = tqdm(total=self._params["num_epochs"], desc="training in progress")
        for epoch in range(self._params["num_epochs"]):
            # Clear cache
            torch.cuda.empty_cache()

            # Learn
            self._run_epoch(p_bar, step="train")
            self._run_epoch(p_bar, step="valid")

            # Update
            self._dashboard.upload_values(self._scheduler.get_last_lr()[0])
            if epoch % 5 == 0:
                self._checkpoint(epoch)

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

            # Learn on batch
            loss, images = self._learn_on_batch(batch, batch_idx, learn=learning_allowed)

            # Store the loss value
            epoch_loss.append(loss)

            # Upload images generated using the batch on WandB
            if batch_idx % (num_batch // 2) == 0:
                self._dashboard.upload_images(images, step=step)

        # Store the results
        self._dashboard.update_loss(epoch_loss, step)

    def _learn_on_batch(self, batch: torch.Tensor, batch_idx: int, learn: bool = True):
        """
        Learns using data within a batch.

        Parameters
        ----------
            batch : torch.Tensor
                batch of tensors
            batch_idx : int
                batch's index
            learn : bool
                boolean indicating whether to train

        Returns
        ----------
            torch.Float
                loss calculated using batch's data

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()

    def _checkpoint(self, epoch: int):
        """
        Saves pipeline's weights.

        Parameters
        ----------
            epoch : int
                the current epoch idx
        """
        # Generates checkpoint images
        images: List[Image.Image] = self._pipeline(
            batch_size=8, generator=torch.manual_seed(0)
        ).images

        # Upload checkpoint images to WandB
        self._dashboard.upload_inference(images)

        # Saves checkpoint image on disk
        utils.save_image_as_plt(images, os.path.join(self._path, "images", f"epoch_{epoch}.png"))

        # Saves pipeline
        self._pipeline.save_pretrained(os.path.join(self._path, "pipeline"))

    def __call__(self, dataset_path: str, weights_path: str):
        """
        Parameters
        ----------
            dataset_path : str
                path to the dataset
            weights_path : str
                path to the model's weights
        """
        # Loading
        loader: Loader = Loader(self._params)
        self._data_loaders: Dict[str: DataLoader] = loader(dataset_path)

        # Pipeline
        self._pipeline: Union[DDPMPipeline, DDIMPipeline] = self._init_pipeline(weights_path)

        # Optimizer and learning rate
        self._optimizer: diffusers.optimization.Optimizer = torch.optim.AdamW(
            self._pipeline.unet.parameters(), lr=self._params["lr"]
        )
        self._scheduler: torch.nn.Module = diffusers.optimization.get_cosine_schedule_with_warmup(
            optimizer=self._optimizer,
            num_warmup_steps=self._params["lr_warmup_steps"],
            num_training_steps=(len(self._data_loaders["train"]) * self._params["num_epochs"]),
        )

        # Launch the training procedure
        self._launch()
