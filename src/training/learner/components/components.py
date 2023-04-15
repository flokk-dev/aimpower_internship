"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: deep learning
import torch
import diffusers

# IMPORT: project
from .noise_scheduler import NoiseSchedulerManager
from .model import ModelManager


class Components:
    """
    Represents a general Components, that will be derived depending on the use case.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _vae : torch.nn.Module
            training's VAE
        _model : torch.nn.Module
            training's model
        _noise_scheduler : diffusers.SchedulerMixin
            training's noise scheduler
        _optimizer : torch.optim.Optimizer
            training's optimizer
        _lr_scheduler : torch.nn.Module
            learning rate's scheduler

    Methods
    ----------
        vae : torch.nn.Module
            Returns the training's VAE
        model : Dict[str, Any]
            Returns the training's model
        noise_scheduler : torch.nn.Module
            Returns the training's noise scheduler
        optimizer : diffusers.SchedulerMixin
            Returns the training's optimizer
        lr_scheduler : torch.optim.Optimizer
            Returns the learning rate's scheduler
    """
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self, params: Dict[str, Any], weights_path: str, num_batches: int):
        """
        Instantiates a Components.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            weights_path : str
                path to the noise_scheduler's weights
            num_batches : int
                number of batches within the data loader
        """
        # Attributes
        self._params: Dict[str, Any] = params

        # VAE
        self._vae = diffusers.AutoencoderKL.from_pretrained(
            weights_path, subfolder="vae"
        ).to(self._DEVICE)

        # Model
        self._model: torch.nn.Module = ModelManager(self._params)(
            self._params["model_id"], weights_path
        ).to(self._DEVICE)

        # Noise scheduler
        self._noise_scheduler: diffusers.SchedulerMixin = NoiseSchedulerManager(self._params)(
            self._params["noise_scheduler_id"], weights_path
        ).to(self._DEVICE)

        # Optimizer and learning rate
        self._optimizer: diffusers.optimization.Optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self._params["lr"]
        )
        self._lr_scheduler: torch.nn.Module = \
            diffusers.optimization.get_cosine_schedule_with_warmup(
                optimizer=self._optimizer,
                num_warmup_steps=self._params["lr_warmup_steps"],
                num_training_steps=(num_batches * self._params["num_epochs"]),
            )

    @property
    def vae(self) -> diffusers.AutoencoderKL:
        """
        Returns the training's VAE.

        Returns
        ----------
            diffusers.AutoencoderKL
                training's VAE
        """
        return self._vae

    @property
    def model(self) -> torch.nn.Module:
        """
        Returns the training's model.

        Returns
        ----------
            torch.nn.Module
                training's model
        """
        return self._model

    @property
    def noise_scheduler(self) -> diffusers.SchedulerMixin:
        """
        Returns the training's noise scheduler.

        Returns
        ----------
            diffusers.SchedulerMixin
                training's noise scheduler
        """
        return self._noise_scheduler

    @property
    def optimizer(self) -> diffusers.optimization.Optimizer:
        """
        Returns the training's optimizer.

        Returns
        ----------
            diffusers.optimization.Optimizer
                training's optimizer
        """
        return self._optimizer

    @property
    def lr_scheduler(self) -> torch.nn.Module:
        """
        Returns the learning rate's scheduler.

        Returns
        ----------
            torch.nn.Module
                learning rate's scheduler
        """
        return self._model
