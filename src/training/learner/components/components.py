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

    def __init__(
            self,
            params: Dict[str, Any],
            num_epochs: int,
            num_batches: int
    ):
        """
        Instantiates a Components.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            num_epochs : int
                number of epochs during the training
            num_batches : int
                number of batches within the data loader
        """
        # Attributes
        self._params: Dict[str, Any] = params

        # VAE
        self._vae = self._init_vae(params["vae"]["pipeline_path"]).to(self._DEVICE)

        # Model
        self._model: torch.nn.Module = ModelManager()(
            model_type=params["model"]["type"],
            model_params=params["model"]["args"],
            pipeline_path=params["model"]["pipeline_path"]
        ).to(self._DEVICE)

        # Noise scheduler
        self._noise_scheduler: diffusers.SchedulerMixin = NoiseSchedulerManager()(
            scheduler_type=params["noise_scheduler"]["type"],
            scheduler_params=params["noise_scheduler"]["args"],
            pipeline_path=params["noise_scheduler"]["pipeline_path"]
        ).to(self._DEVICE)

        # Optimizer and learning rate
        self._optimizer: diffusers.optimization.Optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=params["lr"]
        )
        self._lr_scheduler: torch.nn.Module = \
            diffusers.optimization.get_cosine_schedule_with_warmup(
                optimizer=self._optimizer,
                num_warmup_steps=params["lr_warmup_steps"],
                num_training_steps=(num_batches * num_epochs)
            )

    @property
    def vae(
            self
    ) -> diffusers.AutoencoderKL:
        """
        Returns the training's VAE.

        Returns
        ----------
            diffusers.AutoencoderKL
                training's VAE
        """
        return self._vae

    @staticmethod
    def _init_vae(
            pipeline_path: str
    ):
        """
        Instantiates a VAE.

        Parameters
        ----------
            pipeline_path : str
                path to the pretrained pipeline
        """
        if not pipeline_path:
            pipeline_path = "CompVis/stable-diffusion-v1-4"

        return diffusers.AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path=pipeline_path,
            subfolder="vae"
        )

    @property
    def model(
            self
    ) -> torch.nn.Module:
        """
        Returns the training's model.

        Returns
        ----------
            torch.nn.Module
                training's model
        """
        return self._model

    @property
    def noise_scheduler(
            self
    ) -> diffusers.SchedulerMixin:
        """
        Returns the training's noise scheduler.

        Returns
        ----------
            diffusers.SchedulerMixin
                training's noise scheduler
        """
        return self._noise_scheduler

    @property
    def optimizer(
            self
    ) -> diffusers.optimization.Optimizer:
        """
        Returns the training's optimizer.

        Returns
        ----------
            diffusers.optimization.Optimizer
                training's optimizer
        """
        return self._optimizer

    @property
    def lr_scheduler(
            self
    ) -> torch.nn.Module:
        """
        Returns the learning rate's scheduler.

        Returns
        ----------
            torch.nn.Module
                learning rate's scheduler
        """
        return self._model
