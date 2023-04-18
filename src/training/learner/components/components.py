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
    Represents a Components, that will be derived depending on the use case.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        model : torch.nn.Module
            training's model
        noise_scheduler : diffusers.SchedulerMixin
            training's noise scheduler
        optimizer : torch.optim.Optimizer
            training's optimizer
        lr_scheduler : torch.nn.Module
            learning rate's scheduler
        vae : diffusers.AutoencoderKL
            training's image encoder

    Methods
    ----------
        _init_vae : diffusers.AutoencoderKL
            Instantiates an image encoder
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
        self.vae: diffusers.AutoencoderKL = self._init_vae(
            params["vae"]["pipeline_path"]
        )

        # Model
        self.model: torch.nn.Module = ModelManager()(
            model_type=params["model"]["model_type"],
            model_params=params["model"]["args"],
            pipeline_path=params["model"]["pipeline_path"]
        ).to(self._DEVICE)

        # Noise scheduler
        self.noise_scheduler: diffusers.SchedulerMixin = NoiseSchedulerManager()(
            scheduler_type=params["noise_scheduler"]["noise_scheduler_type"],
            scheduler_params=params["noise_scheduler"]["args"],
            pipeline_path=params["noise_scheduler"]["pipeline_path"]
        )

        # Optimizer and learning rate
        self.optimizer: diffusers.optimization.Optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=params["lr"]
        )

        self.lr_scheduler: torch.nn.Module = \
            diffusers.optimization.get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=params["lr_warmup_steps"],
                num_training_steps=(num_batches * num_epochs)
            )

    def _init_vae(
            self,
            pipeline_path: str
    ) -> diffusers.AutoencoderKL:
        """
        Instantiates an image encoder.

        Parameters
        ----------
            pipeline_path : str
                path to the pretrained pipeline
        """
        if not pipeline_path:
            return None

        return diffusers.AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path=pipeline_path,
            subfolder="vae"
        ).to(self._DEVICE)
