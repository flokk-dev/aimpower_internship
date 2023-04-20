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


class ComponentsV1:
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
    """
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(
            self,
            params: Dict[str, Any],
            num_epochs: int,
            num_batches: int
    ):
        """
        Instantiates a ComponentsV1.

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

        # Model
        self.model: torch.nn.Module = ModelManager()(
            model_type=params["model"]["model_type"],
            model_params=params["model"]["args"],
            pipeline_path=params["pipeline_path"] if params["model"]["load"] else None
        ).to(self._DEVICE)

        # Noise scheduler
        self.noise_scheduler: diffusers.SchedulerMixin = NoiseSchedulerManager()(
            scheduler_type=params["noise_scheduler"]["noise_scheduler_type"],
            scheduler_params=params["noise_scheduler"]["args"],
            pipeline_path=params["pipeline_path"] if params["noise_scheduler"]["load"] else None
        )

        # Optimizer and learning rate
        self.optimizer: diffusers.optimization.Optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=params["optimizer"]["lr"]
        )

        self.lr_scheduler: torch.nn.Module = \
            diffusers.optimization.get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=params["optimizer"]["lr_warmup_steps"],
                num_training_steps=(num_batches * num_epochs)
            )
