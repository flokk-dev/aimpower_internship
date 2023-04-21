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
        self.model: torch.nn.Module = self._init_model()

        # Noise scheduler
        self.noise_scheduler: diffusers.SchedulerMixin = self._init_noise_scheduler()

        # Optimizer and learning rate
        self.optimizer, self.lr_scheduler = self._init_optimizer(num_epochs, num_batches)

    def _init_model(
            self
    ) -> torch.nn.Module:
        """
        Initializes a model.

        Returns
        ----------
            torch.nn.Module
                model
        """
        return ModelManager()(
            model_type=self._params["model"]["model_type"],
            model_params=self._params["model"]["args"],
            pipeline_path=self._params["pipeline_path"] if self._params["model"]["load"] else None
        ).to(self._DEVICE)

    def _init_noise_scheduler(
            self
    ) -> diffusers.SchedulerMixin:
        """
        Initializes a noise scheduler.

        Returns
        ----------
            diffusers.SchedulerMixin
                noise scheduler
        """
        return NoiseSchedulerManager()(
            scheduler_type=self._params["noise_scheduler"]["noise_scheduler_type"],
            scheduler_params=self._params["noise_scheduler"]["args"],
            pipeline_path=self._params["pipeline_path"] if self._params["noise_scheduler"]["load"] else None
        )

    def _init_optimizer(
            self,
            num_epochs: int,
            num_batches: int
    ) -> Tuple[diffusers.optimization.Optimizer, torch.nn.Module]:
        """
        Instantiates an optimizer and a scheduler.

        Parameters
        ----------
            num_epochs : int
                number of epochs during the training
            num_batches : int
                number of batches within the data loader

        Returns
        ----------
            Tuple[diffusers.optimization.Optimizer, torch.nn.Module]
                optimizer and scheduler
        """
        optimizer: diffusers.optimization.Optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self._params["optimizer"]["lr"]
        )

        lr_scheduler: torch.nn.Module = \
            diffusers.optimization.get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self._params["optimizer"]["lr_warmup_steps"],
                num_training_steps=(num_batches * num_epochs)
            )

        return optimizer, lr_scheduler
