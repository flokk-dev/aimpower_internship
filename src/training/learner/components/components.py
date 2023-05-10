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

from torch.optim import Optimizer, AdamW
from diffusers.optimization import get_cosine_schedule_with_warmup

# IMPORT: project
from .component import Components


class ClassicComponents(Components):
    """
    Represents a ClassicComponents.

    Attributes
    ----------
        _config : Dict[str, Any]
            configuration needed to adjust the program behaviour
        data_loader : PromptDataLoader
            data loader
        model : torch.nn.Module
            training's model
        lora_layers : AttnProcsLayers
            LoRA attention layers
        noise_scheduler : diffusers.SchedulerMixin
            training's noise scheduler
        vae : AutoencoderKL
            auto encoder
        text_encoder : CLIPTextModel
            text encoder
        optimizer : torch.optim.Optimizer
            training's optimizer
        lr_scheduler : torch.nn.Module
            learning rate's scheduler

    Methods
    ----------
        _init_data_loader : PromptDataLoader
            Initializes the data loader
        _init_model : UNet2DConditionModel
            Initializes the model
        _init_lora_layers : AttnProcsLayers
            Initializes the LoRA layers
        _init_noise_scheduler : SchedulerMixin
            Initializes the noise scheduler
        _init_vae : AutoencoderKL
            Initializes an auto encoder
        _init_text_encoder : CLIPTextModel
            Initializes a text encoder
        _init_optimizer : AdamW
            Initializes the optimizer
        _init_lr_scheduler : torch.nn.Module
            Initializes the learning rate's scheduler

        _to_device
            Sends the desired components on device
        prepare
            Prepares the components using an accelerator
    """
    def __init__(
            self,
            config: Dict[str, Any],
            dataset_path: str
    ):
        """
        Instantiates a ClassicComponents.

        Parameters
        ----------
            config : Dict[str, Any]
                configuration needed to adjust the program behaviour
            dataset_path : str
                path to the dataset
        """
        # ----- Mother class ----- #
        super(ClassicComponents, self).__init__(config, dataset_path)

        # ----- Attributes ----- #

        # Optimizer
        self.optimizer: Optimizer = self._init_optimizer()

        # Learning rate
        self.lr_scheduler: torch.nn.Module = self._init_lr_scheduler()

    def _init_optimizer(
            self
    ) -> AdamW:
        """
        Initializes the optimizer.

        Returns
        ----------
            AdamW
                training optimizer
        """
        return torch.optim.AdamW(
            self.lora_layers.parameters(), **self._config["optimizer"]["args"]
        )

    def _init_lr_scheduler(
            self
    ) -> torch.nn.Module:
        """
        Instantiates an optimizer and a scheduler.

        Returns
        ----------
            torch.nn.Module
                learning rate's scheduler
        """
        return get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_training_steps=(len(self.data_loader) * self._config["num_epochs"]),
            **self._config["lr_scheduler"]["args"],
        )


class ReinforcementComponents(Components):
    """
    Represents a ReinforcementComponents.

    Attributes
    ----------
        _config : Dict[str, Any]
            configuration needed to adjust the program behaviour
        data_loader : PromptDataLoader
            data loader
        model : torch.nn.Module
            training's model
        lora_layers : AttnProcsLayers
            LoRA attention layers
        noise_scheduler : diffusers.SchedulerMixin
            training's noise scheduler
        vae : AutoencoderKL
            auto encoder
        text_encoder : CLIPTextModel
            text encoder
        optimizer : torch.optim.Optimizer
            training's optimizer
        lr_scheduler : torch.nn.Module
            learning rate's scheduler

    Methods
    ----------
        _init_data_loader : PromptDataLoader
            Initializes the data loader
        _init_model : UNet2DConditionModel
            Initializes the model
        _init_lora_layers : AttnProcsLayers
            Initializes the LoRA layers
        _init_noise_scheduler : SchedulerMixin
            Initializes the noise scheduler
        _init_vae : AutoencoderKL
            Initializes an auto encoder
        _init_text_encoder : CLIPTextModel
            Initializes a text encoder
        _init_optimizer : AdamW
            Initializes the optimizer
        _init_lr_scheduler : torch.nn.Module
            Initializes the learning rate's scheduler

        _to_device
            Sends the desired components on device
        prepare
            Prepares the components using an accelerator
    """
    def __init__(
            self,
            config: Dict[str, Any],
            dataset_path: str
    ):
        """
        Instantiates a ReinforcementComponents.

        Parameters
        ----------
            config : Dict[str, Any]
                configuration needed to adjust the program behaviour
            dataset_path : str
                path to the dataset
        """
        # ----- Mother class ----- #
        super(ReinforcementComponents, self).__init__(config, dataset_path)

        # ----- Attributes ----- #
        # Optimizer
        self.optimizer: Optimizer = self._init_optimizer()

        # Learning rate scheduler
        self.lr_scheduler: torch.nn.Module = self._init_lr_scheduler()

    def _init_optimizer(
            self
    ) -> AdamW:
        """
        Initializes the optimizer.

        Returns
        ----------
            AdamW
                training optimizer
        """
        # Not implemented
        raise NotImplementedError()

    def _init_lr_scheduler(
            self
    ) -> torch.nn.Module:
        """
        Instantiates an optimizer and a scheduler.

        Returns
        ----------
            torch.nn.Module
                learning rate's scheduler
        """
        # Not implemented
        raise NotImplementedError()
