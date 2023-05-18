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
        accelerator : Accelerator
            accelerator needed to improve the training
        data_loader : PromptDataLoader
            data loader
        pipeline : StableDiffusionPipeline
            pipeline to train
        lora_layers : AttnProcsLayers
            model's LoRA layers
        optimizer : torch.optim.Optimizer
            training's optimizer
        lr_scheduler : torch.nn.Module
            learning rate's scheduler

    Methods
    ----------
        model : UNet2DConditionModel
            Returns pipeline's model
        noise_scheduler : SchedulerMixin
            Returns pipeline's noise scheduler
        vae : AutoencoderKL
            Returns pipeline's vae
        text_encoder : CLIPTextModel
            Returns pipeline's text encoder

        _create_lora_layers : AttnProcsLayers
            Initializes the LoRA layers
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
        self.optimizer: Optimizer = torch.optim.AdamW(
            self.lora_layers.parameters(), **self._config["optimizer"]["args"]
        )

        # Learning rate scheduler
        self.lr_scheduler: torch.nn.Module = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_training_steps=(len(self.data_loader) * self._config["num_epochs"]),
            **self._config["lr_scheduler"]["args"],
        )

    def prepare(
            self
    ):
        """ Prepares the components using an accelerator. """
        self.lora_layers, self.optimizer, self.data_loader, self.lr_scheduler = \
            self.accelerator.prepare(
                self.lora_layers, self.optimizer, self.data_loader, self.lr_scheduler
            )


class ReinforcementComponents(Components):
    """
    Represents a ReinforcementComponents.

    Attributes
    ----------
        _config : Dict[str, Any]
            configuration needed to adjust the program behaviour
        accelerator : Accelerator
            accelerator needed to improve the training
        data_loader : PromptDataLoader
            data loader
        pipeline : StableDiffusionPipeline
            pipeline to train
        lora_layers : AttnProcsLayers
            model's LoRA layers

    Methods
    ----------
        model : UNet2DConditionModel
            Returns pipeline's model
        noise_scheduler : SchedulerMixin
            Returns pipeline's noise scheduler
        vae : AutoencoderKL
            Returns pipeline's vae
        text_encoder : CLIPTextModel
            Returns pipeline's text encoder

        _create_lora_layers : AttnProcsLayers
            Initializes the LoRA layers
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
        super(ReinforcementComponents, self).__init__(config, dataset_path)

    def prepare(
            self
    ):
        """ Prepares the components using an accelerator. """
        self.lora_layers, self.data_loader = self.accelerator.prepare(
            self.lora_layers, self.data_loader
        )
