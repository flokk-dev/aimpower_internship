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

from transformers import CLIPTextModel
from diffusers import AutoencoderKL

# IMPORT: project
from .diffusion_components import DiffusionComponents


class StableDiffusionComponents(DiffusionComponents):
    """
    Represents a StableDiffusionComponents.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        noise_scheduler : diffusers.SchedulerMixin
            training's noise scheduler
        model : torch.nn.Module
            training's model
        optimizer : torch.optim.Optimizer
            training's optimizer
        lr_scheduler : torch.nn.Module
            learning rate's scheduler
        vae : AutoencoderKL
            auto encoder
        text_encoder : CLIPTextModel
            text encoder

    Methods
    ----------
        _init_data_loader
            Initializes the data loader
        _init_model
            Initializes the model
        _init_noise_scheduler
            Initializes the noise scheduler
        _init_optimizer
            Initializes the optimizer
        _init_lr_scheduler
            Initializes the learning rate's scheduler
        _init_vae
            Initializes an auto encoder
        _init_text_encoder
            Initializes a text encoder

        _to_device
            Sends the desired components on device
        prepare
            Prepares the components using an accelerator
    """

    def __init__(
            self,
            params: Dict[str, Any],
            dataset_path: str,
            num_epochs: int,
    ):
        """
        Instantiates a StableDiffusionComponents.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            dataset_path : str
                path to the dataset
            num_epochs : int
                number of epochs during the training
        """
        # ----- Mother Class ----- #
        super(StableDiffusionComponents, self).__init__(params, dataset_path, num_epochs)

        # ----- Attributes ----- #
        # VAE
        self.vae: AutoencoderKL = None
        self._init_vae()

        # Text encoder
        self.text_encoder: CLIPTextModel = None
        self._init_text_encoder()

    def _init_vae(
            self,
    ):
        """ Initializes an image encoder. """
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path=self._params["pipeline_path"],
            subfolder="vae",
            # revision=self._params["dtype"]
        )
        self.vae.requires_grad_(False)

    def _init_text_encoder(
            self
    ):
        """ Initializes a text encoder. """
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path=self._params["pipeline_path"],
            subfolder="text_encoder",
            # revision=self._params["dtype"]
        )
        self.text_encoder.requires_grad_(False)

    def _to_device(
            self
    ):
        """ Sends the desired components on device. """
        super()._to_device()

        self.vae.to(
            self.accelerator.device,
            # dtype=torch.float16 if self._params["dtype"] == "fp16" else torch.float32
        )

        self.text_encoder.to(
            self.accelerator.device,
            # dtype=torch.float16 if self._params["dtype"] == "fp16" else torch.float32
        )
