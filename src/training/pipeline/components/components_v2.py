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

import transformers
import diffusers

# IMPORT: project
from .components_v1 import ComponentsV1


class ComponentsV2(ComponentsV1):
    """
    Represents an ComponentsV2.

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
        text_encoder : transformers.CLIPTextModel
            training's text encoder

    Methods
    ----------
        _init_vae : diffusers.AutoencoderKL
            Initializes an image encoder
        _init_text_encoder : transformers.CLIPTextModel
            Initializes a text encoder
    """
    def __init__(
            self,
            params: Dict[str, Any],
            num_epochs: int,
            num_batches: int
    ):
        """
        Instantiates a ComponentsV2.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            num_epochs : int
                number of epochs during the training
            num_batches : int
                number of batches within the data loader
        """
        # Mother class
        super(ComponentsV2, self).__init__(params, num_epochs, num_batches)

        # VAE
        self.vae: diffusers.AutoencoderKL = self._init_vae(
            self._params["vae"]["load"]
        )
        self.vae.requires_grad_(False)

        # Text encoder
        self.text_encoder: transformers.CLIPTextModel = self._init_text_encoder(
            self._params["text_encoder"]["load"]
        )
        self.text_encoder.requires_grad_(False)

    def _init_vae(
            self,
            load: bool
    ) -> diffusers.AutoencoderKL:
        """
        Initializes an image encoder.

        Parameters
        ----------
            load : bool
                whether to load pretrained weights or not

        Returns
        ----------
            diffusers.AutoencoderKL
                image encoder
        """
        if not load:
            return None

        return diffusers.AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path=self._params["pipeline_path"],
            subfolder="vae"
        ).to(self._DEVICE)

    def _init_text_encoder(
            self,
            load: bool
    ) -> transformers.CLIPTextModel:
        """
        Initializes a text encoder.

        Parameters
        ----------
            load : bool
                whether to load pretrained weights or not

        Returns
        ----------
            transformers.CLIPTextModel
                text encoder
        """
        if not load:
            return None

        return transformers.CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path=self._params["pipeline_path"],
            subfolder="text_encoder"
        ).to(self._DEVICE)
