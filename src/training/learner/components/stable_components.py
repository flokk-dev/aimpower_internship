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
from .components import Components


class StableComponents(Components):
    """
    Represents an StableComponents.

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
            Instantiates an image encoder
        _init_text_encoder : transformers.CLIPTextModel
            Instantiates a text encoder
    """
    def __init__(
            self,
            params: Dict[str, Any],
            num_epochs: int,
            num_batches: int
    ):
        """
        Instantiates a StableComponents.

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
        super(StableComponents, self).__init__(params, num_epochs, num_batches)

        # VAE
        self.vae: diffusers.AutoencoderKL = self._init_vae(
            params["vae"]["pipeline_path"]
        )

        # Text encoder
        self.text_encoder: transformers.CLIPTextModel = self._init_text_encoder(
            params["text_encoder"]["pipeline_path"]
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

    def _init_text_encoder(
            self,
            pipeline_path: str
    ) -> transformers.CLIPTextModel:
        """
        Instantiates a text encoder.

        Parameters
        ----------
            pipeline_path : str
                path to the pretrained pipeline
        """
        if not pipeline_path:
            return None

        return transformers.CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path=pipeline_path,
            subfolder="text_encoder"
        ).to(self._DEVICE)
