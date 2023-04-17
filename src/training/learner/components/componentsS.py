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


class AdvancedComponents(Components):
    """
    Represents an ConditionedComponents.

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
        _text_encoder : torch.nn.Module
            training's text encoder
        _feature_extractor : torch.nn.Module
            training's feature extractor

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
        text_encoder : torch.nn.Module
            Returns the training's text encoder
        feature_extractor : torch.nn.Module
            Returns the training's feature extractor
    """
    def __init__(
            self,
            params: Dict[str, Any],
            num_epochs: int,
            num_batches: int
    ):
        """
        Instantiates a AdvancedComponents.

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
        super(AdvancedComponents, self).__init__(params, num_epochs, num_batches)

        # Text encoder
        self._text_encoder = self._init_text_encoder(params["text_encoder"]["pipeline_path"])

        # Feature extractor
        self._feature_extractor = self._init_feature_extractor(
            params["feature_extractor"]["pipeline_path"]
        )

    @property
    def text_encoder(self) -> transformers.CLIPTextModel:
        """
        Returns the training's text encoder.

        Returns
        ----------
            transformers.CLIPTextModel
                training's text encoder
        """
        return self._text_encoder

    @staticmethod
    def _init_text_encoder(
            pipeline_path: str
    ):
        """
        Instantiates a text encoder.

        Parameters
        ----------
            pipeline_path : str
                path to the pretrained pipeline
        """
        if not pipeline_path:
            pipeline_path = "CompVis/stable-diffusion-v1-4"

        return transformers.CLIPTextModel.from_pretrained(
            pipeline_path, subfolder="text_encoder"
        )

    @property
    def feature_extractor(self) -> transformers.CLIPFeatureExtractor:
        """
        Returns the training's feature extractor.

        Returns
        ----------
            transformers.CLIPFeatureExtractor
                training's feature extractor
        """
        return self._feature_extractor

    @staticmethod
    def _init_feature_extractor(
            pipeline_path: str
    ):
        """
        Instantiates a training's VAE.

        Parameters
        ----------
            pipeline_path : str
                path to the pretrained pipeline
        """
        if not pipeline_path:
            pipeline_path = "openai/clip-vit-base-patch32"

        return transformers.CLIPFeatureExtractor.from_pretrained(pipeline_path)
