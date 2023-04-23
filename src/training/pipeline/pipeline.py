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
from diffusers import DiffusionPipeline


class Pipeline:
    """
    Represents a Pipeline.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the pipeline behaviour

    Methods
    ----------
        inference: Dict[str, torch.Tensor]
            Builds the pipeline using its components
    """

    def __init__(
            self,
            params: Dict[str, Any]
    ):
        """
        Instantiates a Pipeline.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the pipeline behaviour
        """
        # ----- Attributes ----- #
        self._params: Dict[str, Any] = params

    def _inference(
            self,
            pipeline: DiffusionPipeline
    ) -> Dict[str, torch.Tensor]:
        """
        Generates images using a diffusion pipeline.

        Parameters
        ----------
            pipeline : DiffusionPipeline
                components needed to instantiate the pipeline

        Returns
        ----------
            Dict[str, torch.Tensor]
                generated image

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()

    def checkpoint(
        self,
        components,
        save_path: str
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
            components
                components needed to generate images
            save_path: str
                path where to save the pipeline

        Returns
        ----------
            Dict[str, torch.Tensor]
                generated image

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()
