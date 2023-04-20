"""
Creator: Flokk
Date: 09/04/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

import diffusers
import torch

# IMPORT: project
from .learner import Learner


class StableLearner(Learner):
    """
    Represents an StableLearner, which will be modified depending on the use case.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _loss : Loss
            training's loss function
        components : Components
            training's components

    Methods
    ----------
        _learn
            Learns on a batch of data
        _forward
            Extracts noise within the noisy image using the noise_scheduler
        _add_noise
            Adds noise to a given tensor
        _encode_image
            Reduces tensor's dimension using a VAE
        _encode_text
            Encodes a text into a tensor using a CLIP
        inference
            Generates and image using the training's noise_scheduler
    """

    def __init__(
            self,
            params: Dict[str, Any],
    ):
        """
        Instantiates an StableLearner.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
        """
        # Mother class
        super(StableLearner, self).__init__(params)

    def _encode_text(
            self,
            text: torch.Tensor
    ) -> torch.Tensor:
        """
        Encodes a text into a tensor using a CLIP.

        Parameters
        ----------
            text : torch.Tensor
                text to encode

        Returns
        ----------
            torch.Tensor
                encoded text
        """
        with torch.no_grad():
            return self.components.text_encoder(text)[0]