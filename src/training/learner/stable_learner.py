"""
Creator: Flokk
Date: 09/04/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import torch

# IMPORT: deep learning
import diffusers

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

    def _learn(
            self,
            batch: Dict[str, torch.Tensor],
    ) -> float:
        """
        Learns on a batch of data.

        Parameters
        ----------
            batch : Dict[str, torch.Tensor]
                batch of data

        Returns
        ----------
            float
                loss value computed using batch's data
        """
        if self._params["reduce_dimensions"]:
            batch["image"] = self._encode_image(
                batch["image"].type(torch.float32).to(self._DEVICE)
            )

        return super()._learn(batch)

    def _encode_image(
            self,
            image: torch.Tensor
    ) -> torch.Tensor:
        """
        Reduces tensor's dimensions using a VAE.

        Parameters
        ----------
            image : torch.Tensor
                image to encode

        Returns
        ----------
            torch.Tensor
                encoded image
        """
        with torch.no_grad():
            return self.components.vae.encode(image).latent_dist.sample() * \
                self.components.vae.config.scaling_factor

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

    def _init_pipeline(self):
        """
        Initializes a pipeline using the training's components.

        Returns
        ----------
            diffusers.StableDiffusionPipeline
                diffusion pipeline
        """
        pipeline = diffusers.StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=self._params["components"]["vae"]["pipeline_path"],
            unet=self.components.model
        ).to(self._DEVICE)

        pipeline.set_progress_bar_config(disable=True)
        pipeline.safety_checker = None

        return pipeline
