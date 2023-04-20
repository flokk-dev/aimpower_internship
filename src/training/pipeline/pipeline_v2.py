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
from transformers import CLIPFeatureExtractor

import diffusers
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

# IMPORT: project
import utils

from .pipeline_v1 import PipelineV1
from .components import ComponentsV2


class PipelineV2(PipelineV1):
    """
    Represents an PipelineV2, which will be modified depending on the use case.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _loss : Loss
            training's loss function
        components : ComponentsV2
            training's components

    Methods
    ----------
        _init_components
            Initializes the pipeline's components
        learn
            Learns on a batch of data
        _forward
            Extracts noise within the noisy image using the noise_scheduler
        _encode_image
            Reduces tensor's dimension using a VAE
        _encode_text
            Encodes a text into a tensor using a CLIP
        _add_noise
            Adds noise to a given tensor
        inference
            Generates and image using the training's noise_scheduler
        _build
            Builds the hugging face pipeline using the pipeline's components
    """

    def __init__(
            self,
            params: Dict[str, Any],
            num_epochs: int,
            num_batches: int
    ):
        """
        Instantiates a LearnerV2.

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
        super(PipelineV2, self).__init__(params, num_epochs, num_batches)

    def _init_components(
            self,
            params: Dict[str, Any],
            num_epochs: int,
            num_batches: int
    ) -> ComponentsV2:
        """
        Initializes the pipeline's components.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            num_epochs : int
                number of epochs during the training
            num_batches : int
                number of batches within the data loader

        Returns
        ----------
            ComponentsV2
                pipeline's components
        """
        return ComponentsV2(params["components"], num_epochs, num_batches)

    def learn(
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

        return super().learn(batch)

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

    def __call__(self) -> diffusers.StableDiffusionPipeline:
        """
        Returns
        ----------
            diffusers.StableDiffusionPipeline
                diffusion pipeline
        """
        pipeline = diffusers.StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=self._params["components"]["vae"]["pipeline_path"],
            unet=self.components.model
        ).to(self._DEVICE)

        pipeline.safety_checker = None

        # Returns
        return pipeline


class StablePipeline(PipelineV2):
    """
    Represents a StablePipeline.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _loss : Loss
            training's loss function
        components : ComponentsV2
            training's components

    Methods
    ----------
        _init_components
            Initializes the pipeline's components
        learn
            Learns on a batch of data
        _forward
            Extracts noise within the noisy image using the noise_scheduler
        _encode_image
            Reduces tensor's dimension using a VAE
        _encode_text
            Encodes a text into a tensor using a CLIP
        _add_noise
            Adds noise to a given tensor
        inference
            Generates and image using the training's noise_scheduler
        _build
            Builds the hugging face pipeline using the pipeline's components
    """
    def __init__(
            self,
            params: Dict[str, Any],
            num_epochs: int,
            num_batches: int
    ):
        """
        Instantiates a StablePipeline.

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
        super(StablePipeline, self).__init__(params, num_epochs, num_batches)

    def _forward(
            self,
            batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts noise within the noisy image using the noise_scheduler.

        Parameters
        ----------
            batch : Dict[str, torch.Tensor]
                batch of data

        Returns
        ----------
            torch.Tensor
                added noise
            torch.Tensor
                extracted noise
        """
        # Image
        batch["image"]: torch.Tensor = batch["image"].type(torch.float32).to(self._DEVICE)

        # Conditioning
        condition: torch.Tensor = torch.randn(
            batch["image"].shape[0],
            self._params["components"]["model"]["args"]["sequence_length"],
            self._params["components"]["model"]["args"]["feature_dim"]
        ).to(self._DEVICE)

        # Predicts added noise
        noisy_image, noise, timestep = self._add_noise(batch["image"])
        return noise, self.components.model(noisy_image, timestep, condition).sample

    def inference(
            self,
    ) -> Dict[str, torch.Tensor]:
        """
        Generates and image using the training's components.

        Returns
        ----------
            Dict[str, torch.Tensor]
                generated image
        """
        pipeline = self()
        prompts = ["", "", "", "", ""]

        # Generates images
        generated_images = pipeline(
            prompts,
            num_inference_steps=50,
            generator=torch.manual_seed(0)
        ).images

        # Adjusts colors
        images: List[torch.Tensor] = list()
        for image in generated_images:
            images.append(
                utils.adjust_image_colors(
                    utils.to_tensor(image)
                )
            )

        # Returns
        return {"image": torch.stack(images, dim=0)}


class GStablePipeline(PipelineV2):
    """
    Represents a GStablePipeline.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _loss : Loss
            training's loss function
        components : ComponentsV2
            training's components

    Methods
    ----------
        _init_components
            Initializes the pipeline's components
        learn
            Learns on a batch of data
        _forward
            Extracts noise within the noisy image using the noise_scheduler
        _encode_image
            Reduces tensor's dimension using a VAE
        _encode_text
            Encodes a text into a tensor using a CLIP
        _add_noise
            Adds noise to a given tensor
        inference
            Generates and image using the training's noise_scheduler
        _build
            Builds the hugging face pipeline using the pipeline's components
    """

    def __init__(
            self,
            params: Dict[str, Any],
            num_epochs: int,
            num_batches: int
    ):
        """
        Instantiates a GStablePipeline.

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
        super(GStablePipeline, self).__init__(params, num_epochs, num_batches)

    def _forward(
            self,
            batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts noise within the noisy image using the noise_scheduler.

        Parameters
        ----------
            batch : Dict[str, torch.Tensor]
                batch of data

        Returns
        ----------
            torch.Tensor
                added noise
            torch.Tensor
                extracted noise
        """
        # Image
        batch["image"] = batch["image"].type(torch.float32).to(self._DEVICE)

        # Prompt
        batch["guider"] = self._encode_text(
            batch["guider"].type(torch.int32).to(self._DEVICE)
        )

        # Predicts added noise
        noisy_image, noise, timestep = self._add_noise(batch["image"])
        return noise, self.components.model(noisy_image, timestep, batch["guider"]).sample

    def inference(
            self,
    ) -> Dict[str, torch.Tensor]:
        """
        Generates and image using the training's components.

        Returns
        ----------
            Dict[str, torch.Tensor]
                generated image
        """
        pipeline = self()
        prompts: List[str] = [
            "a blue bird with horns", "a cartoon red turtle with fire",
            "a green monkey with a sword", "a big red lion with a smile"
        ]

        # Generates images
        generated_images = pipeline(
            prompts,
            num_inference_steps=50,
            generator=torch.manual_seed(0)
        ).images

        # Adjusts colors
        images: List[torch.Tensor] = list()
        for image in generated_images:
            images.append(
                utils.adjust_image_colors(
                    utils.to_tensor(image)
                )
            )

        # Returns
        return {
            prompt: images[idx].unsqueeze(0)
            for idx, prompt
            in enumerate(prompts)
        }
