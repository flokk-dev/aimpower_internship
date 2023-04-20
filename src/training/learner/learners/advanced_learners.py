"""
Creator: Flokk
Date: 09/04/2023
Version: 1.0
Purpose:
"""

# IMPORT: utils
from typing import *
from tqdm import tqdm

# IMPORT: deep learning
import torch
import diffusers

# IMPORT: project
import utils

from src.training.learner.advanced_learner import AdvancedLearner
from src.training.learner.components import AdvancedComponents


class StableLearner(AdvancedLearner):
    """
    Represents a StableLearner.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _loss : Loss
            training's loss function
        components : AdvancedComponents
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
            num_epochs: int,
            num_batches: int
    ):
        """
        Instantiates a StableLearner.

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
        super(StableLearner, self).__init__(params)

        # Components
        self.components = AdvancedComponents(params["components"], num_epochs, num_batches)

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
        batch["prompt"] = self._encode_text(
            batch["prompt"].type(torch.float32).to(self._DEVICE)
        )

        # Predicts added noise
        noisy_image, noise, timestep = self._add_noise(batch["image"])
        return noise, self.components.model(noisy_image, timestep, batch["prompt"]).sample

    def inference(
            self,
    ) -> Dict[str, torch.Tensor]:
        """
        Generates and image using the training's pipeline.

        Returns
        ----------
            Dict[str, torch.Tensor]
                generated image
        """
        # Pipeline
        pipeline = diffusers.DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=self._params["components"]["model"]["pipeline_path"],
            unet=self.components.model
        ).to(self._DEVICE)

        pipeline.set_progress_bar_config(disable=True)

        # Prompts
        prompts = [
            "a blue bird with horns", "a cartoon red turtle with fire",
            "a green monkey with a sword", "a big red lion with a smile"
        ]

        # Validation
        images = list()
        for prompt in prompts:
            images.append(
                utils.to_tensor(
                    pipeline(
                        prompt,
                        num_inference_steps=30,
                        generator=torch.manual_seed(0)
                    ).images[0]
                )
            )

        print(images[0].shape)
        return {"image": torch.stack(images, dim=0)}
