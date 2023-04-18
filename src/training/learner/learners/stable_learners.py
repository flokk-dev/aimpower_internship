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

# IMPORT: project
import utils

from src.training.learner.stable_learner import StableLearner
from src.training.learner.components import StableComponents


class BasicStableLearner(StableLearner):
    """
    Represents a BasicStableLearner.

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
        Instantiates a BasicStableLearner.

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
        super(BasicStableLearner, self).__init__(params)

        # Components
        self.components = StableComponents(params["components"], num_epochs, num_batches)

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
        print(type(self.components.model))

        # Image
        batch["image"]: torch.Tensor = batch["image"].type(torch.float32).to(self._DEVICE)
        if self._params["reduce_dimensions"]:
            batch["image"] = self._encode_image(batch["image"])

        # Prompt
        prompt = torch.randn(1, 4, 1280).type(torch.float32).to(self._DEVICE)

        # Predicts added noise
        noisy_image, noise, timestep = self._add_noise(batch["image"])
        return noise, self.components.model(noisy_image, timestep, prompt).sample

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
        pass


class ConditionedStableLearner(StableLearner):
    """
    Represents a ConditionedStableLearner.

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
        Instantiates a ConditionedStableLearner.

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
        super(ConditionedStableLearner, self).__init__(params)

        # Components
        self.components = StableComponents(params["components"], num_epochs, num_batches)

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
        if self._params["reduce_dimensions"]:
            batch["image"] = self._encode_image(batch["image"])

        # Prompt
        batch["prompt"] = batch["prompt"].type(torch.float32).to(self._DEVICE)
        batch["prompt"] = self._encode_text(batch["prompt"])

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
        pass
