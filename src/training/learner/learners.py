"""
Creator: Flokk
Date: 09/04/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: deep learning
import torch

# IMPORT: project
from .learner import Learner


class DiffusionLearner(Learner):
    """
    Represents a DiffusionLearner.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _loss : Loss
            training's loss function
        components : ComponentsV1
            training's components

    Methods
    ----------
        learn
            Learns on a batch of data
        _forward
            Extracts noise within the noisy image using the noise_scheduler
        _add_noise
            Adds noise to a given tensor
    """

    def __init__(
            self,
            params: Dict[str, Any],
            dataset_path: str,
            num_epochs: int
    ):
        """
        Instantiates a DiffusionLearner.

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
        super(DiffusionLearner, self).__init__(params, dataset_path, num_epochs)

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
        # Adds noise
        noisy_image, noise, timestep = self._add_noise(batch["image"])

        # Predicts added noise
        return noise, self.components.model(noisy_image, timestep).sample


class GuidedDiffusionLearner(Learner):
    """
    Represents a GuidedDiffusionLearner.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _loss : Loss
            training's loss function
        components : ComponentsV1
            training's components

    Methods
    ----------
        learn
            Learns on a batch of data
        _forward
            Extracts noise within the noisy image using the noise_scheduler
        _add_noise
            Adds noise to a given tensor
    """

    def __init__(
            self,
            params: Dict[str, Any],
            dataset_path: str,
            num_epochs: int
    ):
        """
        Instantiates a GuidedDiffusionLearner.

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
        super(GuidedDiffusionLearner, self).__init__(params, dataset_path, num_epochs)

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
        # Adds noise
        noisy_image, noise, timestep = self._add_noise(batch["image"])

        # Predicts added noise
        return noise, self.components.model(
            noisy_image, timestep, batch["label"].type(torch.int32)
        ).sample


class StableDiffusionLearner(Learner):
    """
    Represents a StableDiffusionLearner.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _loss : Loss
            training's loss function
        components : ComponentsV1
            training's components

    Methods
    ----------
        learn
            Learns on a batch of data
        _forward
            Extracts noise within the noisy image using the noise_scheduler
        _add_noise
            Adds noise to a given tensor
    """

    def __init__(
            self,
            params: Dict[str, Any],
            dataset_path: str,
            num_epochs: int
    ):
        """
        Instantiates a GuidedDiffusionLearner.

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
        super(StableDiffusionLearner, self).__init__(params, dataset_path, num_epochs)

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
        print(f"image: {batch['image'].dtype}, {batch['image'].shape}")
        batch["image"] = (
                self.components.vae.encode(batch["image"].to("cuda")).latent_dist.sample() *
                self.components.vae.config.scaling_factor
        )
        print(f"vaed image: {batch['image'].dtype}, {batch['image'].shape}")

        return super().learn(batch)

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
        # Adds noise
        noisy_image, noise, timestep = self._add_noise(batch["image"])

        # Encode prompt
        print(f"prompt: {batch['prompt'].dtype}, {batch['prompt'].shape}")
        batch["prompt"] = self.components.text_encoder(
            batch["prompt"].to("cuda")
        )[0].type(torch.float16)
        print(f"encoded prompt: {batch['prompt'].dtype}, {batch['prompt'].shape}")

        # Predicts added noise
        return noise, self.components.model(
            noisy_image, timestep, batch["prompt"]
        ).sample
