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
from .components import Components


class Learner:
    """
    Represents a Learner.

    Attributes
    ----------
        _config : Dict[str, Any]
            configuration needed to adjust the program behaviour
        components : Components
            training's components

    Methods
    ----------
        learn
            Learns on a batch of data
        _forward
            Extracts noise within the noisy image using the model
        _add_noise
            Adds noise to a given tensor
    """

    def __init__(
            self,
            config: Dict[str, Any]
    ):
        """
        Instantiates a Learner.

        Parameters
        ----------
            config : Dict[str, Any]
                configuration needed to adjust the program behaviour
        """
        # ----- Attributes ----- #
        self._config: Dict[str, Any] = config

        # Components
        self.components: Components = None

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

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()

    def _forward(
            self,
            batch: Dict[str, torch.Tensor],
    ):
        """
        Extracts noise within the noisy image using the model.

        Parameters
        ----------
            batch : Dict[str, torch.Tensor]
                batch of data
        """
        # Encode prompt
        batch["prompt"] = self.components.text_encoder(
            batch["prompt"]
        )[0]

        # Encode image
        batch["image"] = self.components.vae.encode(batch["image"]).latent_dist.sample() * \
            self.components.vae.config.scaling_factor

    def _add_noise(
            self,
            tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Adds noise to a given tensor.

        Parameters
        ----------
            tensor : torch.Tensor
                tensor to add noise to

        Returns
        ----------
            torch.Tensor
                noisy tensor
            torch.Tensor
                added noise
            torch.Tensor
                noise's timestep
        """
        # Sample random noise
        noise: torch.Tensor = torch.randn_like(tensor, device=tensor.device)

        # Sample random timestep
        timestep: torch.Tensor = torch.randint(
            low=0,
            high=self.components.noise_scheduler.config.num_train_timesteps,
            size=(noise.shape[0],),
            device=tensor.device
        )

        # Add noise to the input data
        noisy_input: torch.Tensor = self.components.noise_scheduler.add_noise(
            tensor, noise, timestep
        )

        return noisy_input, noise, timestep

    def __call__(
            self,
            batch: Dict[str, torch.Tensor],
    ) -> float:
        """
        Parameters
        ----------
            batch : Dict[str, torch.Tensor]
                batch of data

        Returns
        ----------
            float
                loss value computed using batch's data
        """
        return self._learn(batch)
