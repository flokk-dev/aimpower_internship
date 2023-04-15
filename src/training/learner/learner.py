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
    Represents a Learner, which will be modified depending on the use case.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _loss : Loss
            training's loss function
        _components : Components
            training's components

    Methods
    ----------
        _learn
            Learns on a batch of data
        _forward
            Extracts noise within the noisy image using the noise_scheduler
        _add_noise
            Adds noise to a given tensor
        inference
            Generates and image using the training's noise_scheduler
    """
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(
            self,
            params: Dict[str, Any],
    ):
        """
        Instantiates a Learner.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
        """
        # Attributes
        self._params: Dict[str, Any] = params

        self._loss: torch.nn.Module = torch.nn.MSELoss().to(self._DEVICE)
        self._components = None

    @property
    def components(self) -> Components:
        """
        Returns the training's components.

        Returns
        ----------
            Components
                training's components
        """
        return self._components

    def _learn(
            self,
            batch: Union[torch.Tensor, Tuple[torch.Tensor, str]],
    ) -> float:
        """
        Learns on a batch of data.

        Parameters
        ----------
            batch : Union[torch.Tensor, Tuple[torch.Tensor, str]]
                batch of data

        Returns
        ----------
            float
                loss value computed using batch's data
        """
        # Forward batch to the noise_scheduler
        noise, noise_pred = self._forward(batch)

        # Loss backward
        loss_value: torch.Tensor = self._loss(noise_pred, noise)
        loss_value.backward()

        # Update the training components
        self._components.optimizer.step()
        self._components.lr_scheduler.step()
        self._components.optimizer.zero_grad()

        # Returns
        return loss_value.detach().item()

    def _forward(
            self,
            batch: Union[torch.Tensor, Tuple[torch.Tensor, str]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts noise within the noisy image using the noise_scheduler.

        Parameters
        ----------
            batch : Union[torch.Tensor, Tuple[torch.Tensor, str]]
                batch of data

        Returns
        ----------
            torch.Tensor
                added noise
            torch.Tensor
                extracted noise

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()

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
        noise: torch.Tensor = torch.randn(tensor.shape).to(self._DEVICE)

        # Sample random timestep
        timestep: torch.Tensor = torch.randint(
            0, self._components.scheduler.config.num_train_timesteps, (noise.shape[0],)
        ).to(self._DEVICE)

        # Add noise to the input data
        noisy_input: torch.Tensor = self._components.scheduler.add_noise(
            tensor, noise, timestep
        ).to(self._DEVICE)

        return noisy_input, noise, timestep

    def inference(
            self,
            to_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generates and image using the training's noise_scheduler.

        Parameters
        ----------
            to_dict : bool
                wether or not to return a dictionary

        Returns
        ----------
            torch.Tensor
                generated image
        """
        raise NotImplementedError()

    def __call__(
            self,
            batch: Union[torch.Tensor, Tuple[torch.Tensor, str]],
    ) -> float:
        """
        Parameters
        ----------
            batch : Union[torch.Tensor, Tuple[torch.Tensor, str]]
                batch of data

        Returns
        ----------
            float
                loss value computed using batch's data
        """
        return self._learn(batch)
