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
import diffusers

# IMPORT: project
from .components import ComponentsV1


class Pipeline:
    """
    Represents a Pipeline, which will be modified depending on the use case.

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
        _init_components
            Initializes the pipeline's components
        learn
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
            num_epochs: int,
            num_batches: int
    ):
        """
        Instantiates a Learner.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            num_epochs : int
                number of epochs during the training
            num_batches : int
                number of batches within the data loader
        """
        # Attributes
        self._params: Dict[str, Any] = params

        self._loss: torch.nn.Module = torch.nn.MSELoss().to(self._DEVICE)
        self.components: ComponentsV1 = self._init_components(params, num_epochs, num_batches)

    def _init_components(
            self,
            params: Dict[str, Any],
            num_epochs: int,
            num_batches: int
    ) -> ComponentsV1:
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
            ComponentsV1
                pipeline's components

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()

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
        noise, noise_pred = self._forward(batch)

        # Loss backward
        loss_value: torch.Tensor = self._loss(noise_pred, noise)
        loss_value.backward()

        # Update the training components
        self.components.optimizer.step()
        self.components.lr_scheduler.step()
        self.components.optimizer.zero_grad()

        # Returns
        return loss_value.detach().item()

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
            0, self.components.noise_scheduler.config.num_train_timesteps, (noise.shape[0],)
        ).to(self._DEVICE)

        # Add noise to the input data
        noisy_input: torch.Tensor = self.components.noise_scheduler.add_noise(
            tensor, noise, timestep
        ).to(self._DEVICE)

        return noisy_input, noise, timestep

    def inference(
            self,
    ) -> Dict[str, torch.Tensor]:
        """
        Generates images using the training's components.

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

    def __call__(self) -> diffusers.DiffusionPipeline:
        """
        Returns
        ----------
            diffusers.DiffusionPipeline
                diffusion pipeline

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()
