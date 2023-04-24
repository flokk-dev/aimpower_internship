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
from .components import DiffusionComponents, StableDiffusionComponents, LoRADiffusionComponents


class Learner:
    """
    Represents a Learner.

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
    _COMPONENTS = {
        "diffusion": DiffusionComponents,
        "stable diffusion": StableDiffusionComponents,
        "lora diffusion": LoRADiffusionComponents
    }

    def __init__(
            self,
            params: Dict[str, Any],
            dataset_path: str,
            num_epochs: int,
    ):
        """
        Instantiates a Learner.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            dataset_path : str
                path to the dataset
            num_epochs : int
                number of epochs during the training
        """
        # ----- Attributes ----- #
        self._params: Dict[str, Any] = params

        # Components
        self.components = self._COMPONENTS[params["components"]["type"]](
            params["components"], dataset_path, num_epochs
        )
        self.components.prepare()

        # Loss
        self._loss = torch.nn.MSELoss().to(self.components.accelerator.device)

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

        with self.components.accelerator.accumulate(self.components.model):
            # Loss backward
            loss_value: torch.Tensor = self._loss(noise_pred.float(), noise.float())
            print(f"loss: {loss_value.dtype}")
            self.components.accelerator.backward(loss_value)

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
        print(f"batch image: {tensor.dtype}")

        # Sample random noise
        noise: torch.Tensor = torch.randn_like(tensor, device=tensor.device)
        print(f"noise: {noise.dtype}")

        # Sample random timestep
        timestep: torch.Tensor = torch.randint(
            low=0,
            high=self.components.noise_scheduler.config.num_train_timesteps,
            size=(noise.shape[0],),
            device=tensor.device
        )
        print(f"timestep: {timestep.dtype}")

        # Add noise to the input data
        noisy_input: torch.Tensor = self.components.noise_scheduler.add_noise(
            tensor, noise, timestep
        )
        print(f"noisy_input: {noisy_input.dtype}")

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
        return self.learn(batch)
