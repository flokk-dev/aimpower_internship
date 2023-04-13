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
from src.training.learner.pipeline import PipelineManager


class Learner:
    """
    Represents a Learner, which will be modified depending on the use case.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        loss : Loss
            training's loss function
        pipeline : diffusers.DiffusionPipeline
            diffusion pipeline
        optimizer : torch.optim.Optimizer
            pipeline's optimizer
        lr_scheduler : torch.nn.Module
            optimizer's scheduler

    Methods
    ----------
        _learn
            Learns on a batch of data
        _forward
            Extracts noise within the noisy image using the pipeline
        _add_noise
            Adds noise to a given tensor
        inference
            Generates and image using the training's pipeline
    """
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(
            self,
            params: Dict[str, Any],
            num_batches: int,
            weights_path: str
    ):
        """
        Instantiates a Learner.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            num_batches : int
                number of batches within the data loader
            weights_path : str
                path to the pipeline's weights
        """
        # Attributes
        self._params: Dict[str, Any] = params

        # Loss
        self.loss: torch.nn.Module = torch.nn.MSELoss().to(self._DEVICE)

        # Pipeline
        self.pipeline: diffusers.DiffusionPipeline = PipelineManager(params)(
            params["pipeline_id"], weights_path, params["model_id"]
        ).to(self._DEVICE)

        # Optimizer and learning rate
        self.optimizer: diffusers.optimization.Optimizer = torch.optim.AdamW(
            self.pipeline.unet.parameters(), lr=self._params["lr"]
        )
        self.lr_scheduler: torch.nn.Module = diffusers.optimization.get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=params["lr_warmup_steps"],
            num_training_steps=(num_batches * params["num_epochs"]),
        )

    def _learn(
            self,
            batch: Union[torch.Tensor, Tuple[torch.Tensor, str]],
            batch_idx: int
    ) -> float:
        """
        Learns on a batch of data.

        Parameters
        ----------
            batch : Union[torch.Tensor, Tuple[torch.Tensor, str]]
                batch of data
            batch_idx : int
                batch's index

        Returns
        ----------
            float
                loss value computed using batch's data
        """
        # Forward batch to the pipeline
        noise, noise_pred = self._forward(batch)

        # Loss backward
        loss_value: torch.Tensor = self.loss(noise_pred, noise)
        loss_value.backward()

        # Update the training components
        if batch_idx % 1 == 0:
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        # Returns
        return loss_value.detach().item()

    def _forward(
            self,
            batch: Union[torch.Tensor, Tuple[torch.Tensor, str]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts noise within the noisy image using the pipeline.

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

        # Sample random timesteps
        timestep: torch.Tensor = torch.randint(
            0, self.pipeline.scheduler.config.num_train_timesteps, (noise.shape[0],)
        ).to(self._DEVICE)

        # Add noise to the input data
        noisy_input: torch.Tensor = self.pipeline.scheduler.add_noise(
            tensor, noise, timestep
        ).to(self._DEVICE)

        return noisy_input, noise, timestep

    def inference(
            self,
            to_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generates and image using the training's pipeline.

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
            batch_idx: int
    ) -> float:
        """
        Parameters
        ----------
            batch : Union[torch.Tensor, Tuple[torch.Tensor, str]]
                batch of data
            batch_idx : int
                batch's index

        Returns
        ----------
            float
                loss value computed using batch's data
        """
        return self._learn(batch, batch_idx)
