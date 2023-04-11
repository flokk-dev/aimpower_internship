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
    Represents a Learner, that will be modified depending on the use case.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _loss : Loss
            training's loss function
        _pipeline : diffusers.DiffusionPipeline
            diffusion pipeline
        _optimizer : torch.optim.Optimizer
            pipeline's optimizer
        _scheduler : torch.nn.Module
            optimizer's scheduler

    Methods
    ----------
        _init_pipeline
            Initializes the training's pipeline
        _launch
            Launches the training
        _run_epoch
            Runs an epoch
        _learn_on_batch
            Learns using data within a batch
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
        self._loss: torch.nn.Module = torch.nn.MSELoss().to(self._DEVICE)

        # Pipeline
        self._pipeline: diffusers.DiffusionPipeline = PipelineManager(params)(
            params["pipeline_id"], weights_path, params["model_id"]
        )

        # Optimizer and learning rate
        self._optimizer: diffusers.optimization.Optimizer = torch.optim.AdamW(
            self._pipeline.unet.parameters(), lr=self._params["lr"]
        )
        self._scheduler: torch.nn.Module = diffusers.optimization.get_cosine_schedule_with_warmup(
            optimizer=self._optimizer,
            num_warmup_steps=params["lr_warmup_steps"],
            num_training_steps=(num_batches * params["num_epochs"]),
        )

    @property
    def loss(self) -> torch.nn.Module:
        """
        Returns the training's loss function.

        Returns
        ----------
            torch.nn.Module
                training's loss function
        """
        return self._loss

    @property
    def pipeline(self) -> diffusers.DiffusionPipeline:
        """
        Returns the training's pipeline.

        Returns
        ----------
            diffusers.DiffusionPipeline
                training's pipeline
        """
        return self._pipeline

    @property
    def optimizer(self) -> diffusers.optimization.Optimizer:
        """
        Returns the training's optimizer.

        Returns
        ----------
            diffusers.optimization.Optimizer
                training's optimizer
        """
        return self._optimizer

    @property
    def scheduler(self) -> torch.nn.Module:
        """
        Returns the training's scheduler.

        Returns
        ----------
            torch.nn.Module
                training's scheduler
        """
        return self._scheduler

    def _learn(
            self,
            batch: torch.Tensor,
            learn: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Learns using data within a batch.

        Parameters
        ----------
            batch : torch.Tensor
                batch of tensors
            learn : bool
                boolean indicating whether to train

        Returns
        ----------
            torch.Tensor
                loss calculated using batch's data
        """
        # Put input data on desired device
        input_tensor: torch.Tensor = batch.type(torch.float32).to(self._DEVICE)

        # Sample random noise
        noise: torch.Tensor = torch.randn(input_tensor.shape).to(self._DEVICE)

        # Sample random timesteps
        timesteps: torch.Tensor = torch.randint(
            0, self._pipeline.scheduler.num_train_timesteps, (noise.shape[0],)
        ).to(self._DEVICE)

        # Add noise to the input data
        noisy_input: torch.Tensor = self._pipeline.scheduler.add_noise(
            input_tensor, noise, timesteps
        ).to(self._DEVICE)

        # Get the model prediction --> noise
        # noise_pred = model(noisy_input, timesteps).sample
        noise_pred: torch.Tensor = self._pipeline.unet(noisy_input, timesteps, return_dict=False)[0]

        # Compare the prediction with the actual noise
        # NB - trying to predict noise (eps) not (noisy_ims-clean_ims) or just (clean_ims)
        loss_value: torch.Tensor = self._loss(noise_pred, noise)

        # Update the model parameters
        self._optimizer.zero_grad()
        with torch.set_grad_enabled(learn):
            loss_value.backward()
            self._optimizer.step()
            self._scheduler.step()

        # Return
        return loss_value.detach().item(), \
            {
                "image": input_tensor[0].cpu(),
                "input": noisy_input[0].cpu(),
                "target": noise[0].cpu(),
                "pred": noise_pred[0].cpu()
            }

    def __call__(
            self,
            batch: torch.Tensor,
            learn: bool = True
    ):
        """
        Parameters
        ----------
            batch : torch.Tensor
                batch of tensors
            learn : bool
                boolean indicating whether to train

        Returns
        ----------
            torch.Tensor
                loss calculated using batch's data
        """
        return self._learn(batch, learn)
