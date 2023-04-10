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
from .trainer import Trainer


class LossTrainer(Trainer):
    """
    Represents a LossTrainer.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _data_loaders : Dict[str: DataLoader]
            training and validation data loaders
        _pipeline : Union[DDPMPipeline, DDIMPipeline]
            diffusion pipeline
        _optimizer : torch.optim.Optimizer
            pipeline's optimizer
        _scheduler : torch.nn.Module
            optimizer's scheduler
        _loss : Loss
            training's loss function

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

    def __init__(self, params: Dict[str, Any]):
        """
        Instantiates a LossTrainer.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
        """
        # Mother class
        super(LossTrainer, self).__init__(params)

    def _learn_on_batch(self, batch: torch.Tensor, batch_idx: int, learn: bool = True):
        """
        Learns using data within a batch.

        Parameters
        ----------
            batch : torch.Tensor
                batch of tensors
            batch_idx : int
                batch's index
            learn : bool
                boolean indicating whether to train

        Returns
        ----------
            torch.Float
                loss calculated using batch's data
            Dict[str, torch.Tensor]
                images generated using the batch
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
