"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

import os
import torch

# IMPORT: data processing
from torchvision import transforms

# IMPORT: visualization
import wandb


class Dashboard:
    """
    Represents a dashboard.

    Attributes
    ----------
        _loss : List[float]
            history of the loss value during training
        _tensor_to_pil: torchvision.transforms.Transform
            transform tensor to PIL image


    Methods
    ----------
        collect_info
            Specifies the model to follow
        shutdown
            Shutdowns the dashboard
        update_loss
            Updates the history of loss
        upload_values
            Uploads the history of learning rate and loss
        upload_images
            Uploads examples of results
    """

    def __init__(
            self,
            params: Dict[str, Any],
            train_id: str,
            mode="online"
    ):
        """
        Instantiates a Dashboard.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            train_id : str
                id of the training
        """
        # Initialize the wandb entity
        os.environ["WANDB_SILENT"] = "false"
        wandb.init(entity="flokk-dev", project="aimpower_internship_test", mode=mode)

        wandb.run.name = train_id
        wandb.config = params

        # Attributes
        self._loss: List[float] = list()
        self._tensor_to_pil = transforms.ToPILImage()

    @staticmethod
    def collect_info(
            model: torch.nn.Module
    ):
        """
        Specifies the model to follow.

        Parameters
        ----------
            model : torch.nn.Module
                model to follow.
        """
        wandb.watch(model)

    @staticmethod
    def shutdown():
        """ Shutdowns the dashboard. """
        wandb.finish()

    def update_loss(
            self,
            loss: List[float]
    ):
        """
        Updates the history of loss.

        Parameters
        ----------
            loss : List[float]
                loss values during an epoch
        """
        # Update the loss
        current_loss = sum(loss) / len(loss)
        self._loss.append(current_loss)

    def upload_values(
            self,
            lr: float
    ):
        """
        Uploads the history of learning rate, loss and metrics.

        Parameters
        ----------
            lr : float
                learning rate value during the epoch
        """
        result = dict()

        # LOSSES
        result["loss"] = self._loss[-1]

        # LEARNING RATE
        result["lr"] = lr

        # LOG ON WANDB
        wandb.log(result)

    def upload_inference(
            self,
            tensor: torch.Tensor
    ):
        """
        Uploads examples of results.

        Parameters
        ----------
            tensor : torch.Tensor
                tensor generated by the diffusion pipeline
        """
        """        
        wandb.log({
            "test": [
                wandb.Image(self._tensor_to_pil(tensor[batch_idx]))
                for batch_idx
                in range(tensor.shape[0])
            ]
        })"""
        wandb.log({
            "test": [
                wandb.Image(image)
                for image
                in tensor
            ]
        })
