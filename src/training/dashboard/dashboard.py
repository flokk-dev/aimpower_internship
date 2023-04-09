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

# IMPORT: project
import utils


class Dashboard:
    """
    Represents a dashboard.

    Attributes
    ----------
        _loss : Dict[str, List[float]]
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

    def __init__(self, params: Dict[str, Any], train_id: str, mode="online"):
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
        self._loss: Dict[str, List[float]] = {"train": list(), "valid": list()}
        self._tensor_to_pil = transforms.ToPILImage()

    @staticmethod
    def collect_info(model: torch.nn.Module):
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

    def update_loss(self, loss: List[float], step: str):
        """
        Updates the history of loss.

        Parameters
        ----------
            loss : List[float]
                loss values during an epoch
            step : str
                training step
        """
        # Update the loss
        current_loss = sum(loss) / len(loss)
        self._loss[step].append(current_loss)

    def upload_values(self, lr):
        """
        Uploads the history of learning rate, loss and metrics.

        Parameters
        ----------
            lr : float
                learning rate value during the epoch
        """
        result = dict()

        # LOSSES
        result["train loss"] = self._loss["train"][-1]
        result["valid loss"] = self._loss["valid"][-1]

        # LEARNING RATE
        result["lr"] = lr

        # LOG ON WANDB
        wandb.log(result)

    def upload_images(
            self,
            input_batch: torch.Tensor, prediction_batch: torch.Tensor, target_batch: torch.Tensor,
            step: str
    ):
        """
        Uploads examples of results.

        Parameters
        ----------
            input_batch : torch.Tensor
                batch of input tensors
            prediction_batch : torch.Tensor
                batch of predicted tensors
            target_batch : torch.Tensor
                batch of target tensors
            step : str
                training step

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        images = {
            f"input_{step}": utils.adjust_image_colors(input_batch[0]),
            f"target_{step}": utils.adjust_image_colors(target_batch[0]),
            f"prediction_{step}": utils.adjust_image_colors(prediction_batch[0]),
        }

        for image_id in images.keys():
            images[image_id] = [wandb.Image(self._tensor_to_pil(images[image_id]))]

        wandb.log(images)

    @staticmethod
    def upload_inference(images: torch.Tensor):
        """
        Uploads examples of results.

        Parameters
        ----------
            images : torch.Tensor
                generated images
        """
        wandb.log({"inference": images})
