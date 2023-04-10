"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import os

from PIL import Image
import torch
from diffusers import DDPMPipeline, DDIMPipeline

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
        print(f"update_loss: {loss}, {sum(loss)}, {len(loss)}")

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
            image: torch.Tensor,
            input_tensor: torch.Tensor,
            target_tensor: torch.Tensor,
            pred_tensor: torch.Tensor,
            step: str
    ):
        """
        Uploads examples of results.

        Parameters
        ----------
            image : torch.Tensor
                input tensor without noise
            input_tensor : torch.Tensor
                input tensor with noise
            target_tensor : torch.Tensor
                target tensor
            pred_tensor : torch.Tensor
                predicted tensor
            step : str
                training step

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        values = torch.unique(image)
        print(f"dashboard image + {min(values)} + {max(values)} + {image.shape}")
        values = torch.unique(input_tensor)
        print(f"dashboard noisy + {min(values)} + {max(values)} + {input_tensor.shape}")
        values = torch.unique(target_tensor)
        print(f"dashboard target + {min(values)} + {max(values)} + {target_tensor.shape}")
        values = torch.unique(pred_tensor)
        print(f"dashboard pred + {min(values)} + {max(values)} + {pred_tensor.shape}")

        images = {
            f"image_{step}": utils.adjust_image_colors(image),
            f"input_{step}": utils.adjust_image_colors(input_tensor),
            f"target_{step}": utils.adjust_image_colors(target_tensor),
            f"pred_{step}": utils.adjust_image_colors(pred_tensor),
        }

        values = torch.unique(images["image_train"])
        print(f"dashboard image + {min(values)} + {max(values)} + {images['image_train'].shape}")
        values = torch.unique(images["input_train"])
        print(f"dashboard noisy + {min(values)} + {max(values)} + {images['input_train'].shape}")
        values = torch.unique(images["target_train"])
        print(f"dashboard target + {min(values)} + {max(values)} + {images['target_train'].shape}")
        values = torch.unique(images["pred_train"])
        print(f"dashboard pred + {min(values)} + {max(values)} + {images['pred_train'].shape}")

        for image_id in images.keys():
            images[image_id] = [wandb.Image(self._tensor_to_pil(images[image_id]))]

        wandb.log(images)

    @staticmethod
    def upload_inference(pipeline: Union[DDPMPipeline, DDIMPipeline]):
        """
        Uploads examples of results.

        Parameters
        ----------
            pipeline : Union[DDPMPipeline, DDIMPipeline]
                trained diffusion pipeline
        """
        images: List[Image] = pipeline(
            batch_size=5, generator=torch.manual_seed(0)
        ).images

        image = transforms.PILToTensor()(images[0])
        values = torch.unique(image)
        print(f"dashboard adjusted + {min(values)} + {max(values)} + {image.shape}")

        wandb.log({
            "inference": [wandb.Image(image) for image in images]
        })
