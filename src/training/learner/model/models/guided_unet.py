"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: deep learning
import torch
from diffusers.models.unet_2d import UNet2DOutput

# IMPORT: project
from .unet import UNet


class GuidedUNet(UNet):
    """ Represents a guided U-Net model. """

    def __init__(
            self,
            params: Dict[str, Any]
    ):
        """
        Instantiates a GuidedUNet.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
        """
        # Mother class
        super(GuidedUNet, self).__init__(params)

        # Attributes
        self._class_emb = torch.nn.Embedding(params["num_classes"], 4)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DOutput, Tuple]:
        """
        Parameters
        ----------
            sample : torch.Tensor
                noisy inputs tensor
            timestep : Union[torch.Tensor, float, int]
                timesteps
            class_labels : Optional[torch.Tensor]
                optional class labels for conditioning
            return_dict : bool
                whether or not to return a dictionary

        Returns
        ----------
            torch.nn.Tensor
                noise prediction
        """
        # Stores the shape of the noisy input
        b, h, w, h = sample.shape

        # Generates the additional input channels
        cond_channels: torch.Tensor = self._class_emb(class_labels)
        cond_channels: torch.Tensor = cond_channels.view(
            b, cond_channels.shape[1], 1, 1
        ).expand(b, cond_channels.shape[1], w, h)

        # Concatenates noisy input and conditional channels
        cond_input: torch.Tensor = torch.cat((sample, cond_channels), 1)

        # Forwards it through the U-Net
        return super().forward(cond_input, timestep, return_dict=return_dict)

    def __str__(self) -> str:
        """
        Returns
        ----------
            str
                the model's name
        """
        return "guided U-Net"
