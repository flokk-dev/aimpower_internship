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
            img_size: int,
            in_channels: int,
            out_channels: int,
            num_classes: int = 10,
            emb_size: int = 4
    ):
        """
        Instantiates a GuidedUNet.

        Parameters
        ----------
            img_size : int
                the size of the input image
            in_channels : int
                the number of input channels
            out_channels : int
                the number of output channels
            num_classes : int
                the number of classes within the dataset
            emb_size : int
                size of the text embedding
        """
        # Mother class
        super(GuidedUNet, self).__init__(img_size, in_channels + emb_size, out_channels)

        # Attributes
        self._class_emb = torch.nn.Embedding(num_classes, emb_size)

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
                noise's timestep
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
        print(cond_input.shape)
        return super().forward(cond_input, timestep, return_dict=return_dict)

    def __str__(self) -> str:
        """
        Returns
        ----------
            str
                the model's name
        """
        return "guided U-Net"
