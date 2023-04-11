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
from torch import nn

from diffusers import UNet2DModel


class GuidedUNet(nn.Module):
    """ Represents a guided U-Net model. """

    def __init__(self, params: Dict[str, Any]):
        """
        Instantiates a GuidedUNet.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
        """
        # Mother class
        super(GuidedUNet, self).__init__()

        # Attributes
        self._class_emb = nn.Embedding(10, 4)
        self._model = UNet2DModel(
            sample_size=params["img_size"], in_channels=1, out_channels=1,
            layers_per_block=2, block_out_channels=(64, 64, 128, 128),

            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D"
            ),
        )

        self.dtype = self._model.dtype

    def forward(
        self,
        noisy_input: torch.Tensor,
        timesteps: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
            noisy_input : torch.Tensor
                pass
            timesteps : Union[torch.Tensor, float, int]
                pass
            class_labels : Optional[torch.Tensor]
                pass

        Returns
        ----------
            torch.nn.Tensor
                noise prediction
        """
        # Stores the shape of the noisy input
        b, h, w, h = noisy_input.shape

        # Generates the additional input channels
        cond_channels = self.class_emb(class_labels)
        cond_channels = cond_channels.view(
            b, cond_channels.shape[1], 1, 1
        ).expand(b, cond_channels.shape[1], w, h)

        # Concatenates noisy input and conditional channels
        cond_input = torch.cat((noisy_input, cond_channels), 1)

        # Forwards it through the U-Net
        return self.model(cond_input, timesteps)

    def __str__(self) -> str:
        """
        Returns
        ----------
            str
                the model's name
        """
        return "guided U-Net"
