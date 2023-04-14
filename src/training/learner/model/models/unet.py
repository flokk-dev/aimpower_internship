"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: deep learning
from diffusers import UNet2DModel


class UNet(UNet2DModel):
    """ Represents a U-Net model. """

    def __init__(
            self,
            img_size: int,
            in_channels: int,
            out_channels: int,
            block_out_channels: Tuple[int],
            class_embed_type: Optional[str] = None
    ):
        """
        Instantiates a UNet.

        Parameters
        ----------
            img_size : int
                size of the input image
            in_channels : int
                number of input channels
            out_channels : int
                number of output channels
            block_out_channels : int
                output size of the blocks.
        """
        # Mother class
        super(UNet, self).__init__(
            sample_size=img_size,
            in_channels=in_channels, out_channels=out_channels,
            layers_per_block=2, block_out_channels=block_out_channels,
            down_block_types=(
                "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"
            ),
            class_embed_type=class_embed_type,
        )

    def __str__(
            self
    ) -> str:
        """
        Returns
        ----------
            str
                the model's name
        """
        return "U-Net"
