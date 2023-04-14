"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: deep learning
from diffusers import UNet2DConditionModel


class GuidedUNet(UNet2DConditionModel):
    """ Represents a guided U-Net model. """

    def __init__(
            self,
            img_size: int,
            in_channels: int,
            out_channels: int,
            block_out_channels: Tuple[int],
            num_class_embeds: int = 10,
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
            num_class_embeds : int
                number of classes needed to setup the embedding
        """
        # Mother class
        super(GuidedUNet, self).__init__(
            sample_size=img_size,
            in_channels=in_channels, out_channels=out_channels,
            layers_per_block=2, block_out_channels=block_out_channels,
            num_class_embeds=num_class_embeds
        )

    def __str__(self) -> str:
        """
        Returns
        ----------
            str
                the model's name
        """
        return "guided U-Net"
