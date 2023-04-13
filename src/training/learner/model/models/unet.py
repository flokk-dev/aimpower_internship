"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: deep learning
from diffusers import UNet2DModel


class UNet(UNet2DModel):
    """ Represents a U-Net model. """

    def __init__(
            self,
            img_size: int,
            in_channels: int,
            out_channels: int
    ):
        """
        Instantiates a UNet.

        Parameters
        ----------
            img_size : int
                the size of the input image
            in_channels : int
                the number of input channels
            out_channels : int
                the number of output channels
        """
        # Mother class
        super(UNet, self).__init__(
            sample_size=img_size,
            in_channels=in_channels, out_channels=out_channels,
            layers_per_block=2, block_out_channels=(128, 128, 256, 256),

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
