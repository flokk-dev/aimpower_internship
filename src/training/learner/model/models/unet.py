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
    """
    Represents a U-Net model.

    Attributes
    ----------
        _name : str
            the model's name
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Instantiates a UNet.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
        """
        # Mother class
        """
        super(UNet, self).__init__(
            sample_size=params["img_size"], in_channels=1, out_channels=1,
            layers_per_block=2, block_out_channels=(128, 128, 256, 256, 512, 512),

            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D"
            )
        )
        """

        super(UNet, self).__init__(
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

        # Attributes
        self._name: str = "U-Net"

    def __str__(self) -> str:
        """
        Returns
        ----------
            str
                the model's name
        """
        return self._name
