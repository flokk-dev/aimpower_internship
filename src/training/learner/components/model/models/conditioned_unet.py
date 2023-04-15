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


def load_conditioned_unet(
        weights_path: str
) -> UNet2DConditionModel:
    """
    Loads a pretrained model.

    Parameters
    ----------
        weights_path : str
            path to the model's weights

    Returns
    ----------
        UNet2DConditionModel
            training's model
    """
    return UNet2DConditionModel.from_pretrained(weights_path)


def init_conditioned_unet(
        params: Dict[str, Any]
) -> UNet2DConditionModel:
    """
    Initializes a model.

    Parameters
    ----------
        params : Dict[str, Any]
            parameters needed to adjust the program behaviour

    Returns
    ----------
        UNet2DConditionModel
            training's model
    """
    return UNet2DConditionModel(
        sample_size=params["img_size"],
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        layers_per_block=2,
        block_out_channels=params["block_out_channels"],
    )
