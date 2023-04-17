"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: deep learning
from diffusers import UNet2DModel, UNet2DConditionModel


def init_unet(
        params: Dict[str, Any]
) -> UNet2DModel:
    """
    Initializes a model.

    Parameters
    ----------
        params : Dict[str, Any]
            parameters needed to adjust the program behaviour

    Returns
    ----------
        UNet2DModel
            training's model
    """
    return UNet2DModel(
        sample_size=params["sample_size"],
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        layers_per_block=2,
        block_out_channels=params["block_out_channels"]
    )


def load_unet(
        pipeline_path: str
) -> UNet2DModel:
    """
    Loads a pretrained model.

    Parameters
    ----------
        pipeline_path : str
            path to the pipeline

    Returns
    ----------
        UNet2DModel
            training's model
    """
    return UNet2DModel.from_pretrained(pipeline_path, subfolder="unet")


def init_guided_unet(
        params: Dict[str, Any]
) -> UNet2DModel:
    """
    Initializes a model.

    Parameters
    ----------
        params : Dict[str, Any]
            parameters needed to adjust the program behaviour

    Returns
    ----------
        UNet2DModel
            training's model
    """
    return UNet2DModel(
        sample_size=params["sample_size"],
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        layers_per_block=2,
        block_out_channels=params["block_out_channels"],
        num_class_embeds=params["num_labels"]
    )


def load_guided_unet(
        pipeline_path: str
) -> UNet2DModel:
    """
    Loads a pretrained model.

    Parameters
    ----------
        pipeline_path : str
            path to the pipeline

    Returns
    ----------
        UNet2DModel
            training's model
    """
    return UNet2DModel.from_pretrained(pipeline_path, subfolder="unet")


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
        sample_size=params["sample_size"],
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        layers_per_block=2,
        block_out_channels=params["block_out_channels"],
    )


def load_conditioned_unet(
        pipeline_path: str
) -> UNet2DConditionModel:
    """
    Loads a pretrained model.

    Parameters
    ----------
        pipeline_path : str
            path to the pipeline

    Returns
    ----------
        UNet2DConditionModel
            training's model
    """
    return UNet2DConditionModel.from_pretrained(pipeline_path, subfolder="unet")
