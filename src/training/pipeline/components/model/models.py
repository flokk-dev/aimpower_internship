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


# ---------- U-Net ---------- #

def init_unet(
        sample_size: int,
        in_channels: int,
        out_channels: int,
        layers_per_block: int,
        block_out_channels: Tuple[int]
) -> UNet2DModel:
    """
    Initializes a model.

    Parameters
    ----------
        sample_size : int
            height and width of input/output sample
        in_channels : int
            number of channels in the input image
        out_channels : int
            number of channels in the output image
        layers_per_block : int
            number of layers per block
        block_out_channels : Tuple[int]
            blocks' output channels

    Returns
    ----------
        UNet2DModel
            training's model
    """
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,

        layers_per_block=layers_per_block,
        block_out_channels=block_out_channels
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


# ---------- Guided U-Net ---------- #

def init_guided_unet(
        sample_size: int,
        in_channels: int,
        out_channels: int,
        layers_per_block: int,
        block_out_channels: Tuple[int],
        num_class_embeds: int
) -> UNet2DModel:
    """
    Initializes a model.

    Parameters
    ----------
        sample_size : int
            height and width of input/output sample
        in_channels : int
            number of channels in the input image
        out_channels : int
            number of channels in the output image
        layers_per_block : int
            number of layers per block
        block_out_channels : Tuple[int]
            blocks' output channels
        num_class_embeds : int
            number of classes used to guid the model

    Returns
    ----------
        UNet2DModel
            training's model
    """
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,

        layers_per_block=layers_per_block,
        block_out_channels=block_out_channels,

        num_class_embeds=num_class_embeds
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


# ---------- Conditioned U-Net ---------- #

def init_conditioned_unet(
        sample_size: int,
        in_channels: int,
        out_channels: int,
        layers_per_block: int,
        block_out_channels: Tuple[int],
        cross_attention_dim: int = 768,
        attention_head_dim: int = 11
) -> UNet2DConditionModel:
    """
    Initializes a model.

    Parameters
    ----------
        sample_size : int
            height and width of input/output sample
        in_channels : int
            number of channels in the input image
        out_channels : int
            number of channels in the output image
        layers_per_block : int
            number of layers per block
        block_out_channels : Tuple[int]
            blocks' output channels
        cross_attention_dim : int
            dimension of the cross attention features
        attention_head_dim : int
            dimension of the attention heads

    Returns
    ----------
        UNet2DConditionModel
            training's model
    """
    return UNet2DConditionModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,

        layers_per_block=layers_per_block,
        block_out_channels=block_out_channels,

        cross_attention_dim=cross_attention_dim,
        attention_head_dim=attention_head_dim
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
