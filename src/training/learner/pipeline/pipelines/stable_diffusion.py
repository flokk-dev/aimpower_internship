"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import torch

# IMPORT: deep learning
from diffusers import StableDiffusionPipeline


def load_stable_diffusion(
        weights_path: str
) -> StableDiffusionPipeline:
    """
    Loads a pretrained pipeline.

    Parameters
    ----------
        weights_path : str
            path to the pipeline's weights

    Returns
    ----------
        DDPMPipeline
            training's pipeline
    """
    return StableDiffusionPipeline.from_pretrained(weights_path)


def init_stable_diffusion() -> StableDiffusionPipeline:
    """
    Initializes a pipeline.

    Returns
    ----------
        DDPMPipeline
            training's pipeline
    """
    return StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
