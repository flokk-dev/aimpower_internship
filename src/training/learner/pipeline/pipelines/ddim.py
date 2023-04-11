"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import torch

# IMPORT: deep learning
from diffusers import DDIMPipeline, DDIMScheduler


def load_ddim(
        weights_path: str
) -> DDIMPipeline:
    """
    Loads a pretrained pipeline.

    Parameters
    ----------
        weights_path : str
            path to the pipeline's weights

    Returns
    ----------
        DDIMPipeline
            training's pipeline
    """
    return DDIMPipeline.from_pretrained(weights_path)


def init_ddim(
        model: torch.nn.Module
) -> DDIMPipeline:
    """
    Initializes a pipeline.

    Parameters
    ----------
        model : torch.nn.Module
            path to the pipeline's weights

    Returns
    ----------
        DDIMPipeline
            training's pipeline
    """
    return DDIMPipeline(
        unet=model,
        scheduler=DDIMScheduler(num_train_timesteps=50),
    )
