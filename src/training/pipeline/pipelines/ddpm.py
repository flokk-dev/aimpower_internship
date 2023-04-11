"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import torch

# IMPORT: deep learning
from diffusers import DDPMPipeline, DDPMScheduler


def load_ddpm(
        weights_path: str
) -> DDPMPipeline:
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
    return DDPMPipeline.from_pretrained(weights_path)


def init_ddpm(
        model: torch.nn.Module
) -> DDPMPipeline:
    """
    Initializes a pipeline.

    Parameters
    ----------
        model : torch.nn.Module
            path to the pipeline's weights

    Returns
    ----------
        DDPMPipeline
            training's pipeline
    """
    return DDPMPipeline(
        unet=model,
        scheduler=DDPMScheduler(num_train_timesteps=1000),
    )
