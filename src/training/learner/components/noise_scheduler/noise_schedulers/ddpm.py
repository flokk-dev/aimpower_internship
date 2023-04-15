"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import torch

# IMPORT: deep learning
from diffusers import DDPMScheduler


def load_ddpm(
        weights_path: str
) -> DDPMScheduler:
    """
    Loads a pretrained noise scheduler.

    Parameters
    ----------
        weights_path : str
            path to the noise scheduler's weights

    Returns
    ----------
        DDPMScheduler
            training's noise scheduler
    """
    return DDPMScheduler.from_pretrained(weights_path)


def init_ddpm() -> DDPMScheduler:
    """
    Initializes a noise scheduler.

    Returns
    ----------
        DDPMScheduler
            training's noise scheduler
    """
    return DDPMScheduler(num_train_timesteps=1000)
