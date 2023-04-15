"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import torch

# IMPORT: deep learning
from diffusers import DDIMScheduler


def load_ddim(
        weights_path: str
) -> DDIMScheduler:
    """
    Loads a pretrained noise scheduler.

    Parameters
    ----------
        weights_path : str
            path to the noise scheduler's weights

    Returns
    ----------
        DDIMScheduler
            training's noise scheduler
    """
    return DDIMScheduler.from_pretrained(weights_path)


def init_ddim() -> DDIMScheduler:
    """
    Initializes a noise scheduler.

    Returns
    ----------
        DDIMScheduler
            training's noise scheduler
    """
    noise_scheduler = DDIMScheduler(num_train_timesteps=50)
    noise_scheduler.set_timesteps(50)

    return noise_scheduler
