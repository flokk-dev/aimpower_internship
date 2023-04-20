"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: deep learning
from diffusers import DDPMScheduler, DDIMScheduler


def init_ddpm(
        num_train_timesteps: int = 100
) -> DDPMScheduler:
    """
    Initializes a noise scheduler.

    Parameters
    ----------
        num_train_timesteps : int
            number of diffusion steps used to train the model

    Returns
    ----------
        DDPMScheduler
            training's noise scheduler
    """
    return DDPMScheduler(num_train_timesteps=num_train_timesteps)


def load_ddpm(
        pipeline_path: str
) -> DDPMScheduler:
    """
    Loads a pretrained noise scheduler.

    Parameters
    ----------
        pipeline_path : str
            path to the pipeline

    Returns
    ----------
        DDPMScheduler
            training's noise scheduler
    """
    return DDPMScheduler.from_pretrained(pipeline_path, subfolder="scheduler")


def init_ddim(
        num_train_timesteps: int = 50
) -> DDIMScheduler:
    """
    Initializes a noise scheduler.

    Parameters
    ----------
        num_train_timesteps : int
            number of diffusion steps used to train the model

    Returns
    ----------
        DDIMScheduler
            training's noise scheduler
    """
    noise_scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)
    noise_scheduler.set_timesteps(num_train_timesteps)

    return noise_scheduler


def load_ddim(
        pipeline_path: str
) -> DDIMScheduler:
    """
    Loads a pretrained noise scheduler.

    Parameters
    ----------
        pipeline_path : str
            path to the pipeline

    Returns
    ----------
        DDIMScheduler
            training's noise scheduler
    """
    return DDIMScheduler.from_pretrained(pipeline_path, subfolder="scheduler")
