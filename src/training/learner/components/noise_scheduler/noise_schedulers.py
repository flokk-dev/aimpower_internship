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
        params: Dict[str, Any]
) -> DDPMScheduler:
    """
    Initializes a noise scheduler.

    Parameters
    ----------
        params : Dict[str, Any]
            parameters needed to adjust the program behaviour

    Returns
    ----------
        DDPMScheduler
            training's noise scheduler
    """
    return DDPMScheduler(num_train_timesteps=params["num_timesteps"])


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
        params: Dict[str, Any]
) -> DDIMScheduler:
    """
    Initializes a noise scheduler.

    Parameters
    ----------
        params : Dict[str, Any]
            parameters needed to adjust the program behaviour

    Returns
    ----------
        DDIMScheduler
            training's noise scheduler
    """
    noise_scheduler = DDIMScheduler(num_train_timesteps=params["num_timesteps"])
    noise_scheduler.set_timesteps(params["num_timesteps"])

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
