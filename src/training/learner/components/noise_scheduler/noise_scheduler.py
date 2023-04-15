"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import diffusers

# IMPORT: project
from .noise_schedulers import \
    load_ddpm, init_ddpm, \
    load_ddim, init_ddim


class NoiseSchedulerManager(dict):
    """
    Represents a NoiseSchedulerManager.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
    """

    def __init__(
            self,
            params: Dict[str, Any]
    ):
        """
        Instantiates a NoiseSchedulerManager.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
        """
        # Mother class
        super(NoiseSchedulerManager, self).__init__({
            "ddpm": {
                "load": load_ddpm,
                "init": init_ddpm
            },
            "ddim": {
                "load": load_ddim,
                "init": init_ddim
            }
        })

        # Attributes
        self._params: Dict[str, Any] = params

    def __call__(
            self,
            noise_scheduler_id: str,
            weights_path: str,
    ) -> diffusers.SchedulerMixin:
        """
        Parameters
        ----------
            noise_scheduler_id : str
                id of the noise scheduler
            weights_path : str
                path to the noise scheduler's weights

        Returns
        ----------
            diffusers.SchedulerMixin
                noise_scheduler associated to the noise_scheduler id
        """
        try:
            if weights_path is not None:
                return self[noise_scheduler_id]["load"](weights_path)
            return self[noise_scheduler_id]["load"](self._params)

        except KeyError:
            raise KeyError(f"The {noise_scheduler_id} isn't handled by the noise_scheduler manager.")
