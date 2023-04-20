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
from .noise_schedulers import init_ddpm, load_ddpm, init_ddim, load_ddim


class NoiseSchedulerManager(dict):
    """ Represents a NoiseSchedulerManager. """

    def __init__(
            self,
    ):
        """ Instantiates a NoiseSchedulerManager. """
        # Mother class
        super(NoiseSchedulerManager, self).__init__({
            "ddpm": {"init": init_ddpm, "load": load_ddpm},
            "ddim": {"load": load_ddim, "init": init_ddim}
        })

    def __call__(
            self,
            scheduler_type: str,
            scheduler_params: Dict[str, Any],
            pipeline_path: str,
    ) -> diffusers.SchedulerMixin:
        """
        Parameters
        ----------
            scheduler_type : str
                id of the noise scheduler
            scheduler_params : Dict[str, Any]
                noise scheduler's parameters
            pipeline_path : str
                path to the pretrained pipeline

        Returns
        ----------
            diffusers.SchedulerMixin
                noise scheduler associated to the noise scheduler type
        """
        try:
            if pipeline_path:
                return self[scheduler_type]["load"](pipeline_path)
            return self[scheduler_type]["init"](**scheduler_params)

        except KeyError:
            raise KeyError(f"The {scheduler_type} isn't handled by the noise scheduler manager.")
