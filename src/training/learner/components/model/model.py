"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import torch

# IMPORT: project
from .models import init_unet, load_unet, init_conditioned_unet, load_conditioned_unet


class ModelManager(dict):
    """ Represents a ModelManager. """

    def __init__(
            self,
    ):
        """ Instantiates a ModelManager. """
        # Mother class
        super(ModelManager, self).__init__({
            "unet": {"init": init_unet, "load": load_unet},
            "conditioned unet": {"init": init_conditioned_unet, "load": load_conditioned_unet}
        })

    def __call__(
            self,
            model_type: str,
            model_params: Dict[str, Any],
            pipeline_path: str
    ) -> torch.nn.Module:
        """
        Parameters
        ----------
            model_type : str
                type of the model
            model_params : Dict[str, Any]
                model's parameters
            pipeline_path : str
                path to the pretrained pipeline

        Returns
        ----------
            torch.nn.Module
                model associated to the model type
        """
        try:
            if pipeline_path:
                return self[model_type]["load"](pipeline_path)
            return self[model_type]["init"](model_params)

        except KeyError:
            raise KeyError(f"The {model_type} isn't handled by the model manager.")
