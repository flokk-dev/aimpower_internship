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
from .models import \
    load_unet, init_unet, \
    load_conditioned_unet, init_conditioned_unet


class ModelManager(dict):
    """
    Represents a ModelManager.

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
        Instantiates a ModelManager.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
        """
        # Mother class
        super(ModelManager, self).__init__({
            "unet": {
                "load": load_unet,
                "init": init_unet
            },
            "conditioned unet": {
                "load": load_conditioned_unet,
                "init": init_conditioned_unet
            }
        })

        # Attributes
        self._params: Dict[str, Any] = params

    def __call__(
            self,
            model_id: str,
            weights_path: str,
    ) -> torch.nn.Module:
        """
        Parameters
        ----------
            model_id : str
                id of the model
            weights_path : str
                path to the model's weights

        Returns
        ----------
            torch.nn.Module
                model associated to the model id
        """
        try:
            if weights_path is not None:
                return self[model_id]["load"](weights_path)
            return self[model_id]["load"](self._params)

        except KeyError:
            raise KeyError(f"The {model_id} isn't handled by the noise_scheduler manager.")
