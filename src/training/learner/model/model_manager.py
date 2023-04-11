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
from .models import UNet, GuidedUNet


class ModelManager(dict):
    """
    Represents a model manager.

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
            "unet": UNet,
            "guided unet": GuidedUNet,
        })

        # Attributes
        self._params: Dict[str, Any] = params

    def __call__(self, model_id: str) -> torch.nn.Module:
        """
        Parameters
        ----------
            model_id : str
                id of the model

        Returns
        ----------
            torch.nn.Module
                model associated to the model id
        """
        try:
            return self[model_id](self._params)
        except KeyError:
            raise KeyError(f"The {model_id} isn't handled by the model manager.")
