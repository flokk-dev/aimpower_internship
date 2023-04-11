"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import torch

# IMPORT: project
from .models import UNet, GuidedUNet


class ModelManager(dict):
    """ Represents a model manager. """

    def __init__(self):
        """ Instantiates a ModelManager. """
        super(ModelManager, self).__init__({
            "unet": UNet,
            "guided unet": GuidedUNet,
        })

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
            return self[model_id]()
        except KeyError:
            raise KeyError(f"The {model_id} isn't handled by the model manager.")
