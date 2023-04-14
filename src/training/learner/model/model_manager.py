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
            "unet": {
                "class": UNet,
                "params": {
                    "img_size": params["img_size"],
                    "in_channels": params["num_channels"],
                    "out_channels": params["num_channels"],
                    "block_out_channels": params["block_out_channels"]
                }
            },
            "guided unet": {
                "class": GuidedUNet,
                "params": {
                    "img_size": params["img_size"],
                    "in_channels": params["num_channels"],
                    "out_channels": params["num_channels"],
                    "block_out_channels": params["block_out_channels"],
                    # "class_embed_type": "timestep"
                    "num_class_embeds": params["num_classes"]
                }
            }
        })

        # Attributes
        self._params: Dict[str, Any] = params

    def __call__(
            self,
            model_id: str
    ) -> torch.nn.Module:
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
            return self[model_id]["class"](**self[model_id]["params"])
        except KeyError:
            raise KeyError(f"The {model_id} isn't handled by the model manager.")
