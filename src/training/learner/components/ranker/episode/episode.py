"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: deep learning
import torch


class Episode:
    """
    Represents an Episode.

    Attributes
    ----------
        _config : Dict[str, Any]
            configuration needed to adjust the program behaviour

    Methods
    ----------

    """
    def __init__(
            self,
            config: Dict[str, Any]
    ):
        """
        Instantiates a ClassicComponents.

        Parameters
        ----------
            config : Dict[str, Any]
                configuration needed to adjust the program behaviour
        """
        # ----- Attributes ----- #
        self._config: Dict[str, Any] = config
