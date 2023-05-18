"""
Creator: Flokk
Date: 09/04/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: deep learning
import torch


class Ranker:
    """
    Represents a Ranker.

    Methods
    ----------
        rank
            Ranks images
    """

    def __init__(
            self
    ):
        """ Instantiates a Ranker. """
        pass

    def _rank(
        self,
        prompt: str,
        images: torch.tensor
    ):
        """
        Instantiates a Ranker.

        Parameters
        ----------
            prompt : str
                prompt that leads to the images
            images : torch.tensor
                images to rank

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()

    def __call__(
        self,
        prompt: str,
        images: torch.tensor
    ):
        """
        Parameters
        ----------
            prompt : str
                prompt that leads to the images
            images : torch.tensor
                images to rank
        """
        return self._rank(prompt, images)
