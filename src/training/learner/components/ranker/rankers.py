"""
Creator: Flokk
Date: 09/04/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: data visualization
import matplotlib.pyplot as plt

# IMPORT: deep learning
import torch

import utils
# IMPORT: project
from .reward_function import PickAPicScore


class HFRanker:
    """
    Represents a HFRanker.

    Methods
    ----------
        rank
            Ranks images
    """

    def __init__(
            self
    ):
        """ Instantiates a HFRanker. """
        # ----- Mother Class ----- #
        super(HFRanker, self).__init__()

    @staticmethod
    def _rank(
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
        """
        # Plot images
        utils.plot_images(prompt, images)

        # Asks to rank them
        user_input = input("Rank the images from best to worst (separated by a '-'):")
        ranking_idx = torch.Tensor([int(e) for e in user_input.split("-")])

        return images[ranking_idx]


class AutomaticHFRanker:
    """
    Represents a AutomaticHFRanker.

    Methods
    ----------
        rank
            Ranks images
    """

    def __init__(
            self
    ):
        """ Instantiates a AutomaticHFRanker. """
        # ----- Mother Class ----- #
        super(AutomaticHFRanker, self).__init__()

        # ----- Attributes ----- #
        # Reward function
        self._reward_fnc: PickAPicScore = PickAPicScore(device="cuda")

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
        """
        rewards = self._reward_fnc(prompt, images)
        _, ranking_idx = torch.sort(rewards, descending=True)

        return images[ranking_idx]
