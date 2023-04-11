"""
Creator: Flokk
Date: 09/04/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
from tqdm import tqdm

# IMPORT: data loading
import torch

# IMPORT: project
from .dataset import DataSet


class InfoDataSet(DataSet):
    """
    Represents a InfoDataSet.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _inputs : List[torch.Tensor]
            input tensors
        _info : List[Dict[str, Any]]
            additional info about the data
    """
    def __init__(
            self,
            params: Dict[str, Any],
            inputs: List[str],
            dataset_info: List[Dict[str, Any]]
    ):
        """
        Instantiates a InfoDataSet.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            inputs : List[str]
                input tensors' paths
            dataset_info : List[Dict[str, Any]]
                additional info about the data
        """
        # Mother Class
        super(InfoDataSet, self).__init__(params, inputs)

        # Attributes
        self._info = dataset_info

    def __getitem__(
            self,
            idx: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Parameters
        ----------
            idx : int
                index of the item to get within the dataset

        Returns
        ----------
            torch.Tensor
                the dataset's element as a tensor
            Dict[str, Any]
                additional info about the data
        """
        if not self._params["lazy_loading"]:
            return self._inputs[idx], self._info[idx]
        else:
            return self._load_image(self._inputs[idx]), self._info[idx]
