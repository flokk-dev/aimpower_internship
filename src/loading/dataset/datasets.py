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

from .dataset import DataSet


class LazyDataSet(DataSet):
    """
    Represents a LazyDataSet.

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
        Instantiates a LazyDataSet.

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
        super(LazyDataSet, self).__init__(params, inputs, dataset_info)

    def __getitem__(
            self,
            idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
            idx : int
                index of the item to get within the dataset

        Returns
        ----------
            torch.Tensor
                the dataset's element as a tensor
            torch.Tensor
                additional info about the data
        """
        return self._load_image(self._inputs[idx]), self._info[idx]["label"]


class TensorDataSet(DataSet):
    """
    Represents a TensorDataSet.

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
        Instantiates a TensorDataSet.

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
        super(TensorDataSet, self).__init__(params, inputs, dataset_info)

        for idx, file_path in enumerate(tqdm(self._inputs, desc="loading the data in RAM.")):
            self._inputs[idx] = self._load_image(file_path)

    def __getitem__(
            self,
            idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
            idx : int
                index of the item to get within the dataset

        Returns
        ----------
            torch.Tensor
                the dataset's element as a tensor
            torch.Tensor
                additional info about the data
        """
        return self._inputs[idx], self._info[idx]["label"]
