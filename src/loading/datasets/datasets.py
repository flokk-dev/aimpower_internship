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


class LazyDataSet(DataSet):
    """
    Represents a lazy dataset.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _inputs : List[str]
            input tensors
    """
    def __init__(self, params: Dict[str, Any], inputs: List[str]):
        """
        Instantiates a LazyDataSet.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            inputs : List[str]
                input tensors' paths
        """
        # Mother Class
        super(LazyDataSet, self).__init__(params, inputs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Parameters
        ----------
            idx : int
                index of the item to get within the dataset

        Returns
        ----------
            torch.Tensor
                the dataset's element as a tensor
        """
        return self._load_image(self._inputs[idx])


class TensorDataSet(DataSet):
    """
    Represents a tensor dataset.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _inputs : List[torch.Tensor]
            input tensors
    """
    def __init__(self, params: Dict[str, Any], inputs: List[str]):
        """
        Instantiates a TensorDataSet.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            inputs : List[str]
                input tensors' paths
        """
        # Mother Class
        super(TensorDataSet, self).__init__(params, inputs)

        if not params["lazy_loading"]:
            for idx, file_path in enumerate(tqdm(self._inputs, desc="loading the data in RAM.")):
                self._inputs[idx] = self._load_image(file_path)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Parameters
        ----------
            idx : int
                index of the item to get within the dataset

        Returns
        ----------
            torch.Tensor
                the dataset's element as a tensor
        """
        values = torch.unique(self._inputs[idx])
        print(f"dataset + {idx} + {min(values)} + {max(values)} + {self._inputs[idx].shape}")

        return self._inputs[idx]
