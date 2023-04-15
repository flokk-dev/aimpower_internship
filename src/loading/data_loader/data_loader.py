"""
Creator: Flokk
Date: 09/04/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: data loading
import torch
from torch.utils.data import DataLoader as TorchDataLoader

# IMPORT: project
from src.loading.dataset import Dataset


class DataLoader(TorchDataLoader):
    """
    Represents a general data loader, that will be derived depending on the use case.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour

    Methods
    ----------
        _collate_fn : Dict[str, torch.Tensor]
            Defines the data loader's behaviour when getting data.
    """

    def __init__(
            self,
            params: Dict[str, Any],
            dataset: Dataset
    ):
        """
        Instantiates a DataLoader.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            dataset: Dataset
                dataset containing file paths
        """
        # Mother Class
        super(DataLoader, self).__init__(
            dataset,
            batch_size=params["batch_size"], shuffle=True, drop_last=True,
            collate_fn=self._collate_fn
        )

        # Attributes
        self._params: Dict[str, Any] = params

    @staticmethod
    def _collate_fn(
            data: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Defines the data loader's behaviour when getting data.

        Parameters
        ----------
            data : List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]
                list containing the recovered data

        Returns
        ----------
            Dict[str, torch.Tensor]
                the dataset's element and additional info
        """
        return {
            "image": torch.stack(
                [e["image"] for e in data]
            ).to(memory_format=torch.contiguous_format).type(torch.float16)
        }
