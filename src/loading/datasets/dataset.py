"""
Creator: Flokk
Date: 09/04/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: data loading
import numpy as np
import cv2

import torch
from torchvision import transforms


class DataSet(torch.utils.data.Dataset):
    """
    Represents a general dataset, that will be derived depending on the use case.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _inputs : List[Union[str, torch.Tensor]]
            input tensors

    Methods
    ----------
        _load_image : torch.Tensor
            Loads an image from path
    """
    def __init__(self, params: Dict[str, Any], inputs: List[str]):
        """
        Instantiates a DataSet.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            inputs : List[str]
                input tensors' paths
        """
        # Mother Class
        super(DataSet, self).__init__()

        # Attributes
        self._params: Dict[str, Any] = params
        self._inputs: List[Union[str, torch.Tensor]] = inputs

        self._pre_process = transforms.Compose([
            transforms.Resize((params["img_size"], params["img_size"]), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def _load_image(self, path: str):
        """
        Loads an image from path.

        Parameters
        ----------
            path : str
                path of the image to load

        Returns
        ----------
            torch.Tensor
                the image as a tensor
        """
        array: np.ndarray = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        tensor: torch.Tensor = torch.from_numpy(array).type(torch.float16)

        return self._pre_process(tensor.permute(2, 0, 1))

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Parameters
        ----------
            idx : int
                index of the item to get within the dataset

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()

    def __len__(self) -> int:
        """
        Returns
        ----------
            int
                dataset's length
        """
        return len(self._inputs)
