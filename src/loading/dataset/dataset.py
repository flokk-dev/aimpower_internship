"""
Creator: Flokk
Date: 09/04/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: data loading
from PIL import Image
import torch

# IMPORT: data processing
from torchvision import transforms


class DataSet(torch.utils.data.Dataset):
    """
    Represents a general DataSet, that will be modified depending on the use case.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _inputs : List[Union[str, torch.Tensor]]
            input tensors
        _info : List[Dict[str, Any]]
            additional info about the data
        _pre_process: transforms.Compose
            pre-processing to apply on each data

    Methods
    ----------
        _load_image : torch.Tensor
            Loads an image from path
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
        super(DataSet, self).__init__()

        # Attributes
        self._params: Dict[str, Any] = params
        self._inputs: List[Union[str, torch.Tensor]] = inputs

        self._info: List[Dict[str, Any]] = dataset_info

        # Data pre-processing
        self._pre_process: transforms.Compose = transforms.Compose([
            transforms.Resize((params["img_size"], params["img_size"]), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def _load_image(
            self,
            path: str
    ):
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
        img: Image = Image.open(path)
        tensor: torch.Tensor = transforms.PILToTensor()(img) / 255

        return self._pre_process(tensor).type(torch.float16)

    def __getitem__(
            self,
            idx: int
    ) -> torch.Tensor:
        """
        Parameters
        ----------
            idx : int
                index of the item to get within the dataset

        Returns
        ----------
            torch.Tensor
                the dataset's element as a tensor

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
