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
from PIL import Image

import torch
from torch.utils.data import Dataset as TorchDataset

# IMPORT: data processing
from torchvision import transforms


class Dataset(TorchDataset):
    """
    Represents a DataSet, which will be modified depending on the use case.

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
        tokenizer : CLIPTokenizer
            Returns the dataset's tokenizer
        _load_image : torch.Tensor
            Loads an image from path
    """
    def __init__(
            self,
            params: Dict[str, Any],
            inputs: List[str],
            info: List[Dict[str, Any]]
    ):
        """
        Instantiates a InfoDataSet.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            inputs : List[str]
                input tensors' paths
            info : List[Dict[str, Any]]
                additional info about the data
        """
        # Mother Class
        super(Dataset, self).__init__()

        # Attributes
        self._params: Dict[str, Any] = params
        self._inputs: List[Union[str, torch.Tensor]] = inputs
        self._info: List[Dict[str, Any]] = info

        self._pre_process: transforms.Compose = transforms.Compose([
            transforms.Resize(params["img_size"], antialias=True),
            transforms.CenterCrop(params["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # Lazy loading
        if not params["lazy_loading"]:
            for idx, file_path in enumerate(tqdm(self._inputs, desc="loading the data in RAM.")):
                self._inputs[idx] = self._load_image(file_path)

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
        return self._pre_process(
            Image.open(path)  # .convert("RGB")
        ).type(torch.float16)

    def __getitem__(
            self,
            idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
            idx : int
                index of the item to get within the dataset

        Returns
        ----------
            Dict[str, torch.Tensor]
                the dataset's element and additional info
        """
        if self._params["lazy_loading"]:
            return {"image": self._load_image(self._inputs[idx])}
        else:
            return {"image": self._inputs[idx]}

    def __len__(self) -> int:
        """
        Returns
        ----------
            int
                dataset's length
        """
        return len(self._inputs)
