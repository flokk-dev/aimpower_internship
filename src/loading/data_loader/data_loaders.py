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

# IMPORT: data processing
from transformers import CLIPTokenizer

# IMPORT: project
from src.loading.dataset import PromptDataset, ImagePromptDataset


class PromptDataLoader(TorchDataLoader):
    """
    Represents a data PromptDataLoader.

    Attributes
    ----------
        _config : Dict[str, Any]
            configuration needed to adjust the program behaviour

    Methods
    ----------
        _collate_fn : Dict[str, torch.Tensor]
            Defines the data loader's behaviour when getting data.
    """
    def __init__(
            self,
            config: Dict[str, Any],
            dataset: PromptDataset
    ):
        """
        Instantiates a PromptDataLoader.

        Parameters
        ----------
            config : Dict[str, Any]
                configuration needed to adjust the program behaviour
            dataset: PromptDataset
                dataset containing file paths
        """
        # Mother Class
        super(PromptDataLoader, self).__init__(
            dataset,
            batch_size=config["batch_size"], shuffle=True, drop_last=True,
            collate_fn=self._collate_fn
        )

        # Attributes
        self._config: Dict[str, Any] = config

    def _collate_fn(
            self,
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
            "prompt": self.dataset.tokenizer.pad(
                {"input_ids": [e["prompt"] for e in data]}, padding=True, return_tensors="pt"
            ).input_ids
        }


class ImagePromptDataLoader(PromptDataLoader):
    """
    Represents a data ImagePromptDataLoader.

    Attributes
    ----------
        _config : Dict[str, Any]
            configuration needed to adjust the program behaviour

    Methods
    ----------
        _collate_fn : Dict[str, torch.Tensor]
            Defines the data loader's behaviour when getting data.
    """
    def __init__(
            self,
            config: Dict[str, Any],
            dataset: ImagePromptDataset
    ):
        """
        Instantiates a ImagePromptDataLoader.

        Parameters
        ----------
            config : Dict[str, Any]
                configuration needed to adjust the program behaviour
            dataset: ImagePromptDataset
                dataset containing file paths
        """
        # Mother Class
        super(ImagePromptDataLoader, self).__init__(config, dataset)

    def _collate_fn(
            self,
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
        images = torch.stack([e["image"] for e in data]).to(
            memory_format=torch.contiguous_format
        ).type(torch.float16)

        return {**super()._collate_fn(data), "image": images}