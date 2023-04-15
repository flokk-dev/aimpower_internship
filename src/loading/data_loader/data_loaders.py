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

# IMPORT: project
from .data_loader import DataLoader
from src.loading.dataset import LabelDataset, PromptDataset


class LabelDataLoader(DataLoader):
    """
    Represents a LabelDataLoader.

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
            dataset: LabelDataset
    ):
        """
        Instantiates a DataLoader.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            dataset: LabelDataset
                dataset containing file paths
        """
        # Mother Class
        super(LabelDataLoader, self).__init__(params, dataset)

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
            "image": torch.stack(
                [e["image"] for e in data]
            ).to(memory_format=torch.contiguous_format).type(torch.float16),

            "label": torch.cat([e["label"] for e in data])
        }


class PromptDataLoader(DataLoader):
    """
    Represents a PromptDataLoader.

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
            dataset: PromptDataset
    ):
        """
        Instantiates a PromptDataLoader.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            dataset: PromptDataset
                dataset containing file paths
        """
        # Mother Class
        super(PromptDataLoader, self).__init__(params, dataset)

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
            "image": torch.stack(
                [e["image"] for e in data]
            ).to(memory_format=torch.contiguous_format).type(torch.float16),

            "prompt": self.dataset.tokenizer.pad(
                {"input_ids": [e["prompt"] for e in data]}, padding=True, return_tensors="pt"
            ).input_ids
        }
