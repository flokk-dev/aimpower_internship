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

# IMPORT: data processing
from transformers import CLIPTokenizer

# IMPORT: project
from .dataset import Dataset


class LabelDataset(Dataset):
    """
    Represents a LabelDataset.

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
        _str_to_tensor: torch.Tensor
            Casts a string into a tensor
    """
    def __init__(
            self,
            params: Dict[str, Any],
            inputs: List[str],
            info: List[Dict[str, Any]]
    ):
        """
        Instantiates a LabelDataset.

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
        super(LabelDataset, self).__init__(params, inputs, info)

    @staticmethod
    def _str_to_tensor(elem: str):
        """
        Casts a string into a tensor.

        Parameters
        ----------
            elem : str
                element to cast

        Returns
        ----------
            torch.Tensor
                casted element
        """
        return torch.tensor([int(elem)]).type(torch.uint8)

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
            return {
                "image": self._load_image(self._inputs[idx]),
                "label": self._str_to_tensor(self._info[idx]["label"])
            }

        return {
            "image": self._inputs[idx],
            "label": self._str_to_tensor(self._info[idx]["label"])
        }


class PromptDataset(Dataset):
    """
    Represents a PromptDataset.

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
        tokenizer: CLIPTokenizer
            object needed to tokenize a prompt

    Methods
    ----------
        _load_image : torch.Tensor
            Loads an image from path
        _init_tokenizer : CLIPTokenizer
            Instantiates a tokenizer
        _tokenize: torch.Tensor
            Tokenizes a prompt
    """
    def __init__(
            self,
            params: Dict[str, Any],
            inputs: List[str],
            info: List[Dict[str, Any]]
    ):
        """
        Instantiates a PromptDataset.

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
        super(PromptDataset, self).__init__(params, inputs, info)

        # Attributes
        self.tokenizer: CLIPTokenizer = self._init_tokenizer(
            self._params["tokenizer"]["pipeline_path"]
        )

    @staticmethod
    def _init_tokenizer(
            pipeline_path: str
    ) -> CLIPTokenizer:
        """
        Instantiates a tokenizer.

        Parameters
        ----------
            pipeline_path : str
                path to the pretrained pipeline
        """
        if not pipeline_path:
            return None

        return CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path=pipeline_path,
            subfolder="tokenizer",
            # revision="fp16"
        )

    def _tokenize(
            self,
            prompt: str
    ) -> torch.Tensor:
        """
        Tokenizes a prompt.

        Parameters
        ----------
            prompt : str
                additional info about a data

        Returns
        ----------
            torch.Tensor
                the prompt as a tensor
        """
        return self.tokenizer(
            prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

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
            return {
                "image": self._load_image(self._inputs[idx]),
                "prompt": self._tokenize(self._info[idx]["prompt"])
            }

        return {
            "image": self._inputs[idx],
            "prompt": self._tokenize(self._info[idx]["prompt"])
        }
