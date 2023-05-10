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
from transformers import CLIPTokenizer


class PromptDataset(TorchDataset):
    """
    Represents a PromptDataset.

    Attributes
    ----------
        _config : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _prompts : List[str]
            additional info about the data
        tokenizer : CLIPTokenizer
            prompt tokenizer

    Methods
    ----------
        _tokenize : torch.Tensor
            Tokenizes a prompt
    """
    def __init__(
            self,
            config: Dict[str, Any],
            prompts: List[str]
    ):
        """
        Instantiates a PromptDataset.

        Parameters
        ----------
            config : Dict[str, Any]
                configuration needed to adjust the program behaviour
            prompts : List[str]
                additional info about the data
        """
        # Mother Class
        super(PromptDataset, self).__init__()

        # ----- Attributes ----- #
        self._config: Dict[str, Any] = config

        # Prompts
        self._prompts: List[str] = prompts

        # Tokenizer
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path=self._config["pipeline_path"],
            subfolder="tokenizer",
            revision="fp16"
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
                the dataset's elements as tensors
        """
        return {
            "prompt": self._tokenize(self._prompts[idx])
        }

    def __len__(self) -> int:
        """
        Returns
        ----------
            int
                dataset's length
        """
        return len(self._prompts)


class ImagePromptDataset(PromptDataset):
    """
    Represents a ImagePromptDataset.

    Attributes
    ----------
        _config : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _prompts : List[str]
            prompts as tensors
        tokenizer : CLIPTokenizer
            prompt tokenizer
        _images : List[str | torch.Tensor]
            images as tensors
        _pre_process : transforms.Compose
            image's pre-processing

    Methods
    ----------
        _tokenize : torch.Tensor
            Tokenizes a prompt
        _load_image : torch.Tensor
            Loads an images from path
    """
    def __init__(
            self,
            config: Dict[str, Any],
            prompts: List[str],
            images: List[str]
    ):
        """
        Instantiates a ImagePromptDataset.

        Parameters
        ----------
            config : Dict[str, Any]
                configuration needed to adjust the program behaviour
            prompts : List[str]
                prompts
            images : List[str]
                images' paths
        """
        # Mother Class
        super(ImagePromptDataset, self).__init__(config, prompts)

        # ----- Attributes ----- #
        # Pre-processing
        self._pre_process: transforms.Compose = transforms.Compose([
            transforms.Resize(config["img_size"], antialias=True),
            transforms.CenterCrop(config["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # Images
        self._images: List[str | torch.Tensor] = images
        if not self._config["lazy_loading"]:
            for idx, file_path in enumerate(tqdm(self._images, desc="loading the data in RAM.")):
                self._images[idx] = self._load_image(file_path)

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
            Image.open(path).convert("RGB")
        )

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
                the dataset's elements as tensors
        """
        if self._config["lazy_loading"]:
            image = self._load_image(self._images[idx])
        else:
            image = self._images[idx]

        return {**super().__getitem__(idx), "image": image}
