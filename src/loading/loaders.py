"""
Creator: Flokk
Date: 09/04/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

import os
import pandas as pd

# IMPORT: project
from .loader import Loader

from .dataset import PromptDataset, ImagePromptDataset
from .data_loader import PromptDataLoader, ImagePromptDataLoader


class PromptLoader(Loader):
    """
    Represents a PromptLoader.

    Attributes
    ----------
        _config : Dict[str, Any]
            configuration needed to adjust the program behaviour

    Methods
    ----------
        _parse_dataset : List[str]
            Parses the dataset to extract some information
        _generate_data_loaders : Dict[str, DataLoader]
            Generates a data loader using extracted dataset's information
    """
    def __init__(
            self,
            config: Dict[str, Any]
    ):
        """
        Instantiates a PromptLoader.

        Parameters
        ----------
            config : Dict[str, Any]
                configuration needed to adjust the program behaviour
        """
        # Mother class
        super(PromptLoader, self).__init__(config)

        # Attributes
        self._config: Dict[str, Any] = config

    def _parse_dataset(
            self,
            dataset_path: str
    ) -> Dict[str, List[str]]:
        """
        Parses the dataset to extract some information.

        Parameters
        ----------
            dataset_path : str
                path to the dataset

        Returns
        ----------
            Dict[str, List[str]]
                dataset's content
        """
        # Parses dataset info via a csv file
        dataset_info: Dict[int, Dict[str, Any]] = pd.read_csv(
            os.path.join(dataset_path, "dataset_info.csv")
        ).to_dict(orient="index")

        # Extracts and uses the info
        prompts: List[str] = list()

        for idx, row in dataset_info.items():
            prompts.append(row["prompt"])

            if idx >= self._config["num_data"] - 1:
                break

        return {"prompts": prompts}

    def _generate_data_loader(
            self,
            dataset_info: Dict[str, List[str]]
    ) -> PromptDataLoader:
        """
        Generates a data loader using extracted dataset's information.

        Parameters
        ----------
            dataset_info : Dict[str, List[str]]
                dataset's content

        Returns
        ----------
            DataLoader
                data loader containing training data
        """
        return PromptDataLoader(
            self._config, PromptDataset(self._config, **dataset_info),
        )


class ImagePromptLoader(Loader):
    """
    Represents a ImagePromptLoader.

    Attributes
    ----------
        _config : Dict[str, Any]
            configuration needed to adjust the program behaviour

    Methods
    ----------
        _parse_dataset : List[str]
            Parses the dataset to extract some information
        _generate_data_loaders : Dict[str, DataLoader]
            Generates a data loader using extracted dataset's information
    """
    def __init__(
            self,
            config: Dict[str, Any]
    ):
        """
        Instantiates a ImagePromptLoader.

        Parameters
        ----------
            config : Dict[str, Any]
                configuration needed to adjust the program behaviour
        """
        # Mother class
        super(ImagePromptLoader, self).__init__(config)

        # Attributes
        self._config: Dict[str, Any] = config

    def _parse_dataset(
            self,
            dataset_path: str
    ) -> Dict[str, List[str]]:
        """
        Parses the dataset to extract some information.

        Parameters
        ----------
            dataset_path : str
                path to the dataset

        Returns
        ----------
            Dict[str, List[str]]
                dataset's content
        """
        # Parses dataset info via a csv file
        dataset_info: Dict[int, Dict[str, Any]] = pd.read_csv(
            os.path.join(dataset_path, "dataset_info.csv")
        ).to_dict(orient="index")

        # Extracts and uses the info
        prompts: List[str] = list()
        image_paths: List[str] = list()

        for idx, row in dataset_info.items():
            prompts.append(row["prompt"])
            image_paths.append(os.path.join(dataset_path, row["image_path"]))

            if idx >= self._config["num_data"] - 1:
                break

        return {"prompts": prompts, "images": image_paths}

    def _generate_data_loader(
            self,
            dataset_info: Dict[str, List[str]]
    ) -> ImagePromptDataLoader:
        """
        Generates a data loader using extracted dataset's information.

        Parameters
        ----------
            dataset_info : Dict[str, List[str]]
                dataset's content

        Returns
        ----------
            ImagePromptDataLoader
                data loader containing training data
        """
        return ImagePromptDataLoader(
            self._config, ImagePromptDataset(self._config, **dataset_info),
        )
