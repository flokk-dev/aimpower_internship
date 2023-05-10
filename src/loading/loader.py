"""
Creator: Flokk
Date: 09/04/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: project
from .data_loader import PromptDataLoader


class Loader:
    """
    Represents a Loader.

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
        Instantiates a Loader.

        Parameters
        ----------
            config : Dict[str, Any]
                configuration needed to adjust the program behaviour
        """
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

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()

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

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()

    def __call__(
            self,
            dataset_path: str
    ) -> PromptDataLoader:
        """
        Parameters
        ----------
            dataset_path : str
                path to the dataset

        Returns
        ----------
            PromptDataLoader
                data loaders containing training data
        """
        return self._generate_data_loader(self._parse_dataset(dataset_path))
