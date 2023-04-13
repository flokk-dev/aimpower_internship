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

# IMPORT: data loading
from torch.utils.data import DataLoader

# IMPORT: project
from .dataset import LazyDataSet, TensorDataSet


class Loader:
    """
    Represents a Loader, which will be modified depending on the use case.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour

    Methods
    ----------
        _parse_dataset : List[str]
            Parses the dataset to extract some info
        _generate_data_loaders : Dict[str, DataLoader]
            Generates a data loaders using extracted file paths
    """
    _DATASETS = {"basic": TensorDataSet, "lazy": LazyDataSet}

    def __init__(
            self,
            params: Dict[str, Any]
    ):
        """
        Instantiates a Loader.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
        """
        # Attributes
        self._params: Dict[str, Any] = params

    def _parse_dataset(
            self,
            dataset_path: str
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Parses the dataset to extract some info.

        Parameters
        ----------
            dataset_path : str
                path to the dataset

        Returns
        ----------
            List[str]
                file paths within the dataset
            List[Dict[str, Any]]
                additional info about the data
        """
        # Parses dataset info via a csv file
        dataset_info: Dict[int, Dict[str, Any]] = pd.read_csv(
            os.path.join(dataset_path, "dataset_info.csv")
        ).to_dict(orient="index")

        # Extracts and uses the info
        file_paths: List[str] = list()
        data_info: List[Dict[str, Any]] = list()

        for idx, row in dataset_info.items():
            file_paths.append(os.path.join(dataset_path, row["image_path"]))
            data_info.append(row)

            if idx >= self._params["num_data"] - 1:
                break

        return file_paths, data_info

    def _generate_data_loader(
            self,
            file_paths: List[str],
            data_info: List[Dict[str, Any]]
    ) -> DataLoader:
        """
        Generates a data loader using extracted file paths.

        Parameters
        ----------
            file_paths : Dict[str, List[str]]
                file paths within the dataset
            data_info : Dict[str, List[Dict[str, Any]]]
                additional info about the data

        Returns
        ----------
            DataLoader
                data loader containing training data
        """
        return DataLoader(
            self._DATASETS[self._params["loading_method"]](
                self._params, file_paths, data_info
            ),
            batch_size=self._params["batch_size"], shuffle=True, drop_last=True
        )

    def __call__(self, dataset_path: str) -> DataLoader:
        """
        Parameters
        ----------
            dataset_path : str
                path to the dataset

        Returns
        ----------
            Dict[str, DataLoader]
                data loaders containing training data
        """
        return self._generate_data_loader(*self._parse_dataset(dataset_path))
