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
    Represents a Loader, that will be modified depending on the use case.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour

    Methods
    ----------
        _parse_dataset : List[str]
            Parses the dataset to extract some info
        _file_depth : int
            Calculates the depth of the file within the dataset
        _generate_data_loaders : Dict[str, DataLoader]
            Verifies the tensor's shape according to the desired dimension
    """

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
        self._dataset_class = LazyDataSet if params["lazy_loading"] else TensorDataSet

    def _parse_dataset(
            self,
            dataset_path: str
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[Dict[str, Any]]]]:
        """
        Parses the dataset to extract some info

        Parameters
        ----------
            dataset_path : str
                path to the dataset

        Returns
        ----------
            Dict[str, List[str]]
                file paths within the dataset
            Dict[str, List[Dict[str, Any]]]
                additional info about the data
        """
        # Parses dataset info via a csv fileordonner
        dataset_info: Dict[int, Dict[str, Any]] = pd.read_csv(
            os.path.join(dataset_path, "dataset_info.csv")
        ).to_dict(orient="index")

        # Extracts and uses the info
        file_paths: Dict[str, List[str]] = {"train": list(), "valid": list()}
        data_info: Dict[str, List[Dict[str, Any]]] = {"train": list(), "valid": list()}

        for data_idx, row in dataset_info.items():
            step = "train" if data_idx < int(self._params["num_data"] * 0.95) else "valid"

            file_paths[step].append(os.path.join(dataset_path, row["image_path"]))
            data_info[step].append(row)

        return file_paths, data_info

    def _generate_data_loaders(
            self,
            file_paths: Dict[str, List[str]],
            data_info: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, DataLoader]:
        """
        Generates data loaders using the extracted file paths.

        Parameters
        ----------
            file_paths : Dict[str, List[str]]
                file paths within the dataset
            data_info : Dict[str, List[Dict[str, Any]]]
                additional info about the data

        Returns
        ----------
            Dict[str, DataLoader]
                the data loaders containing training data
        """
        return {
            "train": DataLoader(
                self._dataset_class(self._params, file_paths["train"], data_info["train"]),
                batch_size=self._params["batch_size"], shuffle=True, drop_last=True
            ),
            "valid": DataLoader(
                self._dataset_class(self._params, file_paths["valid"], data_info["valid"]),
                batch_size=self._params["batch_size"], shuffle=True, drop_last=True
            ),
        }

    def __call__(self, dataset_path: str) -> Dict[str, DataLoader]:
        """
        Parameters
        ----------
            dataset_path : str
                path to the dataset

        Returns
        ----------
            Dict[str, DataLoader]
                the data loaders containing training data
        """
        return self._generate_data_loaders(self._parse_dataset(dataset_path))
