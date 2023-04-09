"""
Creator: Flokk
Date: 09/04/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import os

# IMPORT: data loading
from torch.utils.data import DataLoader

# IMPORT: project
from .datasets import LazyDataSet, TensorDataSet


class Loader:
    """
    Represents a loader.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour

    Methods
    ----------
        _extract_paths : List[str]
            Extracts file paths from a dataset
        _file_depth : int
            Calculates the depth of the file within the dataset
        _generate_data_loaders : Dict[str, DataLoader]
            Verifies the tensor's shape according to the desired dimension
    """

    def __init__(self, params: Dict[str, Any]):
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

    def _extract_paths(self, dataset_path: str) -> Dict[str, List[str]]:
        """
        Extracts file paths from a dataset.

        Parameters
        ----------
            dataset_path : str
                path to the dataset

        Returns
        ----------
            Dict[str, List[str]]
                file paths within the dataset
        """
        file_paths: Dict[str, List[str]] = {"train": list(), "valid": list()}

        data_idx = 0
        print("chemins de fichier")
        for root, dirs, files in os.walk(dataset_path, topdown=False):
            for file_path in map(lambda e: os.path.join(root, e), files):
                key = "train" if data_idx <= int(self._params["num_data"] * 0.95) else "valid"
                file_paths[key].append(file_path)

                data_idx += 1
                print(data_idx)
                if data_idx >= self._params["num_data"]:
                    break

        print(len(file_paths["train"]))
        print(len(file_paths["valid"]))
        return file_paths

    def _generate_data_loaders(self, file_paths: Dict[str, List[str]]) -> Dict[str, DataLoader]:
        """
        Generates data loaders using the extracted file paths.

        Parameters
        ----------
            file_paths : Dict[str, List[str]]
                file paths within the dataset

        Returns
        ----------
            Dict[str, DataLoader]
                the data loaders containing training data
        """
        return {
            "train": DataLoader(
                self._dataset_class(self._params, file_paths["train"]),
                batch_size=self._params["batch_size"], shuffle=True, drop_last=True
            ),
            "valid": DataLoader(
                self._dataset_class(self._params, file_paths["valid"]),
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
        file_paths = self._extract_paths(dataset_path)
        return self._generate_data_loaders(file_paths)
