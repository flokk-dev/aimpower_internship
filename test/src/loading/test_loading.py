"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import os

# IMPORT: project
import paths

from src.loading import Loader
from src.loading.data_loader import DataLoader, \
    LabelDataLoader, PromptDataLoader


# -------------------- CONSTANT -------------------- #

DATA_PATH = os.path.join(paths.TEST_RESOURCES_PATH, "data")


# -------------------- FIXTURES -------------------- #

def gen_loader(loading_type: str = "basic"):
    return Loader(
        params={
            "loading_type": loading_type,
            "num_data": 10,

            "data_loader": {
                "batch_size": 2
            },

            "dataset": {
                "lazy_loading": True,
                "img_size": 512,

                "tokenizer": {
                    "pipeline_path": "CompVis/stable-diffusion-v1-4"
                }
            }
        }
    )


# -------------------- LOADER -------------------- #

def test_loader_parse_dataset():
    loader = gen_loader()

    # Parser
    file_paths, data_info = loader._parse_dataset(DATA_PATH)

    assert isinstance(file_paths, list)
    assert isinstance(file_paths[0], str)
    assert len(file_paths) <= loader._params["num_data"]

    assert isinstance(data_info, list)
    assert isinstance(data_info[0], dict)
    assert len(data_info) <= loader._params["num_data"]


def test_loader_generate_data_loader():
    loader = gen_loader()

    # Generate data loader
    file_paths, data_info = loader._parse_dataset(DATA_PATH)

    data_loader = loader._generate_data_loader(file_paths, data_info)
    assert isinstance(data_loader, DataLoader)


def test_loader_call():
    loader = gen_loader()

    # Generate data loader
    data_loader = loader(DATA_PATH)
    assert isinstance(data_loader, DataLoader)


# -------------------- LABEL LOADER -------------------- #

def test_label_loader_generate_data_loader():
    loader = gen_loader(loading_type="label")

    # Generate data loader
    file_paths, data_info = loader._parse_dataset(DATA_PATH)

    data_loader = loader._generate_data_loader(file_paths, data_info)
    assert isinstance(data_loader, LabelDataLoader)


def test_label_loader_call():
    loader = gen_loader(loading_type="label")

    # Generate data loader
    data_loader = loader(DATA_PATH)
    assert isinstance(data_loader, LabelDataLoader)


# -------------------- PROMPT LOADER -------------------- #

def test_prompt_loader_generate_data_loader():
    loader = gen_loader(loading_type="prompt")

    # Generate data loader
    file_paths, data_info = loader._parse_dataset(DATA_PATH)

    data_loader = loader._generate_data_loader(file_paths, data_info)
    assert isinstance(data_loader, PromptDataLoader)


def test_prompt_loader_call():
    loader = gen_loader(loading_type="prompt")

    # Generate data loader
    data_loader = loader(DATA_PATH)
    assert isinstance(data_loader, PromptDataLoader)
