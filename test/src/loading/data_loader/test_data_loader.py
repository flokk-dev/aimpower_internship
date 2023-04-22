"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import os
import torch

# IMPORT: project
import paths

from src.loading.data_loader import DataLoader, \
    LabelDataLoader, PromptDataLoader
from src.loading.dataset import Dataset, \
    LabelDataset, PromptDataset


# -------------------- CONSTANT -------------------- #

IMAGE_PATHS = [os.path.join(paths.TEST_RESOURCES_PATH, "data", f"img_{i}.png") for i in range(10)]
INFO = [{"label": 0, "prompt": "a trumpet"} for i in range(10)]


# -------------------- FIXTURES -------------------- #

def gen_data_loader(data_loader_type: str = "basic", lazy_loading: bool = True):
    # Data loaders
    data_loaders = {
        "basic": {"data_loader": DataLoader, "dataset": Dataset},
        "label": {"data_loader": LabelDataLoader, "dataset": LabelDataset},
        "prompt": {"data_loader": PromptDataLoader, "dataset": PromptDataset}
    }

    # Params
    params = {
        "data_loader": {
            "batch_size": 32
        },

        "dataset": {
            "lazy_loading": lazy_loading,
            "img_size": 512,

            "tokenizer": {
                "pipeline_path": "CompVis/stable-diffusion-v1-4"
            }
        }
    }

    # Initialization
    return data_loaders[data_loader_type]["data_loader"](
        params["data_loader"],
        data_loaders[data_loader_type]["dataset"](
            params["dataset"],
            inputs=IMAGE_PATHS,
            info=INFO
        )
    )


# -------------------- DATA LOADER -------------------- #

def test_data_loader():
    data_loader = gen_data_loader()

    # Collate
    for batch in data_loader:
        # Image
        assert isinstance(batch["image"], torch.Tensor)
        assert batch["image"].shape == torch.Size((2, 3, 512, 512))

        assert torch.min(batch["image"]) >= -1.0
        assert torch.max(batch["image"]) <= 1.0


# -------------------- DATA LOADER -------------------- #

def test_label_data_loader():
    data_loader = gen_data_loader(data_loader_type="label")

    # Collate
    for batch in data_loader:
        # Image
        assert isinstance(batch["image"], torch.Tensor)
        assert batch["image"].shape == torch.Size((2, 3, 512, 512))

        assert torch.min(batch["image"]) >= -1.0
        assert torch.max(batch["image"]) <= 1.0

        # Label
        assert isinstance(batch["label"], torch.Tensor)
        assert batch["label"].shape == torch.Size((2, ))

        assert torch.min(batch["label"]) >= 0
        assert torch.max(batch["label"]) <= 10


# -------------------- DATA LOADER -------------------- #

def test_prompt_data_loader():
    data_loader = gen_data_loader(data_loader_type="prompt")

    # Collate
    for batch in data_loader:
        # Image
        assert isinstance(batch["image"], torch.Tensor)
        assert batch["image"].shape == torch.Size((2, 3, 512, 512))

        assert torch.min(batch["image"]) >= -1.0
        assert torch.max(batch["image"]) <= 1.0

        # Prompt
        assert isinstance(batch["prompt"], torch.Tensor)
        assert batch["prompt"].shape == torch.Size((2, 4))
