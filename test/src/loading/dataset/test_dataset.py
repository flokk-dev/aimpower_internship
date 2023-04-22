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

from src.loading.dataset import Dataset, \
    LabelDataset, PromptDataset


# -------------------- CONSTANT -------------------- #

IMAGE_PATHS = [os.path.join(paths.TEST_RESOURCES_PATH, "data", f"img_{i}.png") for i in range(10)]
INFO = [{"label": 0, "prompt": "a trumpet"} for i in range(10)]


# -------------------- FIXTURES -------------------- #

def gen_dataset(dataset_type: str = "basic", lazy_loading: bool = True):
    # Datasets
    datasets = {
        "basic": Dataset,
        "label": LabelDataset,
        "prompt": PromptDataset
    }

    # Initialization
    return datasets[dataset_type](
        params={
            "lazy_loading": lazy_loading,
            "img_size": 512,

            "tokenizer": {
                "pipeline_path": "CompVis/stable-diffusion-v1-4"
            }
        },
        inputs=IMAGE_PATHS,
        info=INFO
    )


# -------------------- DATASET -------------------- #

def test_dataset():
    # Lazy
    dataset_lazy = gen_dataset()
    for e in dataset_lazy._inputs:
        assert isinstance(e, str)

    # Not lazy
    dataset_not_lazy = gen_dataset(lazy_loading=False)
    for e in dataset_not_lazy._inputs:
        assert isinstance(e, torch.Tensor)


def test_dataset_load_image():
    dataset = gen_dataset()

    # Image
    image = dataset._load_image(IMAGE_PATHS[0])

    assert isinstance(image, torch.Tensor)
    assert image.shape == torch.Size((3, 512, 512))

    assert torch.min(image) >= -1.0
    assert torch.max(image) <= 1.0


def test_dataset_get_item():
    dataset = gen_dataset()

    # Get item
    elem = dataset[0]
    assert isinstance(elem, dict)

    # Image
    assert isinstance(elem["image"], torch.Tensor)
    assert elem["image"].shape == torch.Size((3, 512, 512))

    assert torch.min(elem["image"]) >= -1.0
    assert torch.max(elem["image"]) <= 1.0


# -------------------- LABEL DATASET -------------------- #

def test_label_dataset_str_to_tensor():
    dataset = gen_dataset(dataset_type="label")

    # Label
    label = dataset._str_to_tensor(dataset._info[0]["label"])

    assert isinstance(label, torch.Tensor)
    assert label.shape == torch.Size((1, ))

    assert label >= 0
    assert label <= 10


def test_label_dataset_get_item():
    dataset = gen_dataset(dataset_type="label")

    # Get item
    elem = dataset[0]
    assert isinstance(elem, dict)

    # Image
    assert isinstance(elem["image"], torch.Tensor)
    assert elem["image"].shape == torch.Size((3, 512, 512))

    assert torch.min(elem["image"]) >= -1.0
    assert torch.max(elem["image"]) <= 1.0

    # Label
    assert isinstance(elem["label"], torch.Tensor)
    assert elem["label"].shape == torch.Size((1, ))

    assert torch.min(elem["label"]) >= 0
    assert torch.max(elem["label"]) <= 10


# -------------------- LABEL DATASET -------------------- #

def test_prompt_dataset_tokenize():
    dataset = gen_dataset(dataset_type="prompt")

    # Token
    prompt = dataset._info[0]["prompt"]
    token = dataset._tokenize(prompt)

    assert isinstance(token, list)
    assert isinstance(token[0], int)

    assert len(token) == len(prompt.split(" ")) + 2


def test_prompt_dataset_get_item():
    dataset = gen_dataset(dataset_type="prompt")

    # Get item
    elem = dataset[0]
    assert isinstance(elem, dict)

    # Image
    assert isinstance(elem["image"], torch.Tensor)
    assert elem["image"].shape == torch.Size((3, 512, 512))

    assert torch.min(elem["image"]) >= -1.0
    assert torch.max(elem["image"]) <= 1.0

    # TOKEN
    assert isinstance(elem["prompt"], list)
    assert isinstance(elem["prompt"][0], int)

    assert len(elem["prompt"]) == len(dataset._info[0]["prompt"].split(" ")) + 2
