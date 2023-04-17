"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import os
from diffusers import UNet2DModel, UNet2DConditionModel

# IMPORT: test
import pytest

# IMPORT: project
import paths

from src.training.learner.components.model import ModelManager
from src.training.learner.components.model.models import \
    init_unet, load_unet, \
    init_conditioned_unet, load_conditioned_unet


# -------------------- CONSTANT -------------------- #

UNET_PATH = os.path.join(paths.TEST_RESOURCES_PATH, "pipelines", "1")
CONDITIONED_UNET_PATH = os.path.join(paths.TEST_RESOURCES_PATH, "pipelines", "2")


# -------------------- FIXTURES -------------------- #

@pytest.fixture(scope="function")
def model_manager():
    return ModelManager()


@pytest.fixture(scope="function")
def params():
    return {
        "type": "unet",
        "pipeline_path": "",

        "args": {
            "sample_size": 64,
            "in_channels": 3,
            "out_channels": 3,
            "block_out_channels": [64, 64, 128, 128],
            "num_labels": 0
        }
    }


# -------------------- U-Net -------------------- #

def test_init_unet(model_manager, params):
    unet = init_unet(params=params["args"])
    assert isinstance(unet, UNet2DModel)

    unet = model_manager.__call__(
        params["type"],
        params["args"],
        pipeline_path=params["pipeline_path"]
    )
    assert isinstance(unet, UNet2DModel)


def test_load_unet(model_manager, params):
    unet = load_unet(pipeline_path=UNET_PATH)
    assert isinstance(unet, UNet2DModel)

    unet = model_manager.__call__(
        params["type"],
        params["args"],
        pipeline_path=UNET_PATH
    )
    assert isinstance(unet, UNet2DModel)


# -------------------- Conditioned U-Net -------------------- #

def test_init_conditioned_unet(model_manager, params):
    params["type"] = "conditioned unet"

    conditioned_unet = init_conditioned_unet(params=params["args"])
    assert isinstance(conditioned_unet, UNet2DConditionModel)

    conditioned_unet = model_manager.__call__(
        params["type"],
        params["args"],
        pipeline_path=params["pipeline_path"]
    )
    assert isinstance(conditioned_unet, UNet2DConditionModel)


def test_load_conditioned_unet(model_manager, params):
    params["type"] = "conditioned unet"

    conditioned_unet = load_conditioned_unet(pipeline_path=CONDITIONED_UNET_PATH)
    assert isinstance(conditioned_unet, UNet2DConditionModel)

    conditioned_unet = model_manager.__call__(
        params["type"],
        params["args"],
        pipeline_path=CONDITIONED_UNET_PATH
    )
    assert isinstance(conditioned_unet, UNet2DConditionModel)
