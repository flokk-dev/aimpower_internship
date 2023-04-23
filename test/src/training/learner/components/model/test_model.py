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
import torch

# IMPORT: project
import paths

from src.training.learner.components.model import ModelManager
from src.training.learner.components.model.models import \
    init_unet, load_unet, \
    init_guided_unet, load_guided_unet, \
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
        "type": "basic",
        "pipeline_path": "",

        "args": {
            "sample_size": 64,
            "in_channels": 3,
            "out_channels": 3,
            "block_out_channels": [64, 64, 128, 128],
            "num_labels": 10
        }
    }


@pytest.fixture(scope="function")
def args():
    noisy_tensor = torch.randn(1, 3, 64, 64)
    timestep = torch.randint(0, 1000, (noisy_tensor.shape[0], ))

    return noisy_tensor, timestep


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


def test_forward_unet(params, args):
    # INIT
    unet = init_unet(params=params["args"])

    output_tensor = unet(*args).sample
    assert output_tensor.shape == torch.Size((1, 3, 64, 64))

    # LOAD
    unet = load_unet(pipeline_path=UNET_PATH)

    output_tensor = unet(*args).sample
    assert output_tensor.shape == torch.Size((1, 3, 64, 64))


# -------------------- U-Net -------------------- #

def test_init_guided_unet(model_manager, params):
    params["type"] = "guided"

    guided_unet = init_guided_unet(params=params["args"])
    assert isinstance(guided_unet, UNet2DModel)

    guided_unet = model_manager.__call__(
        params["type"],
        params["args"],
        pipeline_path=params["pipeline_path"]
    )
    assert isinstance(guided_unet, UNet2DModel)


def test_load_guided_unet(model_manager, params):
    params["type"] = "guided"

    guided_unet = load_guided_unet(pipeline_path=UNET_PATH)
    assert isinstance(guided_unet, UNet2DModel)

    guided_unet = model_manager.__call__(
        params["type"],
        params["args"],
        pipeline_path=UNET_PATH
    )
    assert isinstance(guided_unet, UNet2DModel)


def test_forward_guided_unet(params, args):
    labels = torch.Tensor([0]).type(torch.int32)

    # INIT
    guided_unet = init_guided_unet(params=params["args"])

    output_tensor = guided_unet(*args, labels).sample
    assert output_tensor.shape == torch.Size((1, 3, 64, 64))

    # LOAD
    guided_unet = load_guided_unet(pipeline_path=UNET_PATH)

    output_tensor = guided_unet(*args, labels).sample
    assert output_tensor.shape == torch.Size((1, 3, 64, 64))


# -------------------- Conditioned U-Net -------------------- #

def test_init_conditioned_unet(model_manager, params):
    params["type"] = "conditioned"

    conditioned_unet = init_conditioned_unet(params=params["args"])
    assert isinstance(conditioned_unet, UNet2DConditionModel)

    conditioned_unet = model_manager.__call__(
        params["type"],
        params["args"],
        pipeline_path=params["pipeline_path"]
    )
    assert isinstance(conditioned_unet, UNet2DConditionModel)


def test_load_conditioned_unet(model_manager, params):
    params["type"] = "conditioned"

    conditioned_unet = load_conditioned_unet(pipeline_path=CONDITIONED_UNET_PATH)
    assert isinstance(conditioned_unet, UNet2DConditionModel)

    conditioned_unet = model_manager.__call__(
        params["type"],
        params["args"],
        pipeline_path=CONDITIONED_UNET_PATH
    )
    assert isinstance(conditioned_unet, UNet2DConditionModel)


def test_forward_conditioned_unet(params, args):
    input_tensor = torch.randn(1, 3, 64, 64)
    encoder_hidden_states = torch.randn(1, 3, 1280)

    # INIT
    conditioned_unet = init_conditioned_unet(params=params["args"])

    output_tensor = conditioned_unet(input_tensor, args[1], encoder_hidden_states).sample
    assert output_tensor.shape == torch.Size((1, 3, 64, 64))

    # LOAD
    conditioned_unet = load_conditioned_unet(pipeline_path=CONDITIONED_UNET_PATH)

    output_tensor = conditioned_unet(*args, encoder_hidden_states).sample
    assert output_tensor.shape == torch.Size((1, 3, 64, 64))
