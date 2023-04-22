"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import os
from diffusers import DDIMScheduler, DDPMScheduler

# IMPORT: test
import pytest

# IMPORT: project
import paths

from src.training.pipeline.components.noise_scheduler import NoiseSchedulerManager
from src.training.pipeline.components.noise_scheduler.noise_schedulers import \
    init_ddpm, load_ddpm, \
    init_ddim, load_ddim


# -------------------- CONSTANT -------------------- #

DDPM_PATH = os.path.join(paths.TEST_RESOURCES_PATH, "pipelines", "1")
DDIM_PATH = os.path.join(paths.TEST_RESOURCES_PATH, "pipelines", "2")


# -------------------- FIXTURES -------------------- #

@pytest.fixture(scope="function")
def noise_scheduler_manager():
    return NoiseSchedulerManager()


@pytest.fixture(scope="function")
def params():
    return {
        "type": "ddpm",
        "pipeline_path": "",

        "args": {
            "num_timesteps": 1000
        }
    }


# -------------------- DDPM -------------------- #

def test_init_ddpm(noise_scheduler_manager, params):
    ddpm = init_ddpm(params=params["args"])
    assert isinstance(ddpm, DDPMScheduler)

    ddpm = noise_scheduler_manager.__call__(
        params["type"],
        params["args"],
        pipeline_path=params["pipeline_path"]
    )
    assert isinstance(ddpm, DDPMScheduler)


def test_load_ddpm(noise_scheduler_manager, params):
    ddpm = load_ddpm(pipeline_path=DDPM_PATH)
    assert isinstance(ddpm, DDPMScheduler)

    ddpm = noise_scheduler_manager.__call__(
        params["type"],
        params["args"],
        pipeline_path=DDPM_PATH
    )
    assert isinstance(ddpm, DDPMScheduler)


# -------------------- Conditioned U-Net -------------------- #

def test_init_ddim(noise_scheduler_manager, params):
    params["type"] = "ddim"

    ddim = init_ddim(params=params["args"])
    assert isinstance(ddim, DDIMScheduler)

    ddim = noise_scheduler_manager.__call__(
        params["type"],
        params["args"],
        pipeline_path=params["pipeline_path"]
    )
    assert isinstance(ddim, DDIMScheduler)


def test_load_ddim(noise_scheduler_manager, params):
    params["type"] = "ddim"

    ddim = load_ddim(pipeline_path=DDIM_PATH)
    assert isinstance(ddim, DDIMScheduler)

    ddim = noise_scheduler_manager.__call__(
        params["type"],
        params["args"],
        pipeline_path=DDIM_PATH
    )
    assert isinstance(ddim, DDIMScheduler)
