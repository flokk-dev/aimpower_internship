"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import os

"""
ROOT
"""
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

"""
RESOURCES
"""
RESOURCES_PATH = os.path.join(ROOT_PATH, "resources")
CONFIG_PATH = os.path.join(RESOURCES_PATH, "config.json")

"""
TEST
"""
TEST_PATH = os.path.join(ROOT_PATH, "test")
TEST_DATA_PATH = os.path.join(TEST_PATH, "data")

"""
MODELS
"""
MODELS_PATH = os.path.join(RESOURCES_PATH, "noise_schedulers")

TRAIN_PATH = os.path.join(MODELS_PATH, "training_results.csv")
INFERENCE_PATH = os.path.join(MODELS_PATH, "inference_results.csv")
