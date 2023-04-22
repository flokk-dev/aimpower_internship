"""
Creator: Flokk
Date: 09/04/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: project
from .trainer import Trainer
from .pipeline import DiffusionPipeline, GDiffusionPipeline


class TrainerV1(Trainer):
    """
    Represents a TrainerV1.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _data_loader : Dict[str: DataLoader]
            loader containing training's data

    Methods
    ----------
        _launch
            Launches the training
        _run_epoch
            Runs an epoch
        _checkpoint
            Saves noise_scheduler's weights
    """
    _PIPELINES = {"basic": DiffusionPipeline, "guided": GDiffusionPipeline}

    def __init__(
            self,
            params: Dict[str, Any]
    ):
        """
        Instantiates a TrainerV1.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
        """
        # Mother class
        super(TrainerV1, self).__init__(params)

    def _verify_parameters(
            self,
            params: Dict[str, Any]
    ):
        """
        Verifies if the training's configuration is correct.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour

        Raises
        ----------
            ValueError
                if the configuration isn't conform
        """
        loading_type = params["loader"]["loading_type"]
        pipeline_type = params["pipeline"]["pipeline_type"]
        model_type = params["pipeline"]["components"]["model"]["model_type"]

        # Loading
        if loading_type not in ["basic", "label"]:
            raise ValueError(f"{loading_type} loading isn't handled by the TrainerV1.")

        # Learner
        if pipeline_type not in ["basic", "guided"]:
            raise ValueError(f"{pipeline_type} training isn't handled by the TrainerV1.")

        # Model
        if not pipeline_type == model_type:
            raise ValueError(
                f"{model_type} unet isn't compatible with {pipeline_type} pipeline "
                f"when using the TrainerV1."
            )
