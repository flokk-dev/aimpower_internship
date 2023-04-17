"""
Creator: Flokk
Date: 09/04/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: data saving
from diffusers import DDPMPipeline, StableDiffusionPipeline

# IMPORT: data processing
from transformers import CLIPFeatureExtractor
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

# IMPORT: project
from .trainer import Trainer
from .learner import BasicLearner, GuidedLearner, BasicStableLearner, ConditionedStableLearner


class BasicTrainer(Trainer):
    """
    Represents a BasicTrainer.

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
    _LEARNERS = {"basic": BasicLearner, "guided": GuidedLearner}

    def __init__(
            self,
            params: Dict[str, Any]
    ):
        """
        Instantiates a BasicTrainer.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
        """
        # Mother class
        super(BasicTrainer, self).__init__(params)

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
        learning_type = params["learner"]["learning_type"]
        model_type = params["learner"]["components"]["model"]["type"]

        # Loading
        if loading_type not in ["basic", "label"]:
            raise ValueError(f"{loading_type} loading isn't handled by the BasicTrainer.")

        # Learner
        if learning_type not in ["basic", "guided"]:
            raise ValueError(f"{learning_type} learning isn't handled by the BasicTrainer.")

        # Model
        if not learning_type == model_type:
            raise ValueError(
                f"{model_type} unet isn't compatible with {learning_type} learning "
                f"when using the BasicTrainer."
            )

    def _save(
            self,
            path: str
    ):
        """
        Parameters
        ----------
            path : str
                training's saving path
        """
        DDPMPipeline(
            unet=self._learner.components.model,
            scheduler=self._learner.components.noise_scheduler
        ).save_pretrained(path)


class StableTrainer(Trainer):
    """
    Represents a StableTrainer.

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
    _LEARNERS = {"basic": BasicStableLearner, "conditioned": ConditionedStableLearner}

    def __init__(
            self,
            params: Dict[str, Any]
    ):
        """
        Instantiates a StableTrainer.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
        """
        # Mother class
        super(StableTrainer, self).__init__(params)

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
        learning_type = params["learner"]["learning_type"]
        model_type = params["learner"]["components"]["model"]["type"]

        # Loading
        if loading_type not in ["basic", "prompt"]:
            raise ValueError(f"{loading_type} loading isn't handled by the StableTrainer.")

        # Learner
        if learning_type not in ["basic", "conditioned"]:
            raise ValueError(f"{learning_type} learning isn't handled by the StableTrainer.")

        # Model
        if not model_type == "conditioned":
            raise ValueError(f"{model_type} unet isn't handled by the StableTrainer.")

    def _save(
            self,
            path: str
    ):
        """
        Parameters
        ----------
            path : str
                training's saving path

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        StableDiffusionPipeline(
            text_encoder=self._learner.components.text_encoder,
            vae=self._learner.components.vae,
            unet=self._learner.components.model,
            tokenizer=self._data_loader.tokenizer,
            scheduler=self._learner.components.noise_scheduler,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ),
            feature_extractor=CLIPFeatureExtractor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
        )
