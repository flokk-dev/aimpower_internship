"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import diffusers

# IMPORT: project
from .pipelines import \
    load_ddpm, init_ddpm, \
    load_ddim, init_ddim, \
    load_stable_diffusion, init_stable_diffusion

from src.training.learner.model import ModelManager


class PipelineManager(dict):
    """
    Represents a pipeline manager.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _model_manager : ModelManager
            the model manager
    """

    def __init__(
            self,
            params: Dict[str, Any]
    ):
        """
        Instantiates a PipelineManager.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
        """
        # Mother class
        super(PipelineManager, self).__init__({
            "ddpm":
                lambda arg: load_ddpm(arg)
                if isinstance(arg, str)
                else init_ddpm(arg),
            "ddim":
                lambda arg:
                load_ddim(arg) if isinstance(arg, str)
                else init_ddim(arg),
            "stable_diffusion":
                lambda arg:
                load_stable_diffusion(arg) if isinstance(arg, str)
                else init_stable_diffusion(),
        })

        # Attributes
        self._params: Dict[str, Any] = params
        self._model_manager: ModelManager = ModelManager(params)

    def __call__(
            self,
            pipeline_id: str,
            weights_path: str,
            model_id: str
    ) -> diffusers.DiffusionPipeline:
        """
        Parameters
        ----------
            pipeline_id : str
                id of the pipeline
            weights_path : str
                path to the pipeline's weights
            model_id : str
                id of the model to use

        Returns
        ----------
            diffusers.DiffusionPipeline
                pipeline associated to the pipeline id
        """
        try:
            if weights_path is not None:
                return self[pipeline_id](weights_path)
            else:
                return self[pipeline_id](self._model_manager(model_id))
        except KeyError:
            raise KeyError(f"The {pipeline_id} isn't handled by the pipeline manager.")
