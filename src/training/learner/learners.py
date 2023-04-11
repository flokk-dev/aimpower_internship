"""
Creator: Flokk
Date: 09/04/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: deep learning
import torch

# IMPORT: project
from .learner import Learner


class GuidedLearner(Learner):
    """
    Represents a GuidedLearner.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _loss : Loss
            training's loss function
        _pipeline : Union[DDPMPipeline, DDIMPipeline]
            diffusion pipeline
        _optimizer : torch.optim.Optimizer
            pipeline's optimizer
        _scheduler : torch.nn.Module
            optimizer's scheduler

    Methods
    ----------
        _init_pipeline
            Initializes the training's pipeline
        _launch
            Launches the training
        _run_epoch
            Runs an epoch
        _learn_on_batch
            Learns using data within a batch
    """
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(
            self,
            params: Dict[str, Any],
            num_batches: int,
            weights_path: str
    ):
        """
        Instantiates a Learner.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            num_batches : int
                number of batches within the data loader
            weights_path : str
                path to the pipeline's weights
        """
        # Mother class
        super(GuidedLearner, self).__init__(params, num_batches, weights_path)

    def _learn(
            self,
            batch: torch.Tensor,
            learn: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Learns using data within a batch.

        Parameters
        ----------
            batch : torch.Tensor
                batch of tensors
            learn : bool
                boolean indicating whether to train

        Returns
        ----------
            torch.Tensor
                loss calculated using batch's data
        """
        pass

    def __call__(
            self,
            batch: torch.Tensor,
            learn: bool = True
    ):
        """
        Parameters
        ----------
            batch : torch.Tensor
                batch of tensors
            learn : bool
                boolean indicating whether to train

        Returns
        ----------
            torch.Tensor
                loss calculated using batch's data
        """
        return self._learn(batch, learn)
