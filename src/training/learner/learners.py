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
from .components import ClassicComponents, ReinforcementComponents

from src.training.learner.components.ranker import PickAPicScore


class ClassicLearner(Learner):
    """
    Represents a ClassicLearner.

    Attributes
    ----------
        _config : Dict[str, Any]
            configuration needed to adjust the program behaviour
        components : DiffusionComponents
            training's components
        _loss : torch.nn.Module
            loss function

    Methods
    ----------
        learn
            Learns on a batch of data
        _forward
            Extracts noise within the noisy image using the noise_scheduler
        _add_noise
            Adds noise to a given tensor
    """
    def __init__(
            self,
            config: Dict[str, Any],
            repo_path: str,
            dataset_path: str
    ):
        """
        Instantiates a ClassicLearner.

        Parameters
        ----------
            config : Dict[str, Any]
                configuration needed to adjust the program behaviour
            repo_path: str
                path where to save the pipeline
            dataset_path : str
                path to the dataset
        """
        # ----- Mother Class ----- #
        super(ClassicLearner, self).__init__(config)

        # ----- Attributes ----- #
        # Components
        self.components: ClassicComponents = ClassicComponents(
            config, repo_path, dataset_path
        )
        self.components.prepare()

        # Loss
        self._loss: torch.nn.Module = torch.nn.MSELoss().to(
            self.components.accelerator.device,
            dtype=torch.float16
        )

    def _learn(
            self,
            batch: Dict[str, torch.Tensor],
    ) -> float:
        """
        Learns on a batch of data.

        Parameters
        ----------
            batch : Dict[str, torch.Tensor]
                batch of data

        Returns
        ----------
            float
                loss value computed using batch's data
        """
        with self.components.accelerator.accumulate(self.components.model):
            # Forward
            noise, noise_pred = self._forward(batch)

            # Loss backward
            loss_value: torch.Tensor = self._loss(noise_pred, noise)
            self.components.accelerator.backward(loss_value)

            # Update the training components
            self.components.optimizer.step()
            self.components.lr_scheduler.step()
            self.components.optimizer.zero_grad()

        # Returns
        return loss_value.detach().item()

    def _forward(
            self,
            batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts noise within the noisy image using the model.

        Parameters
        ----------
            batch : Dict[str, torch.Tensor]
                batch of data

        Returns
        ----------
            torch.Tensor
                added noise
            torch.Tensor
                extracted noise
        """
        super()._forward(batch)

        # Adds noise
        noisy_image, noise, timestep = self._add_noise(batch["image"])

        # Predicts added noise
        return noise, self.components.model(
            noisy_image, timestep, batch["prompt"]
        ).sample


class ReinforcementLearner(Learner):
    """
    Represents a ReinforcementLearner.

    Attributes
    ----------
        _config : Dict[str, Any]
            configuration needed to adjust the program behaviour
        components : DiffusionComponents
            training's components
        _reward : torch.nn.Module
            reward function

    Methods
    ----------
        learn
            Learns on a batch of data
        _forward
            Extracts noise within the noisy image using the noise_scheduler
        _add_noise
            Adds noise to a given tensor
    """
    def __init__(
            self,
            config: Dict[str, Any],
            repo_path,
            dataset_path: str
    ):
        """
        Instantiates a ReinforcementLearner.

        Parameters
        ----------
            config : Dict[str, Any]
                configuration needed to adjust the program behaviour
            repo_path: str
                path where to save the pipeline
            dataset_path : str
                path to the dataset
        """
        # ----- Mother Class ----- #
        super(ReinforcementLearner, self).__init__(config)

        # ----- Attributes ----- #
        # Components
        self.components: ReinforcementComponents = ReinforcementComponents(
            config, repo_path, dataset_path
        )
        self.components.prepare()

        # Reward
        self._reward: PickAPicScore = PickAPicScore(device=self.components.accelerator.device)

    def _learn(
            self,
            batch: Dict[str, torch.Tensor],
    ):
        """
        Learns on a batch of data.

        Parameters
        ----------
            batch : Dict[str, torch.Tensor]
                batch of data

        Returns
        ----------
            float
                loss value computed using batch's data
        """
        # Not implemented
        raise NotImplementedError()

    def _forward(
            self,
            batch: Dict[str, torch.Tensor],
    ):
        """
        Extracts noise within the noisy image using the model.

        Parameters
        ----------
            batch : Dict[str, torch.Tensor]
                batch of data

        Returns
        ----------
            torch.Tensor
                added noise
            torch.Tensor
                extracted noise
        """
        super()._forward(batch)

        # Not implemented
        raise NotImplementedError()
