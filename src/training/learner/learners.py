"""
Creator: Flokk
Date: 09/04/2023
Version: 1.0
Purpose:
"""

# IMPORT: utils
from typing import *
from tqdm import tqdm

# IMPORT: deep learning
import torch

# IMPORT: project
import utils

from .learner import Learner
from .components import Components, AdvancedComponents


class BasicLearner(Learner):
    """
    Represents a BasicLearner.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _loss : Loss
            training's loss function
        _components : Components
            training's components

    Methods
    ----------
        _learn
            Learns on a batch of data
        _forward
            Extracts noise within the noisy image using the noise_scheduler
        _add_noise
            Adds noise to a given tensor
        inference
            Generates and image using the training's noise_scheduler
    """
    def __init__(
            self,
            params: Dict[str, Any],
            num_epochs: int,
            num_batches: int
    ):
        """
        Instantiates a BasicLearner.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            num_epochs : int
                number of epochs during the training
            num_batches : int
                number of batches within the data loader
        """
        # Mother class
        super(BasicLearner, self).__init__(params)

        # Components
        self._components = Components(params["components"], num_epochs, num_batches)

    def _forward(
            self,
            batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts noise within the noisy image using the noise_scheduler.

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
        # Inputs
        batch["image"]: torch.Tensor = batch["image"].type(torch.float32).to(self._DEVICE)
        noisy_image, noise, timestep = self._add_noise(batch["image"])

        # If guided
        if self._params["learning_type"] == "guided":
            batch["label"] = batch["label"].type(torch.int32).to(self._DEVICE)
            return noise, self._components.model(noisy_image, timestep, batch["label"]).sample

        # Else
        return noise, self._components.model(noisy_image, timestep).sample

    def inference(
            self,
    ) -> Dict[str, torch.Tensor]:
        """
        Generates and image using the training's pipeline.

        Returns
        ----------
            Dict[str, torch.Tensor]
                generated image
        """
        if self._params["learning_type"] == "guided":
            mul = self._params["learner"]["components"]["model"]["args"]["num_labels"]
        else:
            mul = 1

        # Samples gaussian noise
        num_samples = 5

        image: torch.Tensor = torch.randn(
            (
                num_samples * mul, self._params["in_channels"],
                self._params["img_size"], self._params["img_size"]
            ),
            generator=torch.manual_seed(0)
        ).to(self._DEVICE)

        if self._params["learning_type"] == "guided":
            labels: torch.Tensor = torch.tensor(
                [[i] * num_samples for i in range(mul)]
            ).flatten().to(self._DEVICE)

        # Generates an image based on the gaussian noise
        for timestep in tqdm(self._components.noise_scheduler.timesteps):
            # Predicts the residual noise
            with torch.no_grad():
                if self._params["learning_type"] == "guided":
                    residual: torch.Tensor = self._components.model(image, timestep, labels).sample
                else:
                    residual: torch.Tensor = self._components.model(image, timestep).sample

            # De-noises using the prediction
            image: torch.Tensor = self._components.noise_scheduler.step(
                residual, timestep, image
            ).prev_sample

        image = utils.adjust_image_colors(image.cpu())

        # Returns
        return {"image": image}


class ConditionedLearner(Learner):
    """
    Represents a ConditionedLearner.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _loss : Loss
            training's loss function
        _components : Components
            training's components

    Methods
    ----------
        _learn
            Learns on a batch of data
        _forward
            Extracts noise within the noisy image using the noise_scheduler
        _add_noise
            Adds noise to a given tensor
        inference
            Generates and image using the training's noise_scheduler
    """
    def __init__(
            self,
            params: Dict[str, Any],
            num_epochs: int,
            num_batches: int
    ):
        """
        Instantiates a ConditionedLearner.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            num_epochs : int
                number of epochs during the training
            num_batches : int
                number of batches within the data loader
        """
        # Mother class
        super(ConditionedLearner, self).__init__(params)

        # Components
        self._components = AdvancedComponents(params["components"], num_epochs, num_batches)

    def _forward(
            self,
            batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts noise within the noisy image using the noise_scheduler.

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
        # Puts data on desired device
        batch["image"] = batch["image"].type(torch.float32).to(self._DEVICE)
        batch["prompt"] = batch["prompt"].type(torch.float32).to(self._DEVICE)

        # Predicts added noise
        noisy_image, noise, timestep = self._add_noise(batch["image"])
        return noise, self._components.model(noisy_image, timestep, batch["prompt"]).sample

    def inference(
            self,
            to_dict: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Generates and image using the training's pipeline.

        Parameters
        ----------
            to_dict : bool
                wether or not to return a dictionary

        Returns
        ----------
            Dict[str, torch.Tensor]
                generated image
        """
        pass
