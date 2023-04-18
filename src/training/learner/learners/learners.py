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

from src.training.learner.learner import Learner
from src.training.learner.components import Components


class BasicLearner(Learner):
    """
    Represents a BasicLearner.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _loss : Loss
            training's loss function
        components : Components
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
        self.components = Components(params["components"], num_epochs, num_batches)

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
        # Image
        if not self._params["reduce_dimensions"]:
            batch["image"]: torch.Tensor = batch["image"].type(torch.float32).to(self._DEVICE)

        # Predicts added noise
        noisy_image, noise, timestep = self._add_noise(batch["image"])
        return noise, self.components.model(noisy_image, timestep).sample

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
        # Samples gaussian noise
        image: torch.Tensor = torch.randn(
            (
                10,
                self._params["components"]["model"]["args"]["in_channels"],
                self._params["components"]["model"]["args"]["sample_size"],
                self._params["components"]["model"]["args"]["sample_size"]
            ),
            generator=torch.manual_seed(0)
        ).to(self._DEVICE)

        # Generates an image based on the gaussian noise
        for timestep in tqdm(self.components.noise_scheduler.timesteps):
            # Predicts the residual noise
            with torch.no_grad():
                residual: torch.Tensor = self.components.model(image, timestep).sample

            # De-noises using the prediction
            image: torch.Tensor = self.components.noise_scheduler.step(
                residual, timestep, image
            ).prev_sample

        if self._params["reduce_dimensions"]:
            image = self._decode_image(image)

        print(image.shape)
        print(torch.unique(image))

        image = utils.adjust_image_colors(image.cpu())

        # Returns
        return {"image": image}


class GuidedLearner(Learner):
    """
    Represents a GuidedLearner.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _loss : Loss
            training's loss function
        components : Components
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
        Instantiates a GuidedLearner.

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
        super(GuidedLearner, self).__init__(params)

        # Components
        self.components = Components(params["components"], num_epochs, num_batches)

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
        # Image
        if not self._params["reduce_dimensions"]:
            batch["image"]: torch.Tensor = batch["image"].type(torch.float32).to(self._DEVICE)

        # Label
        batch["label"] = batch["label"].type(torch.int32).to(self._DEVICE)

        # Predicts added noise
        noisy_image, noise, timestep = self._add_noise(batch["image"])
        return noise, self.components.model(noisy_image, timestep, batch["label"]).sample

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
        # Samples gaussian noise
        image: torch.Tensor = torch.randn(
            (
                5 * self._params["components"]["model"]["args"]["num_labels"],
                self._params["components"]["model"]["args"]["in_channels"],
                self._params["components"]["model"]["args"]["sample_size"],
                self._params["components"]["model"]["args"]["sample_size"]
            ),
            generator=torch.manual_seed(0)
        ).to(self._DEVICE)

        labels: torch.Tensor = torch.tensor(
            [[i] * 5 for i in range(self._params["components"]["model"]["args"]["num_labels"])]
        ).flatten().to(self._DEVICE)

        # Generates an image based on the gaussian noise
        for timestep in tqdm(self.components.noise_scheduler.timesteps):
            # Predicts the residual noise
            with torch.no_grad():
                residual: torch.Tensor = self.components.model(image, timestep, labels).sample

            # De-noises using the prediction
            image: torch.Tensor = self.components.noise_scheduler.step(
                residual, timestep, image
            ).prev_sample

        image = utils.adjust_image_colors(image.cpu())

        # Returns
        return {
            str(class_idx): image[class_idx * 5:(class_idx + 1) * 5]
            for class_idx
            in range(image.shape[0] // 5)
        }
