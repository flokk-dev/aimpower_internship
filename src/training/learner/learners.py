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


class BasicLearner(Learner):
    """
    Represents a BasicLearner.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        loss : Loss
            training's loss function
        pipeline : diffusers.DiffusionPipeline
            diffusion pipeline
        optimizer : torch.optim.Optimizer
            pipeline's optimizer
        scheduler : torch.nn.Module
            optimizer's scheduler

    Methods
    ----------
        _forward
            Launches the training
        _add_noise
            Runs an epoch
    """
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(
            self,
            params: Dict[str, Any],
            num_batches: int,
            weights_path: str
    ):
        """
        Instantiates a BasicLearner.

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
        super(BasicLearner, self).__init__(params, num_batches, weights_path)

    def _forward(
            self,
            batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts noise within the noisy image using the pipeline.

        Parameters
        ----------
            batch : torch.Tensor
                batch of data

        Returns
        ----------
            torch.Tensor
                added noise
            torch.Tensor
                extracted noise
        """
        # Puts data on desired device
        image: torch.Tensor = batch[0].type(torch.float32).to(self._DEVICE)

        # Predicts added noise
        noisy_image, noise, timesteps = self._add_noise(image)
        return noise, self.pipeline.unet(noisy_image, timesteps).sample

    def inference(
            self,
    ) -> torch.Tensor:
        """
        Extracts noise within the noisy image using the pipeline.

        Returns
        ----------
            torch.Tensor
                generated image
        """
        # Generates random samples
        images = torch.randn(
            10, self.pipeline.unet.in_channels,
            self.pipeline.unet.sample_size, self.pipeline.unet.sample_size,
            generator=torch.manual_seed(0)
        ).to(self._DEVICE)

        # Sampling loop
        for timestep in tqdm(self.pipeline.scheduler.timesteps):
            # Get model pred
            with torch.no_grad():
                residual = self.pipeline.unet(images, timestep).sample

            # Update sample with step
            images = self.pipeline.scheduler.step(
                residual, timestep, images,
                generator=torch.manual_seed(0)
            ).prev_sample

        return images


class GuidedLearner(Learner):
    """
    Represents a GuidedLearner.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        loss : Loss
            training's loss function
        pipeline : diffusers.DiffusionPipeline
            diffusion pipeline
        optimizer : torch.optim.Optimizer
            pipeline's optimizer
        scheduler : torch.nn.Module
            optimizer's scheduler

    Methods
    ----------
        _forward
            Launches the training
        _add_noise
            Runs an epoch
    """
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(
            self,
            params: Dict[str, Any],
            num_batches: int,
            weights_path: str
    ):
        """
        Instantiates a GuidedLearner.

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

    def _forward(
            self,
            batch: Tuple[torch.Tensor, str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts noise within the noisy image using the pipeline.

        Parameters
        ----------
            batch : Tuple[torch.Tensor, str]
                batch of data

        Returns
        ----------
            torch.Tensor
                added noise
            torch.Tensor
                extracted noise
        """
        # Puts data on desired device
        image: torch.Tensor = batch[0].type(torch.float32).to(self._DEVICE)
        image_classes = utils.str_to_tensor(batch[1]).type(torch.int32).to(self._DEVICE)

        # Predicts added noise
        noisy_image, noise, timesteps = self._add_noise(image)
        return noise, self.pipeline.unet(noisy_image, timesteps, image_classes).sample

    def inference(
            self,
    ) -> torch.Tensor:
        """
        Extracts noise within the noisy image using the pipeline.

        Returns
        ----------
            torch.Tensor
                generated image
        """
        num_samples = 3

        # Generates random samples
        images = torch.randn(
            num_samples * self._params["num_classes"], self.pipeline.unet.in_channels,
            self.pipeline.unet.sample_size, self.pipeline.unet.sample_size,
            generator=torch.manual_seed(0)
        ).to(self._DEVICE)

        # Generates classes to guide the generation
        images_classes = torch.tensor(
            [[i] * num_samples for i in range(10)]
        ).flatten().to(self._DEVICE)

        # Sampling loop
        for timestep in tqdm(self.pipeline.scheduler.timesteps):
            # Get model pred
            with torch.no_grad():
                residual = self.pipeline.unet(images, timestep, images_classes).sample

            # Update sample with step
            images = self.pipeline.scheduler.step(
                residual, timestep, images,
                generator=torch.manual_seed(0)
            ).prev_sample.sample

        return images
