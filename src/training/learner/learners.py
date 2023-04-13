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
import torchvision

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
        lr_scheduler : torch.nn.Module
            optimizer's scheduler

    Methods
    ----------
        _learn
            Learns on a batch of data
        _forward
            Extracts noise within the noisy image using the pipeline
        _add_noise
            Adds noise to a given tensor
        inference
            Generates and image using the training's pipeline
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
        image: torch.Tensor = batch[0].type(torch.float16).to(self._DEVICE)

        # Predicts added noise
        noisy_image, noise, timestep = self._add_noise(image)
        return noise, self.pipeline.unet(noisy_image, timestep).sample

    def inference(
            self,
            to_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generates and image using the training's pipeline.

        Parameters
        ----------
            to_dict : bool
                wether or not to return a dictionary

        Returns
        ----------
            torch.Tensor
                generated image
        """
        # Samples gaussian noise
        image: torch.Tensor = torch.randn(
            (
                10, self._params["num_channels"],
                self._params["img_size"], self._params["img_size"]
            ),
            generator=torch.manual_seed(0)
        ).to(self._DEVICE)

        # Generates an image based on the gaussian noise
        for timestep in tqdm(self.pipeline.scheduler.timesteps):
            # Predicts the residual noise
            with torch.no_grad():
                residual: torch.Tensor = self.pipeline.unet(image, timestep).sample

            # De-noises using the prediction
            image: torch.Tensor = self.pipeline.scheduler.step(
                residual, timestep, image
            ).prev_sample

        image = utils.adjust_image_colors(image.cpu())

        # Returns
        if to_dict:
            return {"image": image}
        return image


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
        lr_scheduler : torch.nn.Module
            optimizer's scheduler

    Methods
    ----------
        _learn
            Learns on a batch of data
        _forward
            Extracts noise within the noisy image using the pipeline
        _add_noise
            Adds noise to a given tensor
        inference
            Generates and image using the training's pipeline
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
            batch: Tuple[torch.Tensor,  List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts noise within the noisy image using the pipeline.

        Parameters
        ----------
            batch : Tuple[torch.Tensor, List[str]]
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
        image_classes: torch.Tensor = utils.as_tensor(batch[1]).type(torch.int32).to(self._DEVICE)

        # Predicts added noise
        noisy_image, noise, timestep = self._add_noise(image)
        return noise, self.pipeline.unet(noisy_image, timestep, image_classes).sample

    def inference(
            self,
            to_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generates and image using the training's pipeline.

        Parameters
        ----------
            to_dict : bool
                wether or not to return a dictionary

        Returns
        ----------
            torch.Tensor
                generated image
        """
        num_samples: int = 5

        # Samples gaussian noise
        image: torch.Tensor = torch.randn(
            (
                num_samples * self._params["num_classes"], self._params["num_channels"],
                self._params["img_size"], self._params["img_size"]
            ),
            generator=torch.manual_seed(0)
        ).to(self._DEVICE)

        # Generates classes to guide the generation
        image_classes: torch.Tensor = torch.tensor(
            [[i] * num_samples for i in range(10)]
        ).flatten().to(self._DEVICE)

        # Generates an image based on the gaussian noise
        for timestep in tqdm(self.pipeline.scheduler.timesteps):
            # Predicts the residual noise
            with torch.no_grad():
                residual: torch.Tensor = self.pipeline.unet(image, timestep, image_classes).sample

            # De-noises using the prediction
            image: torch.Tensor = self.pipeline.scheduler.step(
                residual, timestep, image
            ).prev_sample

        image = utils.adjust_image_colors(image.cpu())

        # Returns
        if to_dict:
            return {
                str(class_idx): image[class_idx*num_samples:(class_idx+1)*num_samples]
                for class_idx
                in range(image.shape[0] // num_samples)
            }
        return image
