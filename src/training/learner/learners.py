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
from .components import Components, ConditionedComponents


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
        super(BasicLearner, self).__init__(params)

        # Components
        self._components = Components(params, weights_path, num_batches)

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
        noisy_image, noise, timestep = self._add_noise(image)
        return noise, self._components.model(noisy_image, timestep).sample

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
        for timestep in tqdm(self._components.noise_scheduler.timesteps):
            # Predicts the residual noise
            with torch.no_grad():
                residual: torch.Tensor = self._components.model(image, timestep).sample

            # De-noises using the prediction
            image: torch.Tensor = self._components.noise_scheduler.step(
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
        super(GuidedLearner, self).__init__(params)

        # Components
        self._components = Components(params, weights_path, num_batches)

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
        return noise, self._components.model(noisy_image, timestep, image_classes).sample

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
        for timestep in tqdm(self._components.noise_scheduler.timesteps):
            # Predicts the residual noise
            with torch.no_grad():
                residual: torch.Tensor = self._components.model(
                    image, timestep, image_classes
                ).sample

            # De-noises using the prediction
            image: torch.Tensor = self._components.noise_scheduler.step(
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
        super(ConditionedLearner, self).__init__(params)

        # Components
        self._components = ConditionedComponents(params, weights_path, num_batches)

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
        pass

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
        pass
