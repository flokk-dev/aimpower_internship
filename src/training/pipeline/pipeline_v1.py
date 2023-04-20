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
import diffusers

# IMPORT: project
import utils

from .pipeline import Pipeline
from .components import ComponentsV1


class PipelineV1(Pipeline):
    """
    Represents an PipelineV1, which will be modified depending on the use case.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _loss : Loss
            training's loss function
        components : ComponentsV1
            training's components

    Methods
    ----------
        _init_components
            Initializes the pipeline's components
        learn
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
        Instantiates a LearnerV1.

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
        super(PipelineV1, self).__init__(params, num_epochs, num_batches)

    def _init_components(
            self,
            params: Dict[str, Any],
            num_epochs: int,
            num_batches: int
    ) -> ComponentsV1:
        """
        Initializes the pipeline's components.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            num_epochs : int
                number of epochs during the training
            num_batches : int
                number of batches within the data loader

        Returns
        ----------
            ComponentsV1
                pipeline's components
        """
        return ComponentsV1(params["components"], num_epochs, num_batches)

    def __call__(self) -> diffusers.DDPMPipeline:
        """
        Returns
        ----------
            diffusers.DDPMPipeline
                diffusion pipeline
        """
        pipeline = utils.BasicDiffusionPipeline(
            unet=self.components.model,
            scheduler=self.components.noise_scheduler
        ).to(self._DEVICE)

        pipeline.safety_checker = None

        # Returns
        return pipeline


class DiffusionPipeline(PipelineV1):
    """
    Represents a DiffusionPipeline.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _loss : Loss
            training's loss function
        components : ComponentsV1
            training's components

    Methods
    ----------
        _init_components
            Initializes the pipeline's components
        learn
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
        Instantiates a DiffusionPipeline.

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
        super(DiffusionPipeline, self).__init__(params, num_epochs, num_batches)

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
        batch["image"]: torch.Tensor = batch["image"].to(self._DEVICE)

        # Predicts added noise
        noisy_image, noise, timestep = self._add_noise(batch["image"])
        return noise, self.components.model(noisy_image, timestep).sample

    def inference(
            self,
    ) -> Dict[str, torch.Tensor]:
        """
        Generates and image using the training's components.

        Returns
        ----------
            Dict[str, torch.Tensor]
                generated image
        """
        pipeline = self()

        # Generates images
        generated_images = pipeline(
                batch_size=10,
                num_inference_steps=1000,
                generator=torch.manual_seed(0)
        ).images

        # Adjusts colors
        images: List[torch.Tensor] = list()
        for image in generated_images:
            images.append(
                utils.adjust_image_colors(
                    utils.to_tensor(image)
                )
            )

        # Returns
        return {"image": torch.stack(images, dim=0)}


class GDiffusionPipeline(PipelineV1):
    """
    Represents a GDiffusionPipeline.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        _loss : Loss
            training's loss function
        components : ComponentsV1
            training's components

    Methods
    ----------
        _init_components
            Initializes the pipeline's components
        learn
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
        Instantiates a GDiffusionPipeline.

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
        super(GDiffusionPipeline, self).__init__(params, num_epochs, num_batches)

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
        batch["image"]: torch.Tensor = batch["image"].type(torch.float32).to(self._DEVICE)

        # Label
        batch["guider"] = batch["guider"].type(torch.int32).to(self._DEVICE)

        # Predicts added noise
        noisy_image, noise, timestep = self._add_noise(batch["image"])
        return noise, self.components.model(noisy_image, timestep, batch["guider"]).sample

    def inference(
            self,
    ) -> Dict[str, torch.Tensor]:
        """
        Generates and image using the training's components.

        Returns
        ----------
            Dict[str, torch.Tensor]
                generated image
        """
        pipeline = self()
        batch = 5

        # Generates images
        generated_images = pipeline(
                batch_size=batch,
                num_class_embeds=self._params["components"]["model"]["args"]["num_class_embeds"],
                num_inference_steps=1000,
                generator=torch.manual_seed(0)
        ).images

        # Adjusts colors
        images: List[torch.Tensor] = list()
        for image in generated_images:
            images.append(
                utils.adjust_image_colors(
                    utils.to_tensor(image)
                )
            )

        # Returns
        return {
            str(class_idx): torch.stack(images[class_idx * batch:(class_idx + 1) * batch], dim=0)
            for class_idx
            in range(len(images) // batch)
        }
