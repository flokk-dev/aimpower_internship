"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import os

# IMPORT: deep learning
import torch
from diffusers import \
    DDPMPipeline, \
    StableDiffusionPipeline as HFStableDiffusionPipeline

# IMPORT: project
import utils

from .pipeline import Pipeline
from src.training.learner.components import \
    DiffusionComponents, StableDiffusionComponents, LoRADiffusionComponents


class DiffusionPipeline(Pipeline):
    """
    Represents a DiffusionPipeline.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the pipeline behaviour

    Methods
    ----------
        inference: Dict[str, torch.Tensor]
            Builds the pipeline using its components
    """

    def __init__(
            self,
            params: Dict[str, Any]
    ):
        """
        Instantiates a DiffusionPipeline.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the pipeline behaviour
        """
        # ----- Mother Class ----- #
        super(DiffusionPipeline, self).__init__(params)

    def _inference(
            self,
            pipeline: DDPMPipeline
    ) -> Dict[str, torch.Tensor]:
        """
        Generates images using a diffusion pipeline.

        Parameters
        ----------
            pipeline : DDPMPipeline
                components needed to instantiate the pipeline

        Returns
        ----------
            Dict[str, torch.Tensor]
                generated image
        """
        generated_images = pipeline(
            batch_size=5,
            num_inference_steps=1000,
            generator=torch.manual_seed(0)
        ).images

        # Converts images to tensors
        images: List[torch.Tensor] = utils.images_to_tensors(generated_images)

        # Returns
        return {"image": torch.stack(images, dim=0)}

    def checkpoint(
        self,
        components: DiffusionComponents,
        save_path: str
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
            components : DiffusionComponents
                components needed to generate images
            save_path: str
                path where to save the pipeline

        Returns
        ----------
            Dict[str, torch.Tensor]
                generated image
        """
        pipeline = DDPMPipeline(
            unet=components.accelerator.unwrap_model(components.model),
            scheduler=components.noise_scheduler,
            torch_dtype=torch.float16 if self._params["components"]["fp16"] else torch.float32
        ).to(components.accelerator.device)
        pipeline.safety_checker = None

        # Save
        pipeline.save_pretrained(os.path.join(save_path, "pipeline"))

        # Inference
        return self._inference(pipeline)


class GuidedDiffusionPipeline(Pipeline):
    """
    Represents a GuidedDiffusionPipeline.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the pipeline behaviour

    Methods
    ----------
        inference: Dict[str, torch.Tensor]
            Builds the pipeline using its components
    """

    def __init__(
            self,
            params: Dict[str, Any]
    ):
        """
        Instantiates a GuidedDiffusionPipeline.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the pipeline behaviour
        """
        # ----- Mother Class ----- #
        super(GuidedDiffusionPipeline, self).__init__(params)

    def _inference(
            self,
            pipeline: DDPMPipeline
    ) -> Dict[str, torch.Tensor]:
        """
        Generates images using a diffusion pipeline.

        Parameters
        ----------
            pipeline : DDPMPipeline
                components needed to instantiate the pipeline

        Returns
        ----------
            Dict[str, torch.Tensor]
                generated image
        """
        generated_images = pipeline(
            batch_size=5,
            num_class_embeds=self._params["components"]["model"]["args"]["num_class_embeds"],
            num_inference_steps=1000,
            generator=torch.manual_seed(0)
        ).images

        # Converts images to tensors
        images: List[torch.Tensor] = utils.images_to_tensors(generated_images)

        # Returns
        return {
            str(class_idx): images[class_idx * 5:(class_idx + 1) * 5]
            for class_idx
            in range(self._params["components"]["model"]["args"]["num_class_embeds"])
        }

    def checkpoint(
        self,
        components: DiffusionComponents,
        save_path: str
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
            components : DiffusionComponents
                components needed to generate images
            save_path: str
                path where to save the pipeline

        Returns
        ----------
            Dict[str, torch.Tensor]
                generated image
        """
        pipeline = utils.GuidedDDPMPipeline(
            unet=components.accelerator.unwrap_model(components.model),
            scheduler=components.noise_scheduler,
            torch_dtype=torch.float16 if self._params["components"]["fp16"] else torch.float32
        ).to(components.accelerator.device)
        pipeline.safety_checker = None

        # Save
        pipeline.save_pretrained(os.path.join(save_path, "pipeline"))

        # Inference
        return self._inference(pipeline)


class StableDiffusionPipeline(Pipeline):
    """
    Represents a StableDiffusionPipeline.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the pipeline behaviour

    Methods
    ----------
        inference: Dict[str, torch.Tensor]
            Builds the pipeline using its components
    """

    def __init__(
            self,
            params: Dict[str, Any]
    ):
        """
        Instantiates a StableDiffusionPipeline.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the pipeline behaviour
        """
        # ----- Mother Class ----- #
        super(StableDiffusionPipeline, self).__init__(params)

    def _inference(
            self,
            pipeline: HFStableDiffusionPipeline
    ) -> Dict[str, torch.Tensor]:
        """
        Generates images using a diffusion pipeline.

        Parameters
        ----------
            pipeline : HFStableDiffusionPipeline
                components needed to instantiate the pipeline

        Returns
        ----------
            Dict[str, torch.Tensor]
                generated image
        """
        generated_images = list()

        for idx, prompt in enumerate(self._params["validation_prompts"]):
            generated_images.append(
                pipeline(
                    prompt,
                    num_inference_steps=50,
                    generator=torch.manual_seed(idx)
                ).images[0]
            )

        # Adjusts colors
        images: List[torch.Tensor] = utils.images_to_tensors(generated_images)

        # Returns
        return {
            f"{prompt}_{idx}": images[idx].unsqueeze(0)
            for idx, prompt
            in enumerate(self._params["validation_prompts"])
        }

    def checkpoint(
        self,
        components: StableDiffusionComponents,
        save_path: str
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
            components : StableDiffusionComponents
                components needed to generate images
            save_path: str
                path where to save the pipeline

        Returns
        ----------
            Dict[str, torch.Tensor]
                generated image
        """
        pipeline = HFStableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=self._params["components"]["pipeline_path"],
            unet=components.accelerator.unwrap_model(components.model),
            torch_dtype=torch.float16 if self._params["components"]["fp16"] else torch.float32
        ).to(components.accelerator.device)
        pipeline.safety_checker = None

        # Save
        pipeline.save_pretrained(os.path.join(save_path, "pipeline"))

        # Inference
        return self._inference(pipeline)


class LoRADiffusionPipeline(StableDiffusionPipeline):
    """
    Represents a LoRADiffusionPipeline.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the pipeline behaviour

    Methods
    ----------
        inference: Dict[str, torch.Tensor]
            Builds the pipeline using its components
    """

    def __init__(
            self,
            params: Dict[str, Any]
    ):
        """
        Instantiates a LoRADiffusionPipeline.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the pipeline behaviour
        """
        # ----- Mother Class ----- #
        super(LoRADiffusionPipeline, self).__init__(params)

    def checkpoint(
        self,
        components: LoRADiffusionComponents,
        save_path: str
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
            components : LoRADiffusionComponents
                components needed to generate images
            save_path: str
                path where to save the pipeline

        Returns
        ----------
            Dict[str, torch.Tensor]
                generated image
        """
        pipeline = HFStableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=self._params["components"]["pipeline_path"],
            unet=components.accelerator.unwrap_model(components.model),
            torch_dtype=torch.float16 if self._params["components"]["fp16"] else torch.float32
        ).to(components.accelerator.device)
        pipeline.safety_checker = None

        # Save
        components.accelerator.unwrap_model(
            components.model
        ).to(torch.float32).save_attn_procs(save_path)

        # Inference
        return self._inference(pipeline)
