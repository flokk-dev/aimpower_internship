"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import os

# IMPORT: data loading
from huggingface_hub import HfApi, get_full_repo_name, create_repo

# IMPORT: deep learning
import torch

from diffusers import StableDiffusionPipeline, UNet2DConditionModel, SchedulerMixin, AutoencoderKL
from transformers import CLIPTextModel

# IMPORT: project
import utils


class Pipeline:
    """
    Represents a Pipeline.

    Attributes
    ----------
        _pipeline : StableDiffusionPipeline
            pipeline containing all the training components

    Methods
    ----------
        model : UNet2DConditionModel
            Returns pipeline's model
        noise_scheduler : SchedulerMixin
            Returns pipeline's noise scheduler
        vae : AutoencoderKL
            Returns pipeline's vae
        text_encoder : CLIPTextModel
            Returns pipeline's text encoder
    """
    def __init__(
            self,
            repo_path: str,
            pipeline_path: str,
            device: str
    ):
        """
        Instantiates a Pipeline.

        Parameters
        ----------
            repo_path: str
                path where to save the pipeline
            pipeline_path : str
                path to the pretrained pipeline to load
            device : str
                device on which to put the pipeline
        """
        # ----- Attributes ----- #
        self._repo_path = repo_path
        self._repo_id = get_full_repo_name(os.path.basename(repo_path))
        create_repo(self._repo_id)

        # Pipeline
        self._pipeline: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=pipeline_path,
            torch_dtype=torch.float16
        ).to(device)

        # Disables gradient descent
        self._pipeline.unet.requires_grad_(False)
        self._pipeline.vae.requires_grad_(False)
        self._pipeline.text_encoder.requires_grad_(False)

        # Disables verification regarding NSFW prompts
        self._pipeline.safety_checker = None

    @property
    def model(
        self
    ) -> UNet2DConditionModel:
        """
        Returns the pipeline's model.

        Returns
        ----------
            UNet2DConditionModel
                pipeline's model
        """
        return self._pipeline.unet

    @property
    def noise_scheduler(
        self
    ) -> SchedulerMixin:
        """
        Returns the pipeline's noise_scheduler.

        Returns
        ----------
            SchedulerMixin
                pipeline's noise_scheduler
        """
        return self._pipeline.scheduler

    @property
    def vae(
        self
    ) -> AutoencoderKL:
        """
        Returns the pipeline's vae.

        Returns
        ----------
            AutoencoderKL
                pipeline's vae
        """
        return self._pipeline.vae

    @property
    def text_encoder(
        self
    ) -> CLIPTextModel:
        """
        Returns the pipeline's text encoder.

        Returns
        ----------
            CLIPTextModel
                pipeline's text encoder
        """
        return self._pipeline.text_encoder

    def _generate_images(
        self,
        prompts: str | List[str],
        inference=False,
        return_dict=False
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Generates images.

        Parameters
        ----------
            prompts: str | List[str]
                prompts needed to generate images
            inference: bool
                whether to generate images for inference or for training
            return_dict: bool
                whether to return a dictionary or not

        Returns
        ----------
            torch.Tensor | Dict[str, torch.Tensor]
                generated images
        """
        self._pipeline.unet.eval()

        # Processes the prompt
        if isinstance(prompts, str):
            prompts = [prompts for i in range(4)]

        # Generates images
        images = utils.images_to_tensor([
            self._pipeline(
                prompt,
                num_inference_steps=50,
                generator=torch.manual_seed(i) if inference else None
            ).images[0]
            for i, prompt
            in enumerate(prompts)
        ])

        if return_dict:
            return {f"{prompt}_{i}": images[i].unsqueeze(0) for i, prompt in enumerate(prompts)}
        return images

    def __call__(
        self,
        prompt: str,
        inference=False,
        return_dict=False
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
            prompt: str
                prompt needed for the generation
            inference: bool
                whether to generate images for inference or for training
            return_dict: bool
                whether to return a dictionary or not

        Returns
        ----------
            torch.Tensor | Dict[str, torch.Tensor]
                generated image
        """
        # Saves the trained LoRA layers
        self._pipeline.unet.save_attn_procs(self._repo_path)

        # Uploads the trained LoRA layers
        HfApi().upload_folder(folder_path=self._repo_path, repo_id=self._repo_id)

        # Inference
        return self._generate_images(prompt, return_dict=return_dict)
