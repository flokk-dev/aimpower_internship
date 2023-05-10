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
from huggingface_hub import HfApi, create_repo, upload_folder, get_full_repo_name

# IMPORT: deep learning
import torch
from diffusers import StableDiffusionPipeline

# IMPORT: project
import utils


class Pipeline:
    """
    Represents a Pipeline.

    Attributes
    ----------
        _config : Dict[str, Any]
            configuration needed to adjust the pipeline behaviour

    Methods
    ----------
        checkpoint : Dict[str, torch.Tensor]
            Runs a checkpoint procedure
        _upload
            Uploads LoRA layers to the huggingface hub
        inference: Dict[str, torch.Tensor]
            Builds the pipeline using its components
    """
    def __init__(
            self,
            config: Dict[str, Any]
    ):
        """
        Instantiates a Pipeline.

        Parameters
        ----------
            config : Dict[str, Any]
                configuration needed to adjust the pipeline behaviour
        """
        # ----- Attributes ----- #
        self._config: Dict[str, Any] = config

    def checkpoint(
        self,
        components,
        repo_path: str
    ) -> Dict[str, torch.Tensor]:
        """
        Runs a checkpoint procedure.

        Parameters
        ----------
            components
                components needed to generate images
            repo_path: str
                path where to save the pipeline

        Returns
        ----------
            Dict[str, torch.Tensor]
                generated image
        """
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=self._config["pipeline_path"],
            unet=components.accelerator.unwrap_model(components.model),
            torch_dtype=torch.float16
        ).to(components.accelerator.device)
        pipeline.safety_checker = None

        # Saves
        components.accelerator.unwrap_model(
            components.model
        ).save_attn_procs(repo_path)

        # Uploads
        self._upload(repo_path)

        # Inference
        return self._inference(pipeline)

    @staticmethod
    def _upload(
            repo_path: str
    ):
        """
        Uploads LoRA layers to the huggingface hub.

        Parameters
        ----------
            repo_path: str
                path where to save the pipeline
        """
        # Create repository
        repo_id = get_full_repo_name(repo_path)
        create_repo(repo_id)

        # Uploads local repository
        api = HfApi()
        api.upload_folder(folder_path=repo_path, repo_id=repo_id)

    def _inference(
            self,
            pipeline: StableDiffusionPipeline
    ) -> Dict[str, torch.Tensor]:
        """
        Generates images using a diffusion pipeline.

        Parameters
        ----------
            pipeline : StableDiffusionPipeline
                components needed to instantiate the pipeline

        Returns
        ----------
            Dict[str, torch.Tensor]
                generated image
        """
        pipeline.unet.eval()

        # Generates images
        generated_images = list()
        for idx, prompt in enumerate(self._config["validation_prompts"]):
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
            in enumerate(self._config["validation_prompts"])
        }
