"""
Creator: Flokk
Date: 09/04/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: data processing
import torch

# IMPORT: deep learning
from transformers import AutoProcessor, AutoModel


class PickAPicScore:
    """
    Represents a PickAPicScore.

    Attributes
    ----------
        _device : str
            device to put components on
        _processor : torch.nn.Module
            tensor's processor
        _model : torch.nn.Module
            PickAPic model
    """
    def __init__(
        self,
        device: str
    ):
        """
        Instantiates a PickAPicScore.

        Parameters
        ----------
            device : str
                device to put components on
        """
        # ----- Attributes ----- #
        self._device = device

        # Processor
        self._processor: torch.nn.Module = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )

        # Model
        self._model: torch.nn.Module = AutoModel.from_pretrained(
            pretrained_model_name_or_path="yuvalkirstain/PickScore_v1"
        ).eval()

    def __call__(
        self,
        prompt: str,
        images: List[torch.tensor]
    ) -> torch.Tensor:
        """
        Parameters
        ----------
            prompt : str
                prompt to evaluate
            images : List[torch.tensor]
                images to evaluate

        Returns
        ----------
            List[torch.tensor]
                reward values
        """
        # Images
        processed_images = self._processor(
            images=images, padding=True, truncation=True, max_length=77, return_tensors="pt",
        ).to(self._device)

        # Prompts
        processed_prompt = self._processor(
            text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt",
        ).to(self._device)

        # Computes the reward
        with torch.no_grad():
            # Images embedding
            embedded_images = self._model.get_image_features(**processed_images)
            embedded_images = embedded_images / torch.norm(embedded_images, dim=-1, keepdim=True)

            # Prompt embedding
            embedded_prompt = self._model.get_text_features(**processed_prompt)
            embedded_prompt = embedded_prompt / torch.norm(embedded_prompt, dim=-1, keepdim=True)

            # Computes PickAPic score
            scores: torch.Tensor = self._model.logit_scale.exp() * (embedded_prompt @ embedded_images.T)[0]
            probs: torch.Tensor = torch.softmax(scores, dim=-1)

        return probs.cpu().tolist()
