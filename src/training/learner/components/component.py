"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: deep learning
import torch
from torch.optim import Optimizer, AdamW
from diffusers.optimization import get_cosine_schedule_with_warmup

from accelerate import Accelerator

from diffusers import StableDiffusionPipeline, UNet2DConditionModel, SchedulerMixin, DDPMScheduler

from diffusers import AutoencoderKL, StableDiffusionPipeline
from transformers import CLIPTextModel

from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

# IMPORT: project
from src.loading import PromptLoader, ImagePromptLoader
from src.loading.data_loader import PromptDataLoader

from .pipeline import Pipeline


class Components:
    """
    Represents a Components.

    Attributes
    ----------
        _config : Dict[str, Any]
            configuration needed to adjust the program behaviour
        accelerator : Accelerator
            accelerator needed to improve the training
        data_loader : PromptDataLoader
            data loader
        pipeline : StableDiffusionPipeline
            pipeline to train
        lora_layers : AttnProcsLayers
            model's LoRA layers

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

        _create_lora_layers : AttnProcsLayers
            Initializes the LoRA layers
        prepare
            Prepares the components using an accelerator
    """
    _LOADER = {"prompt": PromptLoader, "prompt_image": ImagePromptLoader}

    def __init__(
            self,
            config: Dict[str, Any],
            repo_path: str,
            dataset_path: str
    ):
        """
        Instantiates a Components.

        Parameters
        ----------
            config : Dict[str, Any]
                configuration needed to adjust the program behaviour
            repo_path: str
                path where to save the pipeline
            dataset_path : str
                path to the dataset
        """
        # ----- Attributes ----- #
        self._config: Dict[str, Any] = config

        # Accelerator
        self.accelerator: Accelerator = Accelerator(
            gradient_accumulation_steps=4,
            mixed_precision="fp16",
            cpu=False if torch.cuda.is_available() else True
        )

        # Data Loader
        self.data_loader: PromptDataLoader = self._LOADER[self._config["loading_type"]](
            self._config
        )(dataset_path)

        # Pipeline
        self.pipeline: Pipeline = Pipeline(
            repo_path=repo_path,
            pipeline_path=self._config["pipeline_path"],
            device=self.accelerator.device
        )

        # LoRA layers
        self.lora_layers: AttnProcsLayers = self._create_lora_layers()

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
        return self.pipeline.model

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
        return self.pipeline.noise_scheduler

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
        return self.pipeline.vae

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
        return self.pipeline.text_encoder

    def _create_lora_layers(
            self
    ) -> AttnProcsLayers:
        """
        Initializes the LoRA layers.

        Returns
        ----------
            AttnProcsLayers
                LoRA layers
        """
        lora_attn_procs = dict()
        for name in self.pipeline.model.attn_processors.keys():
            cross_attention_dim = None \
                if name.endswith("attn1.processor") \
                else self.pipeline.model.config.cross_attention_dim

            hidden_size = None
            if name.startswith("mid_block"):
                hidden_size = self.pipeline.model.config.block_out_channels[-1]

            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(
                    self.pipeline.model.config.block_out_channels
                ))[block_id]

            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.pipeline.model.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim
            )

        # Adds LoRA layers to the model
        self.pipeline.model.set_attn_processor(lora_attn_procs)
        return AttnProcsLayers(self.pipeline.model.attn_processors)

    def prepare(
            self
    ):
        """
        Prepares the components using an accelerator.

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()
