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

from diffusers import UNet2DConditionModel, SchedulerMixin, DDPMScheduler

from diffusers import AutoencoderKL, StableDiffusionPipeline
from transformers import CLIPTextModel

from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

# IMPORT: project
from src.loading import PromptLoader, ImagePromptLoader
from src.loading.data_loader import PromptDataLoader


class Components:
    """
    Represents a Components.

    Attributes
    ----------
        _config : Dict[str, Any]
            configuration needed to adjust the program behaviour
        data_loader : PromptDataLoader
            data loader
        model : torch.nn.Module
            training's model
        lora_layers : AttnProcsLayers
            LoRA attention layers
        noise_scheduler : diffusers.SchedulerMixin
            training's noise scheduler
        vae : AutoencoderKL
            auto encoder
        text_encoder : CLIPTextModel
            text encoder
        optimizer : torch.optim.Optimizer
            training's optimizer
        lr_scheduler : torch.nn.Module
            learning rate's scheduler

    Methods
    ----------
        _init_data_loader : PromptDataLoader
            Initializes the data loader
        _init_model : UNet2DConditionModel
            Initializes the model
        _init_lora_layers : AttnProcsLayers
            Initializes the LoRA layers
        _init_noise_scheduler : SchedulerMixin
            Initializes the noise scheduler
        _init_vae : AutoencoderKL
            Initializes an auto encoder
        _init_text_encoder : CLIPTextModel
            Initializes a text encoder
        _init_optimizer : AdamW
            Initializes the optimizer
        _init_lr_scheduler : torch.nn.Module
            Initializes the learning rate's scheduler

        _to_device
            Sends the desired components on device
        prepare
            Prepares the components using an accelerator
    """
    _LOADER = {"prompt": PromptLoader, "prompt_image": ImagePromptLoader}

    def __init__(
            self,
            config: Dict[str, Any],
            dataset_path: str
    ):
        """
        Instantiates a Components.

        Parameters
        ----------
            config : Dict[str, Any]
                configuration needed to adjust the program behaviour
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
        self.data_loader: PromptDataLoader = self._init_data_loader(dataset_path)

        # Pipeline
        self.pipeline: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=self._config["pipeline_path"],
            torch_dtype=torch.float16
        ).to(self.accelerator.device)

        print("aaaaaaaaaa")
        print(self.pipeline.unet.dtype)
        print(self.pipeline.vae.dtype)
        print(self.pipeline.text_encoder.dtype)

        # LoRA layers
        self.lora_layers: AttnProcsLayers = self._init_lora_layers()

        # ----- Attributes to inherit ----- #
        # Optimizer
        self.optimizer: Optimizer = None

        # Learning rate
        self.lr_scheduler: torch.nn.Module = None

    def _init_data_loader(
            self,
            dataset_path: str
    ) -> PromptDataLoader:
        """
        Initializes the data loader.

        Parameters
        ----------
            dataset_path : str
                path to the dataset
        """
        return self._LOADER[self._config["loading_type"]](
            self._config
        )(dataset_path)

    def _init_lora_layers(
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
        for name in self.pipeline.unet.attn_processors.keys():
            cross_attention_dim = None \
                if name.endswith("attn1.processor") \
                else self.pipeline.unet.config.cross_attention_dim

            hidden_size = None
            if name.startswith("mid_block"):
                hidden_size = self.pipeline.unet.config.block_out_channels[-1]

            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.pipeline.unet.config.block_out_channels))[block_id]

            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.pipeline.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim
            )

        # Adds LoRA layers to the model
        self.pipeline.unet.set_attn_processor(lora_attn_procs)
        return AttnProcsLayers(self.pipeline.unet.attn_processors)

    def _init_optimizer(
            self
    ) -> AdamW:
        """
        Initializes the optimizer.

        Returns
        ----------
            AdamW
                training optimizer

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        return torch.optim.AdamW(
            self.lora_layers.parameters(), **self._config["optimizer"]["args"]
        )

    def _init_lr_scheduler(
            self
    ) -> torch.nn.Module:
        """
        Instantiates an optimizer and a scheduler.

        Returns
        ----------
            torch.nn.Module
                learning rate's scheduler

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        return get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_training_steps=(len(self.data_loader) * self._config["num_epochs"]),
            **self._config["lr_scheduler"]["args"],
        )

    def _to_device(
            self
    ):
        """ Sends the desired components on device. """
        # Model
        self.pipeline.unet.to(self.accelerator.device, dtype=torch.float16)

        # VAE
        self.pipeline.vae.to(self.accelerator.device, dtype=torch.float16)

        # Text encoder
        self.pipeline.text_encoder.to(self.accelerator.device, dtype=torch.float16)

    def prepare(
            self
    ):
        """ Prepares the components using an accelerator. """
        self._to_device()

        # Accelerator
        self.lora_layers, self.optimizer, self.data_loader, self.lr_scheduler = \
            self.accelerator.prepare(
                self.lora_layers, self.optimizer, self.data_loader, self.lr_scheduler
            )
