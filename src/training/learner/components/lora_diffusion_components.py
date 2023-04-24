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

from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

# IMPORT: project
from .stable_diffusion_components import StableDiffusionComponents


class LoRADiffusionComponents(StableDiffusionComponents):
    """
    Represents a LoRADiffusionComponents.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        noise_scheduler : diffusers.SchedulerMixin
            training's noise scheduler
        model : torch.nn.Module
            training's model
        lora_layers : AttnProcsLayers
            LoRA attention layers
        optimizer : torch.optim.Optimizer
            training's optimizer
        lr_scheduler : torch.nn.Module
            learning rate's scheduler
        vae : AutoencoderKL
            auto encoder
        text_encoder : CLIPTextModel
            text encoder

    Methods
    ----------
        _init_data_loader
            Initializes the data loader
        _init_model
            Initializes the model
        _init_noise_scheduler
            Initializes the noise scheduler
        _init_optimizer
            Initializes the optimizer
        _init_lr_scheduler
            Initializes the learning rate's scheduler
        _init_vae
            Initializes an auto encoder
        _init_text_encoder
            Initializes a text encoder

        _to_device
            Sends the desired components on device
        prepare
            Prepares the components using an accelerator
    """

    def __init__(
            self,
            params: Dict[str, Any],
            dataset_path: str,
            num_epochs: int
    ):
        """
        Instantiates a LoRADiffusionComponents.

        Parameters
        ----------
            params : Dict[str, Any]
                parameters needed to adjust the program behaviour
            dataset_path : str
                path to the dataset
            num_epochs : int
                number of epochs during the training
        """
        # ----- Attributes ----- #
        self.lora_layers: AttnProcsLayers = None

        # ----- Mother Class ----- #
        super(LoRADiffusionComponents, self).__init__(params, dataset_path, num_epochs)

    def _init_model(
            self
    ):
        """ Initializes the model. """
        super()._init_model()
        self.model.requires_grad_(False)

        # LoRA layers
        lora_attn_procs = dict()
        for name in self.model.attn_processors.keys():
            cross_attention_dim = None \
                if name.endswith("attn1.processor") \
                else self.model.config.cross_attention_dim

            hidden_size = None
            if name.startswith("mid_block"):
                hidden_size = self.model.config.block_out_channels[-1]

            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.model.config.block_out_channels))[block_id]

            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.model.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim
            )

        self.model.set_attn_processor(lora_attn_procs)
        self.lora_layers = AttnProcsLayers(self.model.attn_processors)

    def _init_optimizer(
            self
    ):
        """ Initializes the optimizer. """
        self.optimizer = torch.optim.AdamW(
            self.lora_layers.parameters(), lr=self._params["optimizer"]["lr"]
        )

    def prepare(
            self
    ):
        """ Prepares the components using an accelerator. """
        # Device
        self._to_device()

        # Accelerator
        """
        self.lora_layers, self.optimizer, self.data_loader, self.lr_scheduler = \
            self.accelerator.prepare(
                self.lora_layers, self.optimizer, self.data_loader, self.lr_scheduler
            )
        """
