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
import transformers

import diffusers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers

# IMPORT: project
from .components_v2 import ComponentsV2


class ComponentsV3(ComponentsV2):
    """
    Represents an ComponentsV3.

    Attributes
    ----------
        _params : Dict[str, Any]
            parameters needed to adjust the program behaviour
        model : torch.nn.Module
            training's model
        noise_scheduler : diffusers.SchedulerMixin
            training's noise scheduler
        optimizer : torch.optim.Optimizer
            training's optimizer
        lr_scheduler : torch.nn.Module
            learning rate's scheduler
        vae : diffusers.AutoencoderKL
            training's image encoder
        text_encoder : transformers.CLIPTextModel
            training's text encoder

    Methods
    ----------
        _init_vae : diffusers.AutoencoderKL
            Instantiates an image encoder
        _init_text_encoder : transformers.CLIPTextModel
            Instantiates a text encoder
    """
    def __init__(
            self,
            params: Dict[str, Any],
            num_epochs: int,
            num_batches: int
    ):
        """
        Instantiates a LoraComponents.

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
        super(ComponentsV3, self).__init__(params, num_epochs, num_batches)

        # Model
        self.model.requires_grad_(False)
        self.lora_layers = self._add_lora_layers()

        self.optimizer: diffusers.optimization.Optimizer = torch.optim.AdamW(
            self.lora_layers.parameters(), lr=self._params["optimizer"]["lr"]
        )

        self.lr_scheduler: torch.nn.Module = \
            diffusers.optimization.get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self._params["optimizer"]["lr_warmup_steps"],
                num_training_steps=(num_batches * num_epochs)
            )

    def _add_lora_layers(self) -> AttnProcsLayers:
        """
        Adds LoRA attention layers to the model.

        Returns
        ----------
            AttnProcsLayers
                LoRA layers
        """
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
        return AttnProcsLayers(self.model.attn_processors)
