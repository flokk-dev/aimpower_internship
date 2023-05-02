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

from diffusers.optimization import Optimizer, get_cosine_schedule_with_warmup
from diffusers import UNet2DModel, UNet2DConditionModel, \
    SchedulerMixin, DDPMScheduler, DDIMScheduler

from accelerate import Accelerator

# IMPORT: project
from src.loading.loader import Loader
from src.loading.data_loader import DataLoader


class DiffusionComponents:
    """
    Represents a DiffusionComponents.

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

        _to_device
            Sends the desired components on device
        prepare
            Prepares the components using an accelerator
    """
    _M_TYPES = {"unet": UNet2DModel, "conditioned unet": UNet2DConditionModel}
    _NS_TYPES = {"ddpm": DDPMScheduler, "ddim": DDIMScheduler}

    def __init__(
            self,
            params: Dict[str, Any],
            dataset_path: str,
            num_epochs: int
    ):
        """
        Instantiates a DiffusionComponents.

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
        self._params: Dict[str, Any] = params

        # Accelerator
        self.accelerator: Accelerator = Accelerator(
            gradient_accumulation_steps=4,
            mixed_precision=params["dtype"],
            cpu=False if torch.cuda.is_available() else True
        )

        # Data Loader
        self.data_loader: DataLoader = None
        self._init_data_loader(dataset_path)

        # Model
        self.model: torch.nn.Module = None
        self._init_model()

        # Noise scheduler
        self.noise_scheduler: SchedulerMixin = None
        self._init_noise_scheduler()

        # Optimizer
        self.optimizer: Optimizer = None
        self._init_optimizer()

        # Learning rate
        self.lr_scheduler: torch.nn.Module = None
        self._init_lr_scheduler(num_epochs)

    def _init_data_loader(
            self,
            dataset_path: str
    ):
        """
        Initializes the data loader.

        Parameters
        ----------
            dataset_path : str
                path to the dataset
        """
        self.data_loader = Loader(self._params)(
            dataset_path
        )

    def _init_model(
            self
    ):
        """ Initializes the model. """
        # Loads
        if self._params["model"]["load"]:
            self.model = self._M_TYPES[self._params["model"]["type"]].from_pretrained(
                pretrained_model_name_or_path=self._params["pipeline_path"],
                subfolder="unet",
                revision=self._params["dtype"]
            )

        # Instantiates
        else:
            self.model = self._M_TYPES[self._params["model"]["type"]](
                **self._params["model"]["args"]
            )

    def _init_noise_scheduler(
            self
    ):
        """ Initializes the noise scheduler. """
        # Loads
        if self._params["noise_scheduler"]["load"]:
            self.noise_scheduler = self._NS_TYPES[self._params["noise_scheduler"]["type"]].from_pretrained(
                pretrained_model_name_or_path=self._params["pipeline_path"],
                subfolder="scheduler",
                revision=self._params["dtype"]
            )

        # Instantiates
        else:
            self.noise_scheduler = self._NS_TYPES[self._params["noise_scheduler"]["type"]](
                **self._params["noise_scheduler"]["args"]
            )

    def _init_optimizer(
            self
    ):
        """ Initializes the optimizer. """
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), **self._params["optimizer"]["args"]
        )

    def _init_lr_scheduler(
            self,
            num_epochs: int
    ):
        """
        Instantiates an optimizer and a scheduler.

        Parameters
        ----------
            num_epochs : int
                number of epochs during the training
        """
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_training_steps=(len(self.data_loader) * num_epochs),
            **self._params["lr_scheduler"]["args"],
        )

    def _to_device(
            self
    ):
        """ Sends the desired components on device. """
        self.model.to(
            self.accelerator.device,
            # dtype=torch.float16 if self._params["dtype"] == "fp16" else torch.float32
        )

    def prepare(
            self
    ):
        """ Prepares the components using an accelerator. """
        # Device
        self._to_device()

        # Accelerator
        self.model, self.optimizer, self.data_loader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.data_loader, self.lr_scheduler
        )
