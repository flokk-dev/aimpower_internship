"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
from pynvml import *

import datetime

# IMPORT: dataset processing
import PIL

import torch
import torchvision

# IMPORT: data visualization
import matplotlib.pyplot as plt

# IMPORT: deep learning
from diffusers.utils import randn_tensor
from diffusers import DDPMPipeline
from diffusers.pipeline_utils import ImagePipelineOutput


# ---------- INFO ---------- #

def get_datetime():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)

    return f"{(info.used / 1024**3):.3f}Go"


# ---------- VISUALIZATION ---------- #

def plot_images(prompt, images):
    # Adjust image colors
    adjust_image_colors(images)

    # Adds the subplots
    num_images = images.shape[0]

    plt.figure(figsize=(num_images * 5, num_images * 1.75))
    for idx in range(num_images):
        plt.subplot(1, num_images, idx + 1)
        plt.imshow(images[idx].permute(1, 2, 0))

        plt.axis("off")
        plt.title(f"idx = {idx}", fontsize=24)

    # Plot the images
    plt.suptitle(prompt, fontsize=28)
    plt.tight_layout()
    plt.show()


def save_plt(images, path):
    # Adjust image colors
    adjust_image_colors(images)

    # Adds the subplots
    num_images = images.shape[0]

    plt.figure(figsize=(num_images * 5, num_images * 5))
    for idx in range(num_images):
        plt.subplot(1, num_images, idx + 1)
        plt.imshow(images[idx].permute(1, 2, 0))

    # Plot the images
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# ---------- DATA PROCESSING ---------- #

def images_to_tensors(images: List[PIL.Image.Image]) -> List[torch.Tensor]:
    return [
        adjust_image_colors(to_tensor(image))
        for image
        in images
    ]


def images_to_tensor(images: List[PIL.Image.Image]) -> torch.Tensor:
    return torch.stack([
        adjust_image_colors(to_tensor(image)) for image in images
    ])


def to_tensor(image):
    return torchvision.transforms.ToTensor()(image)


def adjust_image_colors(image):
    values = torch.unique(image)

    if min(values) >= 0 and max(values) <= 1:
        return (image * 255).type(torch.uint8)
    elif min(values) >= -1 and max(values) <= 1:
        return ((image / 2 + 0.5).clamp(0, 1) * 255).type(torch.uint8)

    return image


# ---------- DEEP LEARNING ---------- #

class GuidedDDPMPipeline(DDPMPipeline):
    """ Modification of the DDPMPipeline class from hugging face. """
    def __init__(self, unet, scheduler):
        super(GuidedDDPMPipeline, self).__init__(
            unet=unet,
            scheduler=scheduler
        )

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_class_embeds: int = 10,
        generator: torch.Generator = None,
        num_inference_steps: int = 1000,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> ImagePipelineOutput:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            num_class_embeds (`int`, *optional*, defaults to None):
                Input dimension of the learnable embedding matrix to be projected to `time_embed_dim`, when performing
                class conditioning with `class_embed_type` equal to `None`.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        # Sample vector to guide generation if needed
        labels: torch.Tensor = torch.tensor(
            [[i] * batch_size for i in range(num_class_embeds)],
            device=self.device
        ).flatten()

        # Sample gaussian noise to begin loop
        batch_size *= num_class_embeds

        if isinstance(self.unet.sample_size, int):
            image_shape = (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size)
        else:
            image_shape = (batch_size, self.unet.in_channels, *self.unet.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t, labels).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, )

        return ImagePipelineOutput(images=image)
