"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import datetime

# IMPORT: dataset processing
import torch
from torchvision import transforms

# IMPORT: data visualization
import matplotlib.pyplot as plt


def get_datetime():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def get_max_item(dictionary):
    key = max(dictionary, key=dictionary.get)
    return key, dictionary[key]


def size_of(tensor: torch.Tensor) -> float:
    total = tensor.element_size()
    for shape in tensor.shape:
        total *= float(shape)

    return total / 1e6


def str_to_tensor(string: str):
    return torch.tensor(int(string))


def adjust_image_colors(image):
    values = torch.unique(image)
    if min(values) >= 0 and max(values) <= 255:
        return image

    return ((image + 1.0) * 127.5).type(torch.uint8)


def save_plt(tensor, path):
    # Converts images into tensors
    pil_to_tensor = transforms.PILToTensor()
    tensor = torch.stack([pil_to_tensor(image) for image in tensor])

    # Adds the subplots
    num_images = tensor.shape[0]

    plt.figure(figsize=(num_images * 5, num_images * 5))
    for b_idx in range(num_images):
        plt.subplot(1, num_images, b_idx + 1)
        plt.imshow(tensor[b_idx].permute(1, 2, 0))

    # Plot the images
    plt.savefig(path, bbox_inches="tight")
