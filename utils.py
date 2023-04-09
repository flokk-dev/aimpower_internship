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


def adjust_image_colors(image):
    values = torch.unique(image)
    if min(values) >= 0 and max(values) <= 255:
        return image

    return ((image + 1.0) * 127.5).type(torch.uint8)
