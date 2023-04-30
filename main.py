"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import os
import json
import argparse

import torch

# IMPORT: project
import paths

from src import Trainer


class Parser(argparse.ArgumentParser):
    def __init__(self):
        # Mother class
        super(Parser, self).__init__(description="Initializes training's parameters.")

        # dataset
        self.add_argument(
            "-d", "--dataset", type=str, nargs="?",
            help="path to the dataset."
        )

        # weights
        self.add_argument(
            "-c", "--config", type=str, nargs="?", default=None,
            help="path to the config file."
        )


if __name__ == "__main__":
    # Training arguments
    parser = Parser()
    args = parser.parse_args()

    # Training parameters
    with open(os.path.join(paths.CONFIG_PATH, args.config)) as json_file:
        parameters = json.load(json_file)

    if parameters["dtype"] == "float16":
        parameters["revision"] = "fp16"
        parameters["dtype"] = torch.float16
    else:
        parameters["revision"] = None
        parameters["dtype"] = torch.float32

    # Launch training
    trainer = Trainer(params=parameters)
    trainer(dataset_path=args.dataset)
