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

# IMPORT: project
import paths
from src import LossTrainer, RewardTrainer


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


TASKS = {"learning": LossTrainer, "reinforcement_learning": RewardTrainer}


if __name__ == "__main__":
    # Training arguments
    parser = Parser()
    args = parser.parse_args()

    # Training parameters
    with open(os.path.join(paths.CONFIG_PATH, args.config)) as json_file:
        configuration = json.load(json_file)

    # Launch training
    trainer = TASKS[configuration["training_type"]](config=configuration)
    trainer(dataset_path=args.dataset)
