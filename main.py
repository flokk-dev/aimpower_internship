"""
Creator: Flokk
Date: 01/03/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import argparse
import json

# IMPORT: projet
import paths

from src.training import LossTrainer


class Parser(argparse.ArgumentParser):
    def __init__(self):
        # Mother class
        super(Parser, self).__init__(description="Get models training parameters.")

        # dataset
        self.add_argument(
            "-d", "--dataset", type=str, nargs="?",
            help="path to the dataset"
        )

        # weights
        self.add_argument(
            "-w", "--weights", type=str, nargs="?", default=None,
            help="path to the model's weights"
        )


TASKS = {
    "loss": LossTrainer, "reward": None
}

if __name__ == "__main__":
    # Training arguments
    parser = Parser()
    args = parser.parse_args()

    # Training parameters
    with open(paths.CONFIG_PATH) as json_file:
        training_parameters = json.load(json_file)

    # Launch training
    trainer = TASKS[training_parameters["loss_type"]](params=training_parameters)
    trainer(dataset_path=args.dataset, weights_path=args.weights)
