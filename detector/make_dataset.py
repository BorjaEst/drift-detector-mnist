"""Script to generate pre-processed MNIST dataset using autoencoder.

The following code is based on the drift detector example from:
https://frouros.readthedocs.io/en/latest/examples/data_drift/MMD_advance.html
"""

# pylint: disable=unused-import
import argparse
import logging
import pathlib
import sys

import torch
import torchvision

from detector import config

# Setup logging -----------------------------------------------------
logger = logging.getLogger(__name__)


# Script arguments definition ---------------------------------------
parser = argparse.ArgumentParser(
    prog="PROG",
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="See '<command> --help' to read about a specific sub-command.",
)
parser.add_argument(
    *["-v", "--verbosity"],
    help="Sets the logging level (default: %(default)s)",
    type=str,
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    default=config.LOG_LEVEL,
)


# Script command actions --------------------------------------------
def _run_command(**options):
    logging.basicConfig(level=options["verbosity"])
    logger.info("Start of MNIST image processing script")

    logger.info("Define transform for images into tensors")
    transform = torchvision.transforms.Compose(
        [
            # Convert images to the range [0.0, 1.0] (normalize)
            torchvision.transforms.ToTensor(),
        ]
    )

    logger.info("Download and transform the training MNIST dataset")
    train_original_dataset = torchvision.datasets.MNIST(
        root=config.DATA_PATH,
        train=True,
        download=True,
        transform=transform,
    )

    logger.info("Download and transform the testing MNIST dataset")
    test_original_dataset = torchvision.datasets.MNIST(
        root=config.DATA_PATH,
        train=False,
        download=True,
        transform=transform,
    )

    logger.info("Merge train and test datasets to avoid bias")
    dataset = torch.utils.data.ConcatDataset(
        datasets=[
            train_original_dataset,
            test_original_dataset,
        ]
    )
    train_dataset, _test_dataset = torch.utils.data.random_split(
        dataset=dataset,
        lengths=[
            len(train_original_dataset),
            len(test_original_dataset),
        ],
    )

    logger.info("Calc dataset sizes for model, autoencoder and reference")
    model_dataset_size = 40000
    autoencoder_dataset_size = 19000
    reference_dataset_size = (
        len(train_dataset) - model_dataset_size - autoencoder_dataset_size
    )

    logger.info("Split into model, autoencoder and reference")
    datasets = torch.utils.data.random_split(
        dataset=train_dataset,
        lengths=[
            model_dataset_size,
            autoencoder_dataset_size,
            reference_dataset_size,
        ],
    )

    logger.info("Save datasets to disk in %s", config.DATA_PATH)
    torch.save(datasets[0], f"{config.DATA_PATH}/model_dataset.pt")
    torch.save(datasets[1], f"{config.DATA_PATH}/autoencoder_dataset.pt")
    torch.save(datasets[2], f"{config.DATA_PATH}/reference_dataset.pt")

    logger.info("End of MNIST image processing script")


# Main call ---------------------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    _run_command(**vars(args))
    sys.exit(0)  # Shell return 0 == success
