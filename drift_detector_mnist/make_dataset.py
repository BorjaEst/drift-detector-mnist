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

from drift_detector_mnist import config

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


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
    default="INFO",
)
# parser.add_argument(
#     *["-b", "--batch_size"],
#     help="Data points processed per pass (default: %(default)s).",
#     type=int,
#     default=128,
# )
parser.add_argument(
    *["data_filepath"],
    nargs="?",
    help="Folder where to generate the datasets (default: %(default)s)",
    type=pathlib.Path,
    default="data",
)


# Script command actions --------------------------------------------
def _run_command(data_filepath, **options):
    logging.basicConfig(level=options["verbosity"])

    logger.debug("Define transform for images into tensors")
    transform = torchvision.transforms.Compose(
        [
            # Convert images to the range [0.0, 1.0] (normalize)
            torchvision.transforms.ToTensor(),
        ]
    )

    logger.debug("Download and transform the training MNIST dataset")
    train_original_dataset = torchvision.datasets.MNIST(
        root=data_filepath,
        train=True,
        download=True,
        transform=transform,
    )

    logger.debug("Download and transform the testing MNIST dataset")
    test_original_dataset = torchvision.datasets.MNIST(
        root=data_filepath,
        train=False,
        download=True,
        transform=transform,
    )

    logger.debug("Merge train and test datasets to avoid bias")
    dataset = torch.utils.data.ConcatDataset(
        datasets=[
            train_original_dataset,
            test_original_dataset,
        ]
    )
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset=dataset,
        lengths=[
            len(train_original_dataset),
            len(test_original_dataset),
        ],
    )

    logger.debug("Calc dataset sizes for model, autoencoder and reference")
    model_dataset_size = 40000
    autoencoder_dataset_size = 19000
    reference_dataset_size = (
        len(train_dataset) - model_dataset_size - autoencoder_dataset_size
    )

    logger.debug("Split into model, autoencoder and reference")
    model_dataset, autoencoder_dataset, reference_dataset = (
        torch.utils.data.random_split(
            dataset=train_dataset,
            lengths=[
                model_dataset_size,
                autoencoder_dataset_size,
                reference_dataset_size,
            ],
        )
    )

    logger.debug("Save the datasets to disk")
    torch.save(model_dataset, f"{data_filepath}/model_dataset.pt")
    torch.save(autoencoder_dataset, f"{data_filepath}/autoencoder_dataset.pt")
    torch.save(reference_dataset, f"{data_filepath}/reference_dataset.pt")

    # logger.debug("Split model dataset into train and validation")
    # model_dataset_size = len(model_dataset)
    # model_train_dataset, model_val_dataset = torch.utils.data.random_split(
    #     dataset=model_dataset,
    #     lengths=[
    #         int(model_dataset_size * 0.8),
    #         int(model_dataset_size * 0.2),
    #     ],
    # )

    # logger.debug("Define data loaders for model and autoencoder")
    # model_train_data_loader = torch.utils.data.DataLoader(
    #     dataset=model_train_dataset, batch_size=batch_size, shuffle=True
    # )
    # model_val_data_loader = torch.utils.data.DataLoader(
    #     dataset=model_val_dataset, batch_size=batch_size, shuffle=False
    # )
    # autoencoder_dataset_size = len(autoencoder_dataset)

    # logger.debug("Split autoencoder dataset into train and validation")
    # autoencoder_train_dataset, autoencoder_val_dataset = (
    #     torch.utils.data.random_split(
    #         dataset=autoencoder_dataset,
    #         lengths=[
    #             int(autoencoder_dataset_size * 0.8),
    #             int(autoencoder_dataset_size * 0.2),
    #         ],
    #     )
    # )

    # logger.debug("Define data loaders for autoencoder training")
    # autoencoder_train_data_loader = torch.utils.data.DataLoader(
    #     dataset=autoencoder_train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    # )

    # logger.debug("Define data loaders for autoencoder validation")
    # autoencoder_val_data_loader = torch.utils.data.DataLoader(
    #     dataset=autoencoder_val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    # )

    # logger.debug("Populate data_filepath with processed data")
    # raise NotImplementedError("TODO")

    # End of program
    logger.info("End of MNIST image processing script")


# Main call ---------------------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    _run_command(**vars(args))
    sys.exit(0)  # Shell return 0 == success
