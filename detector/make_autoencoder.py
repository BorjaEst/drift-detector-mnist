"""Script to generate MNIST autoencoder from a dataset.

The following code is based on the drift detector example from:
https://frouros.readthedocs.io/en/latest/examples/data_drift/MMD_advance.html
"""

# pylint: disable=unused-import,invalid-name
import argparse
import copy
import logging
import pathlib
import sys

import numpy as np
import torch

from detector import config, models, utils

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
parser.add_argument(
    *["--epochs"],
    help="Number of epochs for training (default: %(default)s)",
    type=int,
    default=20,
)
parser.add_argument(
    *["-b", "--batch_size"],
    help="Data points processed per pass (default: %(default)s).",
    type=int,
    default=128,
)
parser.add_argument(
    *["--embedding_dim"],
    help="Dimension of encoded output vector (default: %(default)s)",
    type=int,
    default=5,
)
parser.add_argument(
    *["--patience"],
    help="Epochs improvement before stopping (default: %(default)s)",
    type=int,
    default=3,
)
parser.add_argument(
    *["-n", "--name"],
    help="Model name to use for identification on saves folder.",
    type=str,
    default="mnist_autoencoder",
)
parser.add_argument(
    *["autoencoder_dataset"],
    nargs="?",
    help="Path to training dataset (default: %(default)s)",
    type=pathlib.Path,
    default=f"{config.DATA_PATH}/autoencoder_dataset.pt",
)


# Script command actions --------------------------------------------
def _run_command(autoencoder_dataset, **options):
    logging.basicConfig(level=options["verbosity"])
    logger.info("Start of MNIST autoencoder creation script")

    logger.debug("Generate autoencoder from models")
    autoencoder = models.Autoencoder(
        embedding_dim=options["embedding_dim"],
    ).to(config.device)

    logger.debug("Define optimizer and loss function")
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=1e-3)
    loss = None

    logger.debug("Load MNIST autoencoder dataset")
    autoencoder_dataset = torch.load(autoencoder_dataset)
    autoencoder_dataset_size = len(autoencoder_dataset)

    logger.debug("Split dataset into training and validation")
    autoencoder_train_dataset, autoencoder_val_dataset = (
        torch.utils.data.random_split(
            dataset=autoencoder_dataset,
            lengths=[
                int(autoencoder_dataset_size * 0.8),
                int(autoencoder_dataset_size * 0.2),
            ],
        )
    )

    logger.debug("Define data loaders for autoencoder training")
    autoencoder_train_data_loader = torch.utils.data.DataLoader(
        dataset=autoencoder_train_dataset,
        batch_size=options["batch_size"],
        shuffle=True,
    )

    logger.debug("Define data loaders for autoencoder validation")
    autoencoder_val_data_loader = torch.utils.data.DataLoader(
        dataset=autoencoder_val_dataset,
        batch_size=options["batch_size"],
        shuffle=False,
    )

    logger.debug("Start autoencoder training loop")
    patience_counter = options["patience"]
    for epoch in range(options["epochs"]):
        print(f"Epoch {epoch + 1}:")
        running_loss = 0.0

        autoencoder.train()  # Training
        for i, (inputs, _) in enumerate(autoencoder_train_data_loader, 0):
            inputs = inputs.to(config.device)
            inputs.requires_grad = True
            inputs.retain_grad()

            outputs_e, outputs = autoencoder(inputs)  # pylint: disable=E1102
            loss = utils.contractive_loss(outputs_e, outputs, inputs)

            inputs.requires_grad = False

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:
                print(f"\tTraining loss: {running_loss / inputs.size(0):.4f}")
                running_loss = 0.0

        autoencoder.eval()  # Validation
        val_loss = 0.0
        # with torch.no_grad() is not used here
        # to allow for the calculation of the gradient of the inputs
        for inputs, _ in autoencoder_val_data_loader:
            inputs = inputs.to(config.device)
            inputs.requires_grad = True
            inputs.retain_grad()

            outputs_e, outputs = autoencoder(inputs)  # pylint: disable=E1102
            loss = utils.contractive_loss(outputs_e, outputs, inputs)

            inputs.requires_grad = False

            val_loss += loss.item()

        val_loss /= len(autoencoder_val_data_loader)
        print(f"\tValidation loss: {val_loss:.4f}")

        # Early stopping and save best model
        if val_loss < np.inf:
            best_autoencoder = copy.deepcopy(autoencoder)
            patience_counter = options["patience"]
        else:
            patience_counter -= 1
            if patience_counter == 0:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    logger.debug("Save the best autoencoder model to disk")
    autoencoder = best_autoencoder
    torch.save(autoencoder, f"{config.MODELS_PATH}/{options['name']}.pt")

    logger.info("End of MNIST autoencoder creation script")


# Main call ---------------------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    _run_command(**vars(args))
    sys.exit(0)  # Shell return 0 == success
