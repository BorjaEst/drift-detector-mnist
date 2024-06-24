"""Script to generate MNIST drift detectors using autoencoders.

The following code is based on the drift detector example from:
https://frouros.readthedocs.io/en/latest/examples/data_drift/MMD_advance.html
"""

# pylint: disable=unused-import,invalid-name
import argparse
import logging
import pathlib
import sys
from functools import partial

import frouros.callbacks
import frouros.utils
import numpy as np
import scipy as sp
import torch
from frouros.detectors import concept_drift, data_drift

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
parser.add_argument(
    *["-p", "--num_permutations"],
    help="Permutations for MMD computation (default: %(default)s).",
    type=int,
    default=128,
)
parser.add_argument(
    *["-n", "--name"],
    help="File name to where to save the detector.",
    type=str,
    default="mmd_detector",
)
parser.add_argument(
    *["--autoencoder"],
    help="File name containing the autoencoder for the detection.",
    type=str,
    default="mnist_autoencoder",
)
parser.add_argument(
    *["reference_dataset"],
    nargs="?",
    help="Path to file containing the reference data (default: %(default)s)",
    type=pathlib.Path,
    default="reference_dataset",
)


# Script command actions --------------------------------------------
def _run_command(autoencoder, reference_dataset, **options):
    logging.basicConfig(level=options["verbosity"])
    logger.info("Start of MNIST detector generation script")

    logger.info("Load the autoencoder model from disk")
    autoencoder = torch.load(f"{config.MODELS_PATH}/{autoencoder}.pt")

    logger.info("Load the reference data and encode it")
    reference_dataset = torch.load(f"{config.DATA_PATH}/{reference_dataset}.pt")
    X_ref_sample = np.array([X.tolist() for X, _ in reference_dataset])
    X_ref_sample = X_ref_sample.astype(np.float32)
    X_ref_encoded = autoencoder.encoder(torch.Tensor(X_ref_sample))
    X_ref_encoded = X_ref_encoded.cpu().detach().numpy()

    logger.info("Define the detector kernel and parameters")
    pdist = sp.spatial.distance.pdist(X=X_ref_encoded, metric="euclidean")
    sigma = np.median(pdist)
    kernel = partial(frouros.utils.kernels.rbf_kernel, sigma=sigma)

    logger.info("Create a MMD detector with permutation test")
    detector = data_drift.MMD(
        kernel,
        callbacks=[
            frouros.callbacks.PermutationTestDistanceBased(
                options["num_permutations"],
                # random_state=seed,
                num_jobs=-1,
                method="exact",
                name="permutation_test",
                verbose=options["verbosity"] == "DEBUG",
            ),
        ],
    )

    logger.info("Fit the detector with the reference data")
    fit_logs = detector.fit(X=X_ref_encoded)
    logger.debug("Fit logs: %s", fit_logs)

    logger.info("Save the detector to disk")
    torch.save(detector, f"{config.MODELS_PATH}/{options['name']}.pt")

    # End of program
    logger.info("End of MNIST detector generation script")


# Main call ---------------------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    _run_command(**vars(args))
    sys.exit(0)  # Shell return 0 == success
