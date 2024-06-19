"""Package to create datasets, pipelines and other utilities.

You can use this module for example, to write all the functions needed to
operate the methods defined at `__init__.py` or in your scripts.

All functions here are optional and you can add or remove them as you need.

The following code is based on the drift detector example from:
https://frouros.readthedocs.io/en/latest/examples/data_drift/MMD_advance.html
"""

import logging
from functools import partial

import frouros.callbacks
import frouros.utils
import numpy as np
import scipy as sp
import torch
import torchvision
from frouros.detectors import concept_drift, data_drift

from drift_detector_mnist import config

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


def create_detector(
    name, autoencoder, reference_dataset, num_permutations=5000
):
    """Main/public method to create a detector"""
    logger.debug("Load the autoencoder model from disk")
    autoencoder = torch.load(autoencoder)  # Torch model

    logger.debug("Load the reference data and encode it")
    reference_dataset = torch.load(reference_dataset)
    X_ref_sample = np.array([X.tolist() for X, _ in reference_dataset])
    X_ref_sample = X_ref_sample.astype(np.float32)
    X_ref_encoded = autoencoder.encoder(torch.Tensor(X_ref_sample)).numpy()

    logger.debug("Define the detector kernel and parameters")
    pdist = sp.spatial.distance.pdist(X=X_ref_encoded, metric="euclidean")
    sigma = np.median(pdist)
    kernel = partial(frouros.utils.kernels.rbf_kernel, sigma=sigma)

    logger.debug("Create a MMD detector with permutation test")
    detector = data_drift.MMD(
        kernel,
        callbacks=[
            frouros.callbacks.PermutationTestDistanceBased(
                num_permutations,
                # random_state=seed,
                num_jobs=-1,
                method="exact",
                name="permutation_test",
                verbose=False,
            ),
        ],
    )

    logger.debug("Fit the detector with the reference data")
    fit_logs = detector.fit(X=X_ref_encoded)
    logger.debug(f"Fit logs: {fit_logs}")

    logger.debug("Save the detector to disk")
    detector.save(filepath=f"{name}.")
