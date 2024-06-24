"""Detector module for drift detection of MNIST data."""

import logging
import pathlib

import torch
import numpy as np

from detector import config

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


def warm(**kwargs):
    """Main/public method to start up the model"""
    # if necessary, start the model
    pass


def predict(detector_name, input_file):
    """Performs drift detection on data using a MNIST detector.

    Arguments:
        model_name -- Model name to use for detection.
        input_file -- tp file with images equivalent to MNIST data.

    Returns:
        Return value from torch detector model compare.
    """
    autoencoder_uri = pathlib.Path(config.MODELS_PATH, "mnist_autoencoder.pt")
    detector_uri = pathlib.Path(config.MODELS_PATH, detector_name)
    logger.debug("Loading autoencoder model from uri: %s", autoencoder_uri)
    autoencoder = torch.load(autoencoder_uri)
    logger.debug("Loading model from uri: %s", detector_uri)
    detector_model = torch.load(detector_uri)
    logger.debug("Loading data from input_file: %s", input_file)
    sample_images = np.load(input_file)
    logger.debug("Encode the sample images using the autoencoder")
    with torch.no_grad():
        sample_images = torch.tensor(sample_images).float()
        sample_encoded = autoencoder.encoder.forward(sample_images)
        sample_encoded = sample_encoded.cpu().numpy()
    logger.debug("Run detector using sample file")
    mmd, callbacks_logs = detector_model.compare(X=sample_encoded)
    return {"MMD": mmd.distance, **callbacks_logs}
