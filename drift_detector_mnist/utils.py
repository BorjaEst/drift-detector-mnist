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


def mkdata(data_filepath="data", batch_size=128):
    """Main/public function to run data processing to turn raw data
    from (data/raw) into cleaned data ready to be analyzed.

    """
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
