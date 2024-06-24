"""Module for utility functions used across the AI-model package."""

import torch
from torch import nn


def contractive_loss(outputs_e, outputs, inputs, lambda_=1e-4):
    """Compute the contractive loss for the autoencoder."""
    if outputs.shape != inputs.shape:
        raise ValueError(
            f"outputs.shape : {outputs.shape} != inputs.shape : {inputs.shape}"
        )
    criterion = nn.MSELoss()
    loss1 = criterion(outputs, inputs)

    ones = torch.ones(outputs_e.size()).to(outputs_e.device)
    outputs_e.backward(ones, retain_graph=True)
    loss2 = torch.sqrt(torch.sum(torch.pow(inputs.grad, 2)))
    inputs.grad.data.zero_()

    loss = loss1 + (lambda_ * loss2)
    return loss
