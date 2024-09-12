"""Additional utility for Deep Learning models in PyTorch"""

# The hyper-sinh (HyperSinh) as a custom activation function in PyTorch

# Author: Luca Parisi <luca.parisi@ieee.org>

import torch
import torch.nn as nn
from src.constants import (GT_COEFFICIENT_PYTORCH,
                           LE_FIRST_COEFFICIENT_PYTORCH,
                           LE_SECOND_COEFFICIENT_PYTORCH)


class HyperSinh(nn.Module):  # pragma: no cover
    """
    A class defining the HyperSinh activation function in PyTorch.
    """

    def __init__(self) -> None:
        """
        Initialise the HyperSinh activation function.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Call the HyperSinh activation function.
        Args:
            x: torch.Tensor
                The input tensor.
        Returns:
                The output tensor (torch.Tensor) from the HyperSinh activation function.
        """
        return torch_hyper_sinh(x=x)


def torch_hyper_sinh(x: torch.Tensor) -> torch.Tensor:
    """
    Apply the hyper-sinh activation function to transform inputs accordingly.
    Args:
        x: torch.Tensor
            The input tensor to be transformed via the m-QReLU activation function.
    Returns:
            The transformed x (torch.Tensor) via the hyper-sinh.
    """
    return torch.where(
            x > 0,
            GT_COEFFICIENT_PYTORCH * torch.sinh(x),
            LE_FIRST_COEFFICIENT_PYTORCH * (x**LE_SECOND_COEFFICIENT_PYTORCH),
    )
