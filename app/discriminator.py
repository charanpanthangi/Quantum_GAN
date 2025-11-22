"""Classical discriminator network built with PyTorch.

The discriminator plays the role of a "judge" in the GAN game. It looks
at numbers and predicts whether they likely came from the real dataset or
were invented by the quantum generator. A tiny multilayer perceptron is
sufficient for this toy example.
"""

from __future__ import annotations

import torch
from torch import nn


class Discriminator(nn.Module):
    """Simple feed-forward network for binary classification."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability that each input sample is real."""

        return self.net(x)


def bce_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary cross entropy helper with stable defaults."""

    return nn.functional.binary_cross_entropy(predictions, targets)


def create_discriminator() -> Discriminator:
    """Factory for the discriminator module."""

    return Discriminator()


__all__ = ["Discriminator", "bce_loss", "create_discriminator"]
