"""SVG-only plotting utilities.

This module creates lightweight vector graphics for use inside GitHub and
the CODEX interface. SVG files are text-based, so they are friendly to
version control and easy to preview without downloading binary assets.
"""

from __future__ import annotations

import os
from typing import Iterable, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Force Matplotlib to prefer SVG output.
matplotlib.use("Agg")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_distributions(real: np.ndarray, generated: np.ndarray, title: str, output_path: str) -> None:
    """Plot real vs generated samples as overlapping histograms."""

    _ensure_dir(os.path.dirname(output_path) or ".")
    plt.figure(figsize=(4, 3))
    bins = np.linspace(0, 1, 20)
    plt.hist(real, bins=bins, alpha=0.6, label="Real", color="#4c72b0")
    plt.hist(generated, bins=bins, alpha=0.6, label="Generated", color="#dd8452")
    plt.xlabel("Value (normalized)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_losses(generator_losses: Iterable[float], discriminator_losses: Iterable[float], output_path: str) -> None:
    """Plot generator and discriminator loss curves."""

    _ensure_dir(os.path.dirname(output_path) or ".")
    plt.figure(figsize=(4, 3))
    plt.plot(generator_losses, label="Generator", color="#dd8452")
    plt.plot(discriminator_losses, label="Discriminator", color="#4c72b0")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross entropy")
    plt.title("Training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


__all__ = ["plot_distributions", "plot_losses"]
