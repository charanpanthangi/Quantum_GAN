"""Command-line interface for training the QGAN demo.

Running this script will train the quantum generator and classical
discriminator on a simple Gaussian dataset. All visual outputs are stored
as SVG files so they preview cleanly on GitHub and in the CODEX
interface.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .dataset import normalize, sample_real_distribution
from .plots import plot_distributions, plot_losses
from .qgan import train_qgan
from .quantum_generator import build_qnode, generate_samples, initialize_weights


def kl_divergence_approx(p: np.ndarray, q: np.ndarray, bins: int = 30) -> float:
    """Approximate KL divergence using histogram bins.

    This is a quick, beginner-friendly approximation that avoids complex
    math. It simply compares the binned probabilities of the two arrays.
    """

    hist_p, edges = np.histogram(p, bins=bins, range=(0, 1), density=True)
    hist_q, _ = np.histogram(q, bins=edges, density=True)

    # Add a tiny epsilon to avoid log(0).
    epsilon = 1e-8
    hist_p = hist_p + epsilon
    hist_q = hist_q + epsilon
    return float(np.sum(hist_p * np.log(hist_p / hist_q)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple Quantum GAN with SVG outputs.")
    parser.add_argument("--n_samples", type=int, default=256, help="Number of real samples to draw")
    parser.add_argument("--n_epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="examples", help="Where to save SVG plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load real data and normalize to [0, 1] for plotting.
    real_samples = sample_real_distribution(args.n_samples)
    real_normalized = normalize(real_samples)

    # Create an initial generator snapshot for visualization before training.
    qnode = build_qnode()
    initial_weights = initialize_weights()
    with torch.no_grad():
        initial_generated = generate_samples(initial_weights, qnode, args.n_samples).squeeze(-1).cpu().numpy()
    plot_distributions(
        real_normalized,
        initial_generated,
        "Real vs Generated (initial)",
        os.path.join(args.output_dir, "real_vs_generated_initial.svg"),
    )

    # 2) Train the QGAN.
    result = train_qgan(
        real_samples=real_normalized,
        n_epochs=args.n_epochs,
        lr=args.learning_rate,
        generated_points=args.n_samples,
    )

    generated = result.generated_samples

    # 3) Save plots before/after training.
    plot_distributions(
        real_normalized,
        generated,
        "Real vs Generated (final)",
        os.path.join(args.output_dir, "real_vs_generated_final.svg"),
    )
    plot_losses(result.generator_losses, result.discriminator_losses, os.path.join(args.output_dir, "training_loss.svg"))

    # 4) Compute a simple divergence metric to print progress.
    divergence = kl_divergence_approx(real_normalized, generated)
    print(f"Approximate KL divergence between real and generated: {divergence:.4f}")
    print(f"Training completed. SVG plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
