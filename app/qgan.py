"""Training loop that combines the quantum generator and classical discriminator.

The adversarial process works like a game:
- The discriminator acts as a judge, learning to label real samples as 1 and
  fake samples as 0.
- The quantum generator acts as an artist, tweaking its circuit parameters so
  the discriminator believes its outputs are real.

Alternating updates gradually push both models to improve until the generated
and real distributions align.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import optim

from .discriminator import bce_loss, create_discriminator
from .quantum_generator import build_qnode, generate_samples, initialize_weights


@dataclass
class TrainingResult:
    """Container for QGAN training outputs."""

    generator_weights: torch.Tensor
    discriminator: torch.nn.Module
    generator_losses: List[float]
    discriminator_losses: List[float]
    generated_samples: np.ndarray


def prepare_batch(real_samples: np.ndarray, batch_size: int) -> torch.Tensor:
    """Randomly select a mini-batch of real data and convert to tensor."""

    indices = np.random.choice(len(real_samples), size=batch_size, replace=True)
    batch = real_samples[indices].astype(np.float32)
    return torch.from_numpy(batch).unsqueeze(-1)


def train_qgan(
    real_samples: np.ndarray,
    n_epochs: int = 200,
    batch_size: int = 32,
    lr: float = 0.01,
    n_layers: int = 2,
    n_qubits: int = 1,
    generated_points: int | None = None,
) -> TrainingResult:
    """Run the adversarial QGAN training loop.

    Args:
        real_samples: Array of real data points.
        n_epochs: Number of passes over the data.
        batch_size: Mini-batch size for discriminator training.
        lr: Learning rate shared by both optimizers for simplicity.
        n_layers: How many rotation layers in the quantum circuit.
        n_qubits: Number of qubits for the generator.
        generated_points: Optional number of samples to return after training.

    Returns:
        TrainingResult containing trained objects and diagnostic curves.
    """

    # Build models
    qnode = build_qnode(n_layers=n_layers, n_qubits=n_qubits)
    generator_weights = initialize_weights(n_layers=n_layers, n_qubits=n_qubits)
    discriminator = create_discriminator()

    # Set up optimizers for both players.
    optim_g = optim.Adam([generator_weights], lr=lr)
    optim_d = optim.Adam(discriminator.parameters(), lr=lr)

    generator_losses: List[float] = []
    discriminator_losses: List[float] = []

    for epoch in range(n_epochs):
        # -----------------------
        # 1) Train discriminator
        # -----------------------
        discriminator.train()
        optim_d.zero_grad()

        # Real data labeled as 1
        real_batch = prepare_batch(real_samples, batch_size)
        real_preds = discriminator(real_batch)
        real_labels = torch.ones_like(real_preds)
        loss_real = bce_loss(real_preds, real_labels)

        # Generated data labeled as 0
        fake_batch = generate_samples(generator_weights, qnode, batch_size).detach()
        fake_preds = discriminator(fake_batch)
        fake_labels = torch.zeros_like(fake_preds)
        loss_fake = bce_loss(fake_preds, fake_labels)

        loss_d = loss_real + loss_fake
        loss_d.backward()
        optim_d.step()

        discriminator_losses.append(float(loss_d.item()))

        # -------------------
        # 2) Train generator
        # -------------------
        optim_g.zero_grad()
        fake_batch_for_g = generate_samples(generator_weights, qnode, batch_size)
        preds = discriminator(fake_batch_for_g)
        target_labels = torch.ones_like(preds)
        loss_g = bce_loss(preds, target_labels)
        loss_g.backward()
        optim_g.step()

        generator_losses.append(float(loss_g.item()))

    # Produce a small set of generated samples for inspection.
    final_points = generated_points or len(real_samples)
    with torch.no_grad():
        gen_samples = generate_samples(generator_weights, qnode, final_points)
    gen_np = gen_samples.squeeze(-1).cpu().numpy()

    return TrainingResult(
        generator_weights=generator_weights.detach(),
        discriminator=discriminator,
        generator_losses=generator_losses,
        discriminator_losses=discriminator_losses,
        generated_samples=gen_np,
    )
