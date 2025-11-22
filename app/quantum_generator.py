"""Quantum generator built with PennyLane.

This module defines a tiny parameterized quantum circuit (PQC) that turns
trainable rotation angles into a simple one-dimensional probability
value. The circuit can be viewed as an "artist" that paints numbers while
learning to match a real distribution observed by the discriminator.
"""

from __future__ import annotations

from typing import Iterable
import pennylane as qml
import torch


def create_device(n_qubits: int = 1) -> qml.Device:
    """Create a PennyLane device.

    A default.qubit simulator is enough to run quick examples locally.
    Keeping ``shots=None`` makes the circuit return exact expectation
    values, which are differentiable by PyTorch.
    """

    return qml.device("default.qubit", wires=n_qubits, shots=None)


def _layer(weights: Iterable[float], wires: Iterable[int]) -> None:
    """Apply a single rotation layer.

    Each qubit receives three simple rotations. These gates tilt the
    qubit's state on the Bloch sphere, changing the probability of
    measuring ``|0>`` or ``|1>``. When more than one qubit is present, we
    add a CNOT chain to lightly entangle them, showing how circuits can
    share information.
    """

    for wire, (rx, ry, rz) in zip(wires, weights):
        qml.RX(rx, wires=wire)
        qml.RY(ry, wires=wire)
        qml.RZ(rz, wires=wire)

    # Optional entanglement for two or more qubits.
    if len(wires) > 1:
        for control, target in zip(wires[:-1], wires[1:]):
            qml.CNOT(wires=[control, target])


def build_qnode(n_layers: int = 2, n_qubits: int = 1) -> qml.QNode:
    """Construct a QNode that maps weights to an expectation value.

    Args:
        n_layers: How many rotation blocks to apply.
        n_qubits: Number of qubits. One qubit is enough for this demo.

    Returns:
        A PennyLane QNode that accepts a tensor of ``weights`` with shape
        ``(n_layers, n_qubits, 3)`` and outputs a single expectation
        value in ``[-1, 1]``.
    """

    dev = create_device(n_qubits=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(weights):
        wires = list(range(n_qubits))
        for layer_weights in weights:
            _layer(layer_weights, wires=wires)
        return qml.expval(qml.PauliZ(0))

    return circuit


def quantum_generator(weights: torch.Tensor, qnode: qml.QNode) -> torch.Tensor:
    """Convert circuit output to a value in the ``[0, 1]`` interval.

    Args:
        weights: Trainable parameters for the circuit.
        qnode: Constructed QNode from :func:`build_qnode`.

    Returns:
        A torch scalar tensor representing a single generated sample.
    """

    expectation = qnode(weights)
    # Map from [-1, 1] to [0, 1] so it resembles a probability value.
    scaled_sample = (expectation + 1.0) / 2.0
    return scaled_sample


def generate_samples(weights: torch.Tensor, qnode: qml.QNode, n_samples: int) -> torch.Tensor:
    """Generate a batch of samples by evaluating the circuit repeatedly.

    Args:
        weights: Trainable parameters.
        qnode: QNode to evaluate.
        n_samples: Number of synthetic data points to produce.

    Returns:
        Torch tensor of shape ``(n_samples, 1)`` containing values in
        ``[0, 1]``.
    """

    outputs = []
    for _ in range(n_samples):
        outputs.append(quantum_generator(weights, qnode))
    return torch.stack(outputs).unsqueeze(-1)


def initialize_weights(n_layers: int = 2, n_qubits: int = 1) -> torch.Tensor:
    """Create small random weights for the generator circuit.

    Weights are initialized near zero so the circuit starts close to the
    ``|0>`` state. This keeps early gradients stable and makes it easy to
    see the model learning over time.
    """

    return 0.01 * torch.randn((n_layers, n_qubits, 3), requires_grad=True)


__all__ = [
    "create_device",
    "build_qnode",
    "quantum_generator",
    "generate_samples",
    "initialize_weights",
]
