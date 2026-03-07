# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Custom activation functions to supplement fairseq's built-ins.
Swish (also known as SiLU): f(x) = x * sigmoid(x)
"""

import torch
import torch.nn.functional as F
from fairseq import utils as fairseq_utils


def swish(x: torch.Tensor) -> torch.Tensor:
    """Swish activation: f(x) = x * sigmoid(x). Equivalent to SiLU in PyTorch."""
    return x * torch.sigmoid(x)


def mish(x: torch.Tensor) -> torch.Tensor:
    """Mish activation: f(x) = x * tanh(softplus(x))."""
    return x * torch.tanh(F.softplus(x))


# Registry of custom activation functions not available in fairseq
CUSTOM_ACTIVATIONS = {
    "swish": swish,
    "silu": swish,  # SiLU is identical to Swish
    "mish": mish,
}

CUSTOM_ACTIVATION_NAMES = list(CUSTOM_ACTIVATIONS.keys())


def get_activation_fn(activation: str):
    """
    Returns an activation function by name.
    Extends fairseq's get_activation_fn with custom activations (e.g. swish).
    """
    if activation in CUSTOM_ACTIVATIONS:
        return CUSTOM_ACTIVATIONS[activation]
    return fairseq_utils.get_activation_fn(activation)


def get_available_activation_fns():
    """Returns all available activation function names (fairseq + custom)."""
    return fairseq_utils.get_available_activation_fns() + CUSTOM_ACTIVATION_NAMES
