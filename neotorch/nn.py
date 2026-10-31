"""Eenvoudige neurale bouwstenen voor NeoTorch."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from .tensor import Tensor, randn


class Module:
    """Basisklasse voor NeoTorch-lagen."""

    def parameters(self) -> List[Tensor]:
        return []

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, seed: int | None = None):
        scale = np.sqrt(2.0 / max(in_features, 1))
        self.weight = randn(out_features, in_features, seed=seed) * scale
        bias_seed = None if seed is None else seed + 1
        self.bias = randn(out_features, seed=bias_seed) * 0.01

    def parameters(self) -> List[Tensor]:
        return [self.weight, self.bias]

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., in)  weight: (out, in)
        return Tensor(x.data @ self.weight.data.T + self.bias.data)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.maximum(0.0, x.data))


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.tanh(x.data))


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(1.0 / (1.0 + np.exp(-np.clip(x.data, -40, 40))))


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
