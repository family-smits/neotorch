"""Minimale Tensor-API, gestuurd door NumPy."""

from __future__ import annotations

from typing import Any, Iterable, Sequence, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], float, int]


class Tensor:
    """Lichte tensor-wrapper met basisoperaties."""

    def __init__(self, data: ArrayLike, dtype: Any = np.float32):
        self.data = np.asarray(data, dtype=dtype)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def numpy(self) -> np.ndarray:
        return self.data

    def clone(self) -> "Tensor":
        return Tensor(self.data.copy())

    def reshape(self, *shape: int) -> "Tensor":
        return Tensor(self.data.reshape(shape))

    def flatten(self) -> "Tensor":
        return Tensor(self.data.reshape(-1))

    def T(self) -> "Tensor":
        return Tensor(self.data.T)

    def __add__(self, other: Union["Tensor", float]) -> "Tensor":
        return Tensor(self.data + _as_array(other))

    def __radd__(self, other: Union["Tensor", float]) -> "Tensor":
        return Tensor(_as_array(other) + self.data)

    def __sub__(self, other: Union["Tensor", float]) -> "Tensor":
        return Tensor(self.data - _as_array(other))

    def __rsub__(self, other: Union["Tensor", float]) -> "Tensor":
        return Tensor(_as_array(other) - self.data)

    def __mul__(self, other: Union["Tensor", float]) -> "Tensor":
        return Tensor(self.data * _as_array(other))

    def __rmul__(self, other: Union["Tensor", float]) -> "Tensor":
        return Tensor(_as_array(other) * self.data)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return Tensor(self.data @ other.data)

    def __neg__(self) -> "Tensor":
        return Tensor(-self.data)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape})\n{self.data}"


def _as_array(value: Union[Tensor, float, int, np.ndarray]) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.data
    return np.asarray(value, dtype=np.float32)


def zeros(*shape: int, dtype: Any = np.float32) -> Tensor:
    return Tensor(np.zeros(shape, dtype=dtype))


def ones(*shape: int, dtype: Any = np.float32) -> Tensor:
    return Tensor(np.ones(shape, dtype=dtype))


def randn(*shape: int, seed: int | None = None, dtype: Any = np.float32) -> Tensor:
    rng = np.random.default_rng(seed)
    return Tensor(rng.standard_normal(shape).astype(dtype))


def from_numpy(array: np.ndarray) -> Tensor:
    return Tensor(array)
