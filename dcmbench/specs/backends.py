"""
Array backends (numpy, torch) for evaluating specs and choice probabilities.

The same likelihood code runs on both: numpy for predictions and marginal
utilities, torch for gradient-based estimation.
"""

from __future__ import annotations

import numpy as np


class NumpyBackend:
    name = "numpy"
    log = staticmethod(np.log)
    exp = staticmethod(np.exp)

    @staticmethod
    def indicator(condition):
        return np.asarray(condition, dtype=float)

    @staticmethod
    def asarray(values):
        return np.asarray(values, dtype=float)

    @staticmethod
    def stack(arrays, axis):
        return np.stack(arrays, axis=axis)

    @staticmethod
    def where(condition, a, b):
        return np.where(condition, a, b)

    @staticmethod
    def amax(x, axis):
        return np.max(x, axis=axis, keepdims=True)

    @staticmethod
    def total(x, axis):
        return np.sum(x, axis=axis, keepdims=True)

    @staticmethod
    def broadcast(x, shape):
        return np.broadcast_to(np.asarray(x, dtype=float), shape)

    @staticmethod
    def detach(x):
        return x

    @staticmethod
    def index(positions):
        return np.asarray(positions)


class TorchBackend:
    name = "torch"

    def __init__(self):
        import torch

        self.torch = torch

    def log(self, x):
        return self.torch.log(x)

    def exp(self, x):
        return self.torch.exp(x)

    def indicator(self, condition):
        return condition.to(self.torch.float64) if hasattr(condition, "to") else float(condition)

    def asarray(self, values):
        return self.torch.as_tensor(np.asarray(values, dtype=float), dtype=self.torch.float64)

    def stack(self, arrays, axis):
        return self.torch.stack([self._tensor(a) for a in arrays], dim=axis)

    def where(self, condition, a, b):
        return self.torch.where(self._tensor_bool(condition), self._tensor(a), self._tensor(b))

    def amax(self, x, axis):
        return self.torch.amax(x, dim=axis, keepdim=True)

    def total(self, x, axis):
        return self.torch.sum(x, dim=axis, keepdim=True)

    def broadcast(self, x, shape):
        return self._tensor(x).expand(*shape)

    def detach(self, x):
        return x.detach()

    def index(self, positions):
        return self.torch.as_tensor(np.asarray(positions), dtype=self.torch.long)

    def _tensor(self, x):
        if isinstance(x, self.torch.Tensor):
            return x
        return self.torch.as_tensor(x, dtype=self.torch.float64)

    def _tensor_bool(self, x):
        if isinstance(x, self.torch.Tensor):
            return x.bool()
        return self.torch.as_tensor(np.asarray(x, dtype=bool))
