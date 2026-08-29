"""Small data objects shared by solver backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Control:
    thrust: float
    body_rate: np.ndarray

    @classmethod
    def from_array(cls, value: np.ndarray) -> "Control":
        value = np.asarray(value, dtype=float)
        if value.shape != (4,):
            raise ValueError(f"control must have shape (4,), got {value.shape}")
        return cls(float(value[0]), value[1:4].copy())

    def as_array(self) -> np.ndarray:
        return np.r_[self.thrust, self.body_rate]


@dataclass(frozen=True)
class Reference:
    """One horizon reference: N+1 states and N feed-forward controls."""

    states: np.ndarray
    controls: np.ndarray

    @property
    def feedforward_controls(self) -> np.ndarray:
        """Inverse-dynamics nominal controls used by the control cost."""
        return self.controls

    def validate(self, horizon_steps: int) -> None:
        if self.states.shape != (horizon_steps + 1, 13):
            raise ValueError(
                f"reference states must have shape ({horizon_steps + 1}, 13), "
                f"got {self.states.shape}"
            )
        if self.controls.shape != (horizon_steps, 4):
            raise ValueError(
                f"reference controls must have shape ({horizon_steps}, 4), "
                f"got {self.controls.shape}"
            )
        if not np.all(np.isfinite(self.states)) or not np.all(np.isfinite(self.controls)):
            raise ValueError("reference contains a non-finite value")
