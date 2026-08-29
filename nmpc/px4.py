"""PX4-specific command conversions kept outside the NMPC model."""

from __future__ import annotations

import numpy as np

from .config import NmpcConfig


def thrust_newton_to_px4(thrust_newton: float, config: NmpcConfig) -> float:
    """Convert total positive thrust in newtons to PX4 negative body-z demand."""
    thrust_newton = float(thrust_newton)
    if not np.isfinite(thrust_newton):
        raise ValueError("thrust_newton must be finite")
    normalized = config.px4.hover_throttle * thrust_newton / config.hover_thrust
    normalized = np.clip(
        normalized,
        config.px4.throttle_min,
        config.px4.throttle_max,
    )
    return -float(normalized)
