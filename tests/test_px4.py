import numpy as np
import pytest

from nmpc.config import load_config
from nmpc.px4 import thrust_newton_to_px4


def test_hover_thrust_maps_to_negative_hover_throttle() -> None:
    config = load_config()

    assert thrust_newton_to_px4(config.hover_thrust, config) == pytest.approx(-0.3653)


def test_thrust_mapping_clamps_to_px4_bounds() -> None:
    config = load_config()

    assert thrust_newton_to_px4(0.0, config) == pytest.approx(-0.12)
    assert thrust_newton_to_px4(1.0e6, config) == pytest.approx(-1.0)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_thrust_mapping_rejects_non_finite_input(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        thrust_newton_to_px4(value, load_config())
