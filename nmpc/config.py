"""Typed configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class ModelConfig:
    mass: float
    gravity: float
    rate_tau: float = 0.15


@dataclass(frozen=True)
class ControllerConfig:
    sample_time: float
    control_period: float
    horizon_steps: int
    reference_timeout: float
    qp_solver_cond_N: int
    nlp_solver_type: str
    integrator_type: str
    levenberg_marquardt: float
    warm_start: bool = True


@dataclass(frozen=True)
class CostScaleConfig:
    """Engineering error scales used to make every cost residual dimensionless."""

    weight_factor: float
    position_error_m: np.ndarray
    velocity_error_m_s: np.ndarray
    attitude_error_deg: np.ndarray
    thrust_correction_n: float
    body_rate_correction_deg_s: np.ndarray

    @property
    def state_weights(self) -> np.ndarray:
        """Physical-coordinate Q diagonal with the configured global factor."""
        state_scales = np.r_[
            self.position_error_m,
            self.velocity_error_m_s,
            np.deg2rad(self.attitude_error_deg),
        ]
        return self.weight_factor / np.square(state_scales)

    @property
    def control_weights(self) -> np.ndarray:
        """Physical-coordinate R diagonal with the configured global factor."""
        control_scales = np.r_[
            self.thrust_correction_n,
            np.deg2rad(self.body_rate_correction_deg_s),
        ]
        return self.weight_factor / np.square(control_scales)


@dataclass(frozen=True)
class LimitConfig:
    thrust_min: float
    thrust_max: float
    body_rate_max: np.ndarray
    horizontal_speed_max: float
    vertical_speed_max_up: float
    vertical_speed_max_down: float
    horizontal_acceleration_max: float
    vertical_acceleration_max_up: float
    vertical_acceleration_max_down: float
    jerk_max: float
    jerk_consistency_tolerance: float
    tilt_max_deg: float


@dataclass(frozen=True)
class CodeGenerationConfig:
    model_name: str
    json_file: str
    code_export_directory: str


@dataclass(frozen=True)
class Px4Config:
    hover_throttle: float
    throttle_min: float
    throttle_max: float


@dataclass(frozen=True)
class ManualControlConfig:
    deadzone: float
    timeout: float
    max_horizontal_speed: float
    max_vertical_speed: float
    max_yaw_rate: float
    max_horizontal_acceleration: float
    max_vertical_acceleration: float
    max_yaw_acceleration: float
    max_horizontal_position_lead: float
    max_vertical_position_lead: float


@dataclass(frozen=True)
class EsoConfig:
    enabled: bool
    bandwidth_rad_s: float
    disturbance_clamp_m_s2: float
    activation_delay_s: float
    innovation_limit_m_s: float


@dataclass(frozen=True)
class NmpcConfig:
    model: ModelConfig
    controller: ControllerConfig
    cost_scales: CostScaleConfig
    limits: LimitConfig
    code_generation: CodeGenerationConfig
    px4: Px4Config
    manual_control: ManualControlConfig
    eso: EsoConfig

    @property
    def hover_thrust(self) -> float:
        return self.model.mass * self.model.gravity


def _vector(section: dict[str, Any], key: str, size: int) -> np.ndarray:
    value = np.asarray(section[key], dtype=float)
    if value.shape != (size,):
        raise ValueError(f"{key} must contain {size} values, got shape {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{key} contains a non-finite value")
    return value


def load_config(path: str | Path = "config/nmpc.yaml") -> NmpcConfig:
    """Load and validate the controller YAML file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    model = ModelConfig(**raw["model"])
    controller = ControllerConfig(**raw["controller"])
    scales_raw = raw["cost_scales"]
    cost_scales = CostScaleConfig(
        weight_factor=float(scales_raw.get("weight_factor", 1.0)),
        position_error_m=_vector(scales_raw, "position_error_m", 3),
        velocity_error_m_s=_vector(scales_raw, "velocity_error_m_s", 3),
        attitude_error_deg=_vector(scales_raw, "attitude_error_deg", 3),
        thrust_correction_n=float(scales_raw["thrust_correction_n"]),
        body_rate_correction_deg_s=_vector(
            scales_raw, "body_rate_correction_deg_s", 3
        ),
    )
    limits_raw = raw["limits"]
    limits = LimitConfig(
        thrust_min=float(limits_raw["thrust_min"]),
        thrust_max=float(limits_raw["thrust_max"]),
        body_rate_max=_vector(limits_raw, "body_rate_max", 3),
        horizontal_speed_max=float(limits_raw["horizontal_speed_max"]),
        vertical_speed_max_up=float(limits_raw["vertical_speed_max_up"]),
        vertical_speed_max_down=float(limits_raw["vertical_speed_max_down"]),
        horizontal_acceleration_max=float(limits_raw["horizontal_acceleration_max"]),
        vertical_acceleration_max_up=float(limits_raw["vertical_acceleration_max_up"]),
        vertical_acceleration_max_down=float(limits_raw["vertical_acceleration_max_down"]),
        jerk_max=float(limits_raw["jerk_max"]),
        jerk_consistency_tolerance=float(limits_raw["jerk_consistency_tolerance"]),
        tilt_max_deg=float(limits_raw["tilt_max_deg"]),
    )
    code_generation = CodeGenerationConfig(**raw["code_generation"])
    px4 = Px4Config(**raw["px4"])
    manual_control = ManualControlConfig(**raw["manual_control"])
    eso = EsoConfig(**raw.get("eso", {
        "enabled": False,
        "bandwidth_rad_s": 2.5,
        "disturbance_clamp_m_s2": 1.0,
        "activation_delay_s": 1.0,
        "innovation_limit_m_s": 1.0,
    }))
    config = NmpcConfig(
        model, controller, cost_scales, limits, code_generation, px4, manual_control, eso
    )
    _validate(config)
    return config


def _validate(config: NmpcConfig) -> None:
    if config.model.mass <= 0.0 or config.model.gravity <= 0.0:
        raise ValueError("mass and gravity must be positive")
    if config.controller.sample_time <= 0.0:
        raise ValueError("sample_time must be positive")
    if config.controller.control_period <= 0.0:
        raise ValueError("control_period must be positive")
    # The current baseline runs both the 100 Hz control loop and the MPC/
    # trajectory grid at sample_time=0.01 s.
    if config.controller.horizon_steps < 2:
        raise ValueError("horizon_steps must be at least 2")
    if config.controller.reference_timeout <= config.controller.sample_time:
        raise ValueError("reference_timeout must exceed sample_time")
    eso = config.eso
    if not isinstance(eso.enabled, bool):
        raise ValueError("eso.enabled must be boolean")
    if not np.isfinite(eso.bandwidth_rad_s) or eso.bandwidth_rad_s <= 0.0:
        raise ValueError("eso.bandwidth_rad_s must be finite and positive")
    if not np.isfinite(eso.disturbance_clamp_m_s2) or eso.disturbance_clamp_m_s2 <= 0.0:
        raise ValueError("eso.disturbance_clamp_m_s2 must be finite and positive")
    if not np.isfinite(eso.activation_delay_s) or eso.activation_delay_s < 0.0:
        raise ValueError("eso.activation_delay_s must be finite and non-negative")
    if not np.isfinite(eso.innovation_limit_m_s) or eso.innovation_limit_m_s <= 0.0:
        raise ValueError("eso.innovation_limit_m_s must be finite and positive")
    if not 1 <= config.controller.qp_solver_cond_N <= config.controller.horizon_steps:
        raise ValueError("qp_solver_cond_N must lie in [1, horizon_steps]")
    if config.limits.thrust_min < 0.0:
        raise ValueError("thrust_min cannot be negative")
    if config.limits.thrust_max <= config.limits.thrust_min:
        raise ValueError("thrust_max must exceed thrust_min")
    if config.hover_thrust > config.limits.thrust_max:
        raise ValueError("hover thrust is above thrust_max")
    if np.any(config.limits.body_rate_max <= 0.0):
        raise ValueError("body_rate_max values must be positive")
    if config.limits.horizontal_speed_max <= 0.0:
        raise ValueError("horizontal_speed_max must be positive")
    if config.limits.vertical_speed_max_up <= 0.0:
        raise ValueError("vertical_speed_max_up must be positive")
    if config.limits.vertical_speed_max_down <= 0.0:
        raise ValueError("vertical_speed_max_down must be positive")
    trajectory_limits = np.array(
        [
            config.limits.horizontal_acceleration_max,
            config.limits.vertical_acceleration_max_up,
            config.limits.vertical_acceleration_max_down,
            config.limits.jerk_max,
            config.limits.jerk_consistency_tolerance,
        ]
    )
    if np.any(trajectory_limits <= 0.0):
        raise ValueError("trajectory acceleration and jerk limits must be positive")
    if not 0.0 < config.limits.tilt_max_deg < 90.0:
        raise ValueError("tilt_max_deg must lie in (0, 90) degrees")
    if not 0.0 <= config.px4.throttle_min < config.px4.hover_throttle:
        raise ValueError("PX4 throttle_min must lie below hover_throttle")
    if not config.px4.hover_throttle < config.px4.throttle_max <= 1.0:
        raise ValueError("PX4 throttle_max must lie above hover_throttle and at most 1")
    manual = config.manual_control
    if not 0.0 <= manual.deadzone < 1.0:
        raise ValueError("manual-control deadzone must lie in [0, 1)")
    if manual.timeout <= 0.0:
        raise ValueError("manual-control timeout must be positive")
    manual_positive_values = np.array(
        [
            manual.max_horizontal_speed,
            manual.max_vertical_speed,
            manual.max_yaw_rate,
            manual.max_horizontal_acceleration,
            manual.max_vertical_acceleration,
            manual.max_yaw_acceleration,
            manual.max_horizontal_position_lead,
            manual.max_vertical_position_lead,
        ]
    )
    if np.any(manual_positive_values <= 0.0):
        raise ValueError("manual-control limits must be positive")
    all_cost_scales = np.concatenate(
        (
            config.cost_scales.position_error_m,
            config.cost_scales.velocity_error_m_s,
            config.cost_scales.attitude_error_deg,
            [config.cost_scales.thrust_correction_n],
            config.cost_scales.body_rate_correction_deg_s,
        )
    )
    if not np.all(np.isfinite(all_cost_scales)):
        raise ValueError("cost scales must be finite")
    if np.any(all_cost_scales <= 0.0):
        raise ValueError("cost scales must be positive")
    if not np.isfinite(config.cost_scales.weight_factor):
        raise ValueError("cost weight_factor must be finite")
    if config.cost_scales.weight_factor <= 0.0:
        raise ValueError("cost weight_factor must be positive")
