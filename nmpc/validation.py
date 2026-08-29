"""Quantitative prediction-residual analysis for the NMPC plant model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model.quadrotor import NX, QuadrotorModel, normalize_quaternion


@dataclass(frozen=True)
class ModelValidationSample:
    timestamp_us: int
    state: np.ndarray
    measured_body_rate: np.ndarray
    control: np.ndarray
    segment: str = "unknown"


def _vector_statistics(values: np.ndarray) -> dict[str, object]:
    if values.size == 0:
        return {
            "count": 0,
            "bias": [float("nan")] * 3,
            "rmse_per_axis": [float("nan")] * 3,
            "rmse_norm": float("nan"),
            "max_norm": float("nan"),
        }
    norms = np.linalg.norm(values, axis=1)
    return {
        "count": int(values.shape[0]),
        "bias": np.mean(values, axis=0).tolist(),
        "rmse_per_axis": np.sqrt(np.mean(values**2, axis=0)).tolist(),
        "rmse_norm": float(np.sqrt(np.mean(norms**2))),
        "max_norm": float(np.max(norms)),
    }


def _orientation_error_angle(predicted: np.ndarray, actual: np.ndarray) -> float:
    predicted = normalize_quaternion(predicted)
    actual = normalize_quaternion(actual)
    cosine = np.clip(abs(float(np.dot(predicted, actual))), 0.0, 1.0)
    return float(2.0 * np.arccos(cosine))


class ModelValidationRecorder:
    """Collect synchronized samples and evaluate open-loop prediction residuals."""

    def __init__(self, model: QuadrotorModel, maximum_interval: float = 0.1) -> None:
        if maximum_interval <= 0.0:
            raise ValueError("maximum_interval must be positive")
        self.model = model
        self.maximum_interval = float(maximum_interval)
        self.samples: list[ModelValidationSample] = []

    def add(
        self,
        timestamp_us: int,
        state: np.ndarray,
        measured_body_rate: np.ndarray,
        control: np.ndarray,
        segment: str = "unknown",
    ) -> None:
        state = np.asarray(state, dtype=float)
        measured_body_rate = np.asarray(measured_body_rate, dtype=float)
        control = np.asarray(control, dtype=float)
        if state.shape != (NX,) or measured_body_rate.shape != (3,) or control.shape != (4,):
            raise ValueError(f"validation sample shapes must be state={NX}, rate=3, control=4")
        if not all(np.all(np.isfinite(value)) for value in (state, measured_body_rate, control)):
            raise ValueError("validation sample contains a non-finite value")
        self.samples.append(
            ModelValidationSample(
                int(timestamp_us),
                state.copy(),
                measured_body_rate.copy(),
                control.copy(),
                str(segment),
            )
        )

    def _window_residuals(
        self, horizon: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        position_errors: list[np.ndarray] = []
        velocity_errors: list[np.ndarray] = []
        orientation_errors: list[float] = []
        elapsed_times: list[float] = []
        if horizon <= 0.0:
            raise ValueError("prediction horizon must be positive")

        for start_index in range(len(self.samples) - 1):
            predicted = self.samples[start_index].state.copy()
            elapsed = 0.0
            index = start_index
            valid = True
            while elapsed < horizon and index + 1 < len(self.samples):
                current = self.samples[index]
                following = self.samples[index + 1]
                time_step = (following.timestamp_us - current.timestamp_us) * 1.0e-6
                if time_step <= 0.0 or time_step > self.maximum_interval:
                    valid = False
                    break
                predicted = self.model.step_rk4(predicted, current.control, time_step)
                elapsed += time_step
                index += 1
            if not valid or elapsed < horizon:
                continue
            actual = self.samples[index].state
            position_errors.append(predicted[:3] - actual[:3])
            velocity_errors.append(predicted[3:6] - actual[3:6])
            orientation_errors.append(
                _orientation_error_angle(predicted[6:10], actual[6:10])
            )
            elapsed_times.append(elapsed)

        return (
            np.asarray(position_errors, dtype=float).reshape(-1, 3),
            np.asarray(velocity_errors, dtype=float).reshape(-1, 3),
            np.asarray(orientation_errors, dtype=float),
            np.asarray(elapsed_times, dtype=float),
        )

    def _delay_aligned_rate_statistics(
        self, maximum_delay: float = 0.4, delay_step: float = 0.002
    ) -> dict[str, object]:
        """Find each axis' command-to-measurement delay by minimum RMSE.

        Commands are linearly interpolated only inside continuous timestamp regions, so
        an odometry/timesync jump cannot be selected as an apparently useful delay.
        """
        timestamps = np.asarray([sample.timestamp_us for sample in self.samples], dtype=float) * 1e-6
        commands = np.asarray([sample.control[1:4] for sample in self.samples], dtype=float)
        measured = np.asarray([sample.measured_body_rate for sample in self.samples], dtype=float)
        delays = np.arange(0.0, maximum_delay + 0.5 * delay_step, delay_step)
        best_delays: list[float] = []
        aligned_errors: list[np.ndarray] = []
        counts: list[int] = []

        for axis in range(3):
            best_rmse = float("inf")
            best_delay = 0.0
            best_error = np.empty(0, dtype=float)
            for delay in delays:
                target_times = timestamps - delay
                right = np.searchsorted(timestamps, target_times, side="right")
                left = right - 1
                valid = (left >= 0) & (right < timestamps.size)
                valid_indices = np.flatnonzero(valid)
                if valid_indices.size == 0:
                    continue
                left_valid = left[valid_indices]
                right_valid = right[valid_indices]
                spans = timestamps[right_valid] - timestamps[left_valid]
                continuous = (spans > 0.0) & (spans <= self.maximum_interval)
                valid_indices = valid_indices[continuous]
                left_valid = left_valid[continuous]
                right_valid = right_valid[continuous]
                if valid_indices.size < 2:
                    continue
                spans = timestamps[right_valid] - timestamps[left_valid]
                fractions = (target_times[valid_indices] - timestamps[left_valid]) / spans
                interpolated = commands[left_valid, axis] + fractions * (
                    commands[right_valid, axis] - commands[left_valid, axis]
                )
                errors = interpolated - measured[valid_indices, axis]
                rmse = float(np.sqrt(np.mean(errors**2)))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_delay = float(delay)
                    best_error = errors
            best_delays.append(best_delay)
            aligned_errors.append(best_error)
            counts.append(int(best_error.size))

        rmse_per_axis = [
            float(np.sqrt(np.mean(error**2))) if error.size else float("nan")
            for error in aligned_errors
        ]
        return {
            "best_delay_s_per_axis": best_delays,
            "rmse_per_axis": rmse_per_axis,
            "count_per_axis": counts,
            "search_maximum_delay_s": float(maximum_delay),
            "search_step_s": float(delay_step),
        }

    def summary(self, horizons: tuple[float, ...] = (0.01, 0.1, 0.5)) -> dict[str, object]:
        if len(self.samples) < 2:
            raise ValueError("at least two validation samples are required")

        rate_errors: list[np.ndarray] = []
        acceleration_errors: list[np.ndarray] = []
        valid_intervals = 0
        rejected_intervals = 0
        for current, following in zip(self.samples[:-1], self.samples[1:]):
            time_step = (following.timestamp_us - current.timestamp_us) * 1.0e-6
            if time_step <= 0.0 or time_step > self.maximum_interval:
                rejected_intervals += 1
                continue
            valid_intervals += 1
            rate_errors.append(current.control[1:4] - following.measured_body_rate)
            model_acceleration = self.model.continuous_dynamics(
                current.state, current.control
            )[3:6]
            measured_acceleration = (
                following.state[3:6] - current.state[3:6]
            ) / time_step
            acceleration_errors.append(model_acceleration - measured_acceleration)

        prediction: dict[str, object] = {}
        for horizon in horizons:
            position, velocity, orientation, elapsed = self._window_residuals(horizon)
            key = f"{horizon:.3f}s"
            prediction[key] = {
                "position_error_m": _vector_statistics(position),
                "velocity_error_m_s": _vector_statistics(velocity),
                "orientation_error_rad": {
                    "count": int(orientation.size),
                    "rmse": (
                        float(np.sqrt(np.mean(orientation**2)))
                        if orientation.size
                        else float("nan")
                    ),
                    "max": float(np.max(orientation)) if orientation.size else float("nan"),
                },
                "actual_horizon_mean_s": (
                    float(np.mean(elapsed)) if elapsed.size else float("nan")
                ),
            }

        segments: dict[str, int] = {}
        for sample in self.samples:
            segments[sample.segment] = segments.get(sample.segment, 0) + 1
        return {
            "sample_count": len(self.samples),
            "valid_interval_count": valid_intervals,
            "rejected_interval_count": rejected_intervals,
            "segments": segments,
            "body_rate_tracking_error_rad_s": _vector_statistics(
                np.asarray(rate_errors, dtype=float).reshape(-1, 3)
            ),
            "delay_aligned_body_rate_tracking_error_rad_s": (
                self._delay_aligned_rate_statistics()
            ),
            "translational_acceleration_model_error_m_s2": _vector_statistics(
                np.asarray(acceleration_errors, dtype=float).reshape(-1, 3)
            ),
            "prediction": prediction,
        }
