# NMPC SITL baseline

| Case | Interface | Result | Position RMSE (m) | Velocity RMSE (m/s) | Attitude RMSE (rad) | rx→pub P50 (ms) | rx→pub P95 (ms) | rx→pub P99 (ms) | Trajectory plot |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| hover | direct | PASS | 0.0298 | 0.0137 | 0.0026 | 16.716 | 23.099 | 27.116 | [PNG](hover/trajectory.png) |
| point_1m | direct | PASS | 0.0485 | 0.0323 | 0.0083 | 17.689 | 25.521 | 32.054 | [PNG](point_1m/trajectory.png) |
| circle | direct | PASS | 0.0920 | 0.0567 | 0.0135 | 19.230 | 27.283 | 34.628 | [PNG](circle/trajectory.png) |
| figure8 | direct | PASS | 0.1446 | 0.1006 | 0.0147 | 17.346 | 24.578 | 29.795 | [PNG](figure8/trajectory.png) |

Machine-readable report: [JSON](../../json/baseline_runs_vectorized_full_regression_20260901_suite_summary.json)

Combined trajectory plot: [PNG](trajectory_suite_long.png)
