#!/usr/bin/env python3
"""Generate the acados C solver from config/nmpc.yaml."""

from __future__ import annotations

import argparse
from dataclasses import replace

from nmpc.config import load_config
from nmpc.solver.acados_solver import generate_solver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/nmpc.yaml")
    parser.add_argument("--generate-only", action="store_true", help="skip native compilation")
    parser.add_argument("--sample-time", type=float, help="override prediction grid spacing")
    parser.add_argument("--control-period", type=float, help="override nominal control period")
    parser.add_argument("--model-name", help="override generated model/symbol prefix")
    parser.add_argument("--code-export-directory", help="override generated solver directory")
    parser.add_argument("--json-file", help="override acados JSON basename")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.sample_time is not None or args.control_period is not None:
        sample_time = args.sample_time if args.sample_time is not None else config.controller.sample_time
        control_period = args.control_period if args.control_period is not None else config.controller.control_period
        if sample_time <= 0.0 or control_period < sample_time:
            parser.error("sample/control periods must be positive and control-period >= sample-time")
        config = replace(
            config,
            controller=replace(
                config.controller,
                sample_time=sample_time,
                control_period=control_period,
            ),
        )
    if args.model_name or args.code_export_directory or args.json_file:
        config = replace(
            config,
            code_generation=replace(
                config.code_generation,
                model_name=args.model_name or config.code_generation.model_name,
                code_export_directory=args.code_export_directory or config.code_generation.code_export_directory,
                json_file=args.json_file or config.code_generation.json_file,
            ),
        )
    generate_solver(config, build=not args.generate_only)
    action = "generated" if args.generate_only else "generated and built"
    print(f"{config.code_generation.model_name}: {action}")


if __name__ == "__main__":
    main()
