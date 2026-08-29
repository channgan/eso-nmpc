#!/usr/bin/env python3
"""Generate the acados C solver from config/nmpc.yaml."""

from __future__ import annotations

import argparse

from nmpc.config import load_config
from nmpc.solver.acados_solver import generate_solver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/nmpc.yaml")
    parser.add_argument("--generate-only", action="store_true", help="skip native compilation")
    args = parser.parse_args()
    config = load_config(args.config)
    generate_solver(config, build=not args.generate_only)
    action = "generated" if args.generate_only else "generated and built"
    print(f"{config.code_generation.model_name}: {action}")


if __name__ == "__main__":
    main()
