#!/usr/bin/env python3
"""
MATH-N V1 experiment driver.

Runs the complete pipeline described in the project README:

    generate/load dataset
        -> run each candidate generator (scipy, sympy) over every problem
        -> save full trajectories
        -> compute metrics
        -> run failure analysis
        -> write a single results/v1_report.json

Usage:
    python experiments/run_v1.py [--config configs/default.yaml] [--regenerate-dataset]

This script does not require any command-line arguments to run -- it is
meant to be the one-command entry point for the V1 benchmark described in
the project brief.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import yaml  # noqa: E402

import mathn.generators  # noqa: E402,F401
import mathn.problems  # noqa: E402,F401
from mathn.cli import build_generator, build_verifiers  # noqa: E402
from mathn.core.runner import RetryConfig, RetryController  # noqa: E402
from mathn.dataset import build_v1_dataset, load_dataset, save_dataset  # noqa: E402
from mathn.evaluation.failure_analysis import analyze_failures  # noqa: E402
from mathn.evaluation.metrics import compute_metrics  # noqa: E402
from mathn.logging.trajectory import load_trajectories, save_trajectory  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run the MATH-N V1 benchmark end to end.")
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "default.yaml"))
    parser.add_argument("--regenerate-dataset", action="store_true", help="rebuild datasets/v1 even if it exists")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_path = os.path.join(REPO_ROOT, config["dataset"]["path"])
    trajectories_root = os.path.join(REPO_ROOT, config["logging"]["trajectories_path"])
    results_path = os.path.join(REPO_ROOT, config["results"]["path"])
    max_attempts = config.get("max_attempts", 5)
    generator_names = config.get("generators", ["scipy", "sympy"])

    # 1. Dataset -----------------------------------------------------
    need_generate = args.regenerate_dataset or not os.path.isdir(dataset_path) or not os.listdir(dataset_path)
    if need_generate:
        print(f"Generating V1 dataset -> {dataset_path}")
        problems = build_v1_dataset(
            problems_per_family=config["dataset"]["problems_per_family"],
            base_seed=config["dataset"]["base_seed"],
        )
        save_dataset(problems, dataset_path)
    else:
        print(f"Loading existing V1 dataset from {dataset_path}")
        problems = load_dataset(dataset_path)
    print(f"Dataset size: {len(problems)} problems")
    by_family = {}
    for p in problems:
        by_family[p.family] = by_family.get(p.family, 0) + 1
    for family, count in sorted(by_family.items()):
        print(f"  {family}: {count}")

    retry_cfg = RetryConfig(
        max_attempts=max_attempts,
        generator_timeout_seconds=config["retry"]["generator_timeout_seconds"],
        verifier_timeout_seconds=config["retry"]["verifier_timeout_seconds"],
    )
    verifiers = build_verifiers(config)

    # 2. Run each generator over the full dataset ---------------------
    all_reports = {}
    wall_start = time.time()
    for gen_name in generator_names:
        print(f"\n=== Running generator: {gen_name} ===")
        generator = build_generator(gen_name)
        out_dir = os.path.join(trajectories_root, gen_name)
        os.makedirs(out_dir, exist_ok=True)

        trajectories = []
        n_verified = 0
        t0 = time.time()
        for problem in problems:
            verifier = verifiers[problem.family]
            controller = RetryController(generator, verifier, retry_cfg)
            traj = controller.run(problem)
            save_trajectory(traj, out_dir)
            trajectories.append(traj.to_dict())
            if traj.verified:
                n_verified += 1
        elapsed = time.time() - t0
        print(f"{gen_name}: {n_verified}/{len(problems)} verified in {elapsed:.2f}s "
              f"({len(problems) / elapsed:.1f} problems/s)")

        metrics = compute_metrics(trajectories)
        failures = analyze_failures(trajectories)
        all_reports[gen_name] = {"metrics": metrics, "failure_analysis": failures, "wall_time_seconds": elapsed}

    total_elapsed = time.time() - wall_start

    # 3. Save the combined report --------------------------------------
    os.makedirs(results_path, exist_ok=True)
    report_path = os.path.join(results_path, "v1_report.json")
    with open(report_path, "w") as f:
        json.dump(
            {
                "dataset_size": len(problems),
                "by_family": by_family,
                "max_attempts": max_attempts,
                "generators": all_reports,
                "total_wall_time_seconds": total_elapsed,
            },
            f,
            indent=2,
        )

    print(f"\nFull report written to {report_path}")
    print("\n=== Summary ===")
    for gen_name, report in all_reports.items():
        m = report["metrics"]
        print(
            f"{gen_name:10s}  final_success_rate={m['final_success_rate']:.2%}  "
            f"first_attempt_success_rate={m['first_attempt_success_rate']:.2%}  "
            f"avg_attempts={m['average_attempts']:.2f}"
        )


if __name__ == "__main__":
    main()
