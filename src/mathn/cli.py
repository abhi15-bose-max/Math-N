"""
MATH-N command-line interface.

    python -m mathn.cli generate-dataset --dataset datasets/v1
    python -m mathn.cli run --dataset datasets/v1 --generator scipy --max-attempts 5
    python -m mathn.cli evaluate --trajectories trajectories/scipy --out results/scipy_metrics.json
    python -m mathn.cli inspect --trajectory trajectories/scipy/mathn_ode_000001_000__scipy.json

Experiments are fully driven by the config file plus CLI flags -- no
source edits required to run a benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

# Ensure problem/generator registries are populated.
import mathn.generators  # noqa: F401
import mathn.problems  # noqa: F401
from mathn.core.runner import RetryConfig, RetryController
from mathn.dataset import build_v1_dataset, load_dataset, save_dataset
from mathn.evaluation.failure_analysis import analyze_failures
from mathn.evaluation.metrics import compute_metrics
from mathn.generators.scipy_generator import ScipyGenerator
from mathn.generators.sympy_generator import SympyGenerator
from mathn.logging.trajectory import load_trajectories, save_trajectory
from mathn.verifiers.ode_verifier import ODEVerifier
from mathn.verifiers.parameter_verifier import ParameterVerifier
from mathn.verifiers.root_verifier import RootVerifier
from mathn.verifiers.symbolic_verifier import SymbolicVerifier


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_generator(name: str):
    if name == "scipy":
        return ScipyGenerator()
    if name == "sympy":
        return SympyGenerator()
    raise ValueError(f"Unknown generator '{name}'. Available: scipy, sympy")


def build_verifiers(config: dict):
    v = config.get("verifiers", {})
    return {
        "ode": ODEVerifier(**v.get("ode", {})),
        "roots": RootVerifier(**v.get("roots", {})),
        "parameter_estimation": ParameterVerifier(**v.get("parameter_estimation", {})),
        "symbolic": SymbolicVerifier(**v.get("symbolic", {})),
    }


def cmd_generate_dataset(args):
    config = load_config(args.config)
    dcfg = config.get("dataset", {})
    n = args.problems_per_family or dcfg.get("problems_per_family", 10)
    seed = args.base_seed if args.base_seed is not None else dcfg.get("base_seed", 0)
    problems = build_v1_dataset(problems_per_family=n, base_seed=seed)
    paths = save_dataset(problems, args.dataset)
    print(f"Generated {len(problems)} problems -> {args.dataset} ({len(paths)} files)")
    by_family = {}
    for p in problems:
        by_family.setdefault(p.family, 0)
        by_family[p.family] += 1
    for family, count in sorted(by_family.items()):
        print(f"  {family}: {count}")


def cmd_run(args):
    config = load_config(args.config)
    max_attempts = args.max_attempts or config.get("max_attempts", 5)
    retry_cfg = RetryConfig(
        max_attempts=max_attempts,
        generator_timeout_seconds=config.get("retry", {}).get("generator_timeout_seconds", 30),
        verifier_timeout_seconds=config.get("retry", {}).get("verifier_timeout_seconds", 30),
    )

    problems = load_dataset(args.dataset)
    if args.family:
        problems = [p for p in problems if p.family == args.family]
    if not problems:
        print(f"No problems found in {args.dataset}", file=sys.stderr)
        sys.exit(1)

    generator = build_generator(args.generator)
    verifiers = build_verifiers(config)

    out_dir = os.path.join(args.trajectories, args.generator)
    os.makedirs(out_dir, exist_ok=True)

    trajectories = []
    for problem in problems:
        verifier = verifiers[problem.family]
        controller = RetryController(generator, verifier, retry_cfg)
        traj = controller.run(problem)
        save_trajectory(traj, out_dir)
        trajectories.append(traj)
        status_symbol = "PASS" if traj.verified else "FAIL"
        print(f"[{status_symbol}] {problem.problem_id} ({problem.family}/{problem.subtype}) "
              f"attempts={traj.attempts_used} status={traj.final_status}")

    n_verified = sum(1 for t in trajectories if t.verified)
    print(f"\n{args.generator}: {n_verified}/{len(trajectories)} problems verified. "
          f"Trajectories saved to {out_dir}")


def cmd_evaluate(args):
    trajectories = load_trajectories(args.trajectories)
    if not trajectories:
        print(f"No trajectories found in {args.trajectories}", file=sys.stderr)
        sys.exit(1)
    metrics = compute_metrics(trajectories)
    failures = analyze_failures(trajectories)
    report = {"metrics": metrics, "failure_analysis": failures}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nFull report (metrics + failure analysis) written to {args.out}")


def cmd_inspect(args):
    with open(args.trajectory) as f:
        traj = json.load(f)
    print(json.dumps(traj, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(prog="mathn", description="MATH-N: verifier-driven numerical mathematics")
    parser.add_argument("--config", default="configs/default.yaml", help="path to config YAML")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate-dataset", help="generate the synthetic V1 dataset")
    p_gen.add_argument("--dataset", default="datasets/v1")
    p_gen.add_argument("--problems-per-family", type=int, default=None)
    p_gen.add_argument("--base-seed", type=int, default=None)
    p_gen.set_defaults(func=cmd_generate_dataset)

    p_run = sub.add_parser("run", help="run a candidate generator over a dataset")
    p_run.add_argument("--dataset", default="datasets/v1")
    p_run.add_argument("--generator", required=True, choices=["scipy", "sympy"])
    p_run.add_argument("--family", default=None, help="restrict to one problem family")
    p_run.add_argument("--max-attempts", type=int, default=None)
    p_run.add_argument("--trajectories", default="trajectories")
    p_run.set_defaults(func=cmd_run)

    p_eval = sub.add_parser("evaluate", help="compute metrics + failure analysis over saved trajectories")
    p_eval.add_argument("--trajectories", required=True, help="directory of trajectory JSON files")
    p_eval.add_argument("--out", required=True, help="path to write the JSON report")
    p_eval.set_defaults(func=cmd_evaluate)

    p_inspect = sub.add_parser("inspect", help="pretty-print a single trajectory")
    p_inspect.add_argument("--trajectory", required=True)
    p_inspect.set_defaults(func=cmd_inspect)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
