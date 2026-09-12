"""
Dataset generation and persistence.

A "dataset" here is just a list of Problem instances spanning one or more
problem families, saved as one JSON file per problem (each with a
reproducible seed baked into its problem_id).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

from mathn.core.models import Problem
from mathn.problems.ode import ODEProblemGenerator
from mathn.problems.parameter_estimation import ParameterEstimationProblemGenerator
from mathn.problems.roots import RootProblemGenerator
from mathn.problems.symbolic import SymbolicProblemGenerator

FAMILY_GENERATORS = {
    "ode": ODEProblemGenerator,
    "roots": RootProblemGenerator,
    "parameter_estimation": ParameterEstimationProblemGenerator,
    "symbolic": SymbolicProblemGenerator,
}

# Fixed, deterministic per-family seed offsets. NOTE: do not use Python's
# builtin hash() on strings here -- it is randomized per-process (PEP 456
# hash randomization) unless PYTHONHASHSEED is pinned, which would make
# "the same seed produces the same dataset" false across process runs.
_FAMILY_SEED_OFFSET = {
    "ode": 0,
    "roots": 1_000,
    "parameter_estimation": 2_000,
    "symbolic": 3_000,
}


def build_v1_dataset(problems_per_family: int = 10, base_seed: int = 0) -> List[Problem]:
    """Build the default V1 development dataset: ``problems_per_family``
    problems from each of the four built-in families, cycling through
    each family's subtypes.
    """
    problems: List[Problem] = []
    for family, gen_cls in FAMILY_GENERATORS.items():
        generator = gen_cls()
        family_seed_base = base_seed + _FAMILY_SEED_OFFSET[family]
        problems.extend(generator.generate_many(problems_per_family, base_seed=family_seed_base))
    return problems


def save_dataset(problems: List[Problem], directory: str) -> List[str]:
    os.makedirs(directory, exist_ok=True)
    paths = []
    for p in problems:
        path = os.path.join(directory, f"{p.problem_id}.json")
        with open(path, "w") as f:
            json.dump(p.to_dict(include_protected=True), f, indent=2)
        paths.append(path)
    return paths


def load_dataset(directory: str) -> List[Problem]:
    paths = sorted(Path(directory).glob("*.json"))
    problems = []
    for path in paths:
        with open(path) as f:
            d = json.load(f)
        problems.append(Problem.from_dict(d))
    return problems
