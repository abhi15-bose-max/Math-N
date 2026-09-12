"""
Problem family B: numerical equation solving, f(x) = 0.

Each problem stores enough structured information (coefficients / a
symbolic expression string) that BOTH the candidate generator and the
verifier can independently build a callable f(x). The thing that must be
independent is the *method used to find the root*, not the definition of
f itself (f is the problem statement, agreed upon by construction).
"""

from __future__ import annotations

import math

import numpy as np

from mathn.core.models import Problem
from mathn.problems.base import ProblemGenerator, register_problem_generator

SUBTYPES = ["polynomial", "transcendental", "nonlinear"]


def build_callable(problem: Problem):
    """Independently reconstruct f(x) from the problem's public
    parameters. Used by both generators and the verifier."""
    subtype = problem.subtype
    p = problem.public["params"]
    if subtype == "polynomial":
        a, b, c = p["a"], p["b"], p["c"]
        return lambda x: a * x ** 2 + b * x + c
    if subtype == "transcendental":
        return lambda x: math.cos(x) - x
    if subtype == "nonlinear":
        c = p["c"]
        return lambda x: x * math.exp(x) - c
    raise ValueError(f"Unknown root subtype {subtype!r}")


@register_problem_generator
class RootProblemGenerator(ProblemGenerator):
    family = "roots"

    def __init__(self, subtypes=None):
        self.subtypes = subtypes or SUBTYPES

    def generate(self, seed: int, index: int = 0) -> Problem:
        rng = np.random.default_rng(seed)
        subtype = self.subtypes[index % len(self.subtypes)]

        if subtype == "polynomial":
            r1 = float(rng.uniform(-5.0, 5.0))
            gap = float(rng.uniform(1.5, 4.0))
            r2 = r1 + gap * (1 if rng.uniform() > 0.5 else -1)
            a = 1.0
            b = -(r1 + r2)
            c = r1 * r2
            delta = min(0.45, abs(r1 - r2) / 2.0 - 0.05)
            lo, hi = r1 - delta, r1 + delta
            params = {"a": a, "b": b, "c": c}
            expression = f"{a:.6g}*x^2 + ({b:.6g})*x + ({c:.6g})"
            constructed_root = r1
        elif subtype == "transcendental":
            # cos(x) - x = 0 has a unique root (the Dottie number) on [0, 1]
            lo, hi = 0.0, 1.0
            params = {}
            expression = "cos(x) - x"
            constructed_root = 0.7390851332151607
        elif subtype == "nonlinear":
            # x*exp(x) = c is strictly increasing for x > 0 => unique positive root
            c = float(rng.uniform(0.5, 12.0))
            lo, hi = 0.0, 6.0
            f = lambda x: x * math.exp(x) - c
            while f(hi) < 0:
                hi *= 1.5
            params = {"c": c}
            expression = f"x*exp(x) - ({c:.6g})"
            from scipy.optimize import brentq

            constructed_root = float(brentq(f, lo, hi, xtol=1e-13, rtol=1e-13))
        else:
            raise ValueError(subtype)

        problem_id = f"mathn_root_{seed:06d}_{index:03d}"
        public = {
            "subtype": subtype,
            "expression": expression,
            "params": params,
            "domain": [lo, hi],
            "query": "find x in domain such that f(x) = 0",
        }
        protected = {"constructed_root": float(constructed_root)}
        return Problem(
            problem_id=problem_id,
            family="roots",
            subtype=subtype,
            seed=seed,
            public=public,
            protected=protected,
            metadata={"tags": ["roots", subtype]},
        )
