"""
Problem family D: symbolic-to-numerical problems.

A symbolic expression is generated, and the query asks for a numerical
quantity derived from it (a derivative value at a point, or a definite
integral). SymPy is a natural *candidate generator* here (symbolic
differentiation / integration, then numeric evaluation) but the verifier
uses an independent, purely numerical method (finite differences /
quadrature) so SymPy is never asked to confirm its own symbolic result.
"""

from __future__ import annotations

import math

import numpy as np

from mathn.core.models import Problem
from mathn.problems.base import ProblemGenerator, register_problem_generator

SUBTYPES = ["derivative_at_point", "definite_integral"]


def build_callable(problem: Problem):
    """Independently reconstruct f(x) as a plain Python callable (no
    SymPy involved) from the problem's public parameters."""
    coeffs = problem.public["params"]["coeffs"]  # polynomial coefficients, low->high degree
    trig = problem.public["params"]["trig"]  # {"sin": amp, "cos": amp} multiplied by freq

    def f(x):
        val = sum(c * x ** i for i, c in enumerate(coeffs))
        val += trig["sin_amp"] * math.sin(trig["freq"] * x)
        val += trig["cos_amp"] * math.cos(trig["freq"] * x)
        return val

    return f


def sympy_expression(problem: Problem):
    """Build the equivalent SymPy expression -- this is what a SymPy-based
    candidate generator is expected to use."""
    import sympy as sp

    x = sp.symbols("x")
    coeffs = problem.public["params"]["coeffs"]
    trig = problem.public["params"]["trig"]
    expr = sum(sp.Float(c) * x ** i for i, c in enumerate(coeffs))
    expr += sp.Float(trig["sin_amp"]) * sp.sin(sp.Float(trig["freq"]) * x)
    expr += sp.Float(trig["cos_amp"]) * sp.cos(sp.Float(trig["freq"]) * x)
    return x, expr


@register_problem_generator
class SymbolicProblemGenerator(ProblemGenerator):
    family = "symbolic"

    def __init__(self, subtypes=None):
        self.subtypes = subtypes or SUBTYPES

    def generate(self, seed: int, index: int = 0) -> Problem:
        rng = np.random.default_rng(seed)
        subtype = self.subtypes[index % len(self.subtypes)]

        coeffs = [float(rng.uniform(-2.0, 2.0)) for _ in range(3)]  # up to quadratic
        trig = {
            "sin_amp": float(rng.uniform(-1.5, 1.5)),
            "cos_amp": float(rng.uniform(-1.5, 1.5)),
            "freq": float(rng.uniform(0.5, 2.0)),
        }
        params = {"coeffs": coeffs, "trig": trig}

        if subtype == "derivative_at_point":
            x0 = float(rng.uniform(-2.0, 2.0))
            public = {
                "subtype": subtype,
                "params": params,
                "point": x0,
                "query": "compute f'(point)",
            }
        elif subtype == "definite_integral":
            a = float(rng.uniform(-2.0, 0.0))
            b = a + float(rng.uniform(0.5, 3.0))
            public = {
                "subtype": subtype,
                "params": params,
                "interval": [a, b],
                "query": "compute the definite integral of f over interval",
            }
        else:
            raise ValueError(subtype)

        problem_id = f"mathn_sym_{seed:06d}_{index:03d}"
        return Problem(
            problem_id=problem_id,
            family="symbolic",
            subtype=subtype,
            seed=seed,
            public=public,
            protected={},
            metadata={"tags": ["symbolic", subtype]},
        )
