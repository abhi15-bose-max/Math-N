"""
Problem family A: ODE / initial-value problems.

Each subtype is a small, well-understood ODE with known analytic
structure. The generator samples parameters with a seeded RNG, computes
an independent reference value for y(t_query) at generation time (stored
under ``protected`` purely for dataset auditing), and exposes to the
candidate generator only the equation description, parameters, initial
condition, and query time.

Independence note: the *verifier* (see verifiers/ode_verifier.py) never
reads ``protected["reference_y"]`` to make its PASS/FAIL decision -- it
always independently re-integrates the ODE with its own solver
configuration. The stored reference is used only for dataset-level
sanity checks and evaluation diagnostics.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from mathn.core.models import Problem
from mathn.problems.base import ProblemGenerator, register_problem_generator

SUBTYPES = [
    "exponential_decay",
    "exponential_growth",
    "logistic_growth",
    "linear_ode",
    "coupled_linear_system",
]


def _rhs_for(subtype: str, params: dict):
    """Build the right-hand-side callable f(t, y) -> dy/dt for a subtype.
    Used both by the reference solver at generation time and (rebuilt
    independently) by the verifier.
    """
    if subtype == "exponential_decay":
        k = params["k"]
        return lambda t, y: [-k * y[0]]
    if subtype == "exponential_growth":
        r = params["r"]
        return lambda t, y: [r * y[0]]
    if subtype == "logistic_growth":
        r, K = params["r"], params["K"]
        return lambda t, y: [r * y[0] * (1.0 - y[0] / K)]
    if subtype == "linear_ode":
        a, b = params["a"], params["b"]
        return lambda t, y: [a * y[0] + b]
    if subtype == "coupled_linear_system":
        a, b, c, d = params["a"], params["b"], params["c"], params["d"]
        return lambda t, y: [a * y[0] + b * y[1], c * y[0] + d * y[1]]
    raise ValueError(f"Unknown ODE subtype {subtype!r}")


def _analytic_reference(subtype: str, params: dict, y0, t_query: float):
    """Closed-form reference where one exists; falls back to a very
    tight numerical integration otherwise (coupled system).
    """
    if subtype == "exponential_decay":
        return y0[0] * np.exp(-params["k"] * t_query)
    if subtype == "exponential_growth":
        return y0[0] * np.exp(params["r"] * t_query)
    if subtype == "logistic_growth":
        r, K = params["r"], params["K"]
        y0v = y0[0]
        return K / (1.0 + ((K - y0v) / y0v) * np.exp(-r * t_query))
    if subtype == "linear_ode":
        a, b = params["a"], params["b"]
        y0v = y0[0]
        if abs(a) < 1e-12:
            return y0v + b * t_query
        return (y0v + b / a) * np.exp(a * t_query) - b / a
    if subtype == "coupled_linear_system":
        rhs = _rhs_for(subtype, params)
        sol = solve_ivp(rhs, [0.0, t_query], y0, method="Radau", rtol=1e-12, atol=1e-14, dense_output=True)
        return float(sol.y[0, -1])
    raise ValueError(f"Unknown ODE subtype {subtype!r}")


@register_problem_generator
class ODEProblemGenerator(ProblemGenerator):
    family = "ode"

    def __init__(self, subtypes=None):
        self.subtypes = subtypes or SUBTYPES

    def generate(self, seed: int, index: int = 0) -> Problem:
        rng = np.random.default_rng(seed)
        subtype = self.subtypes[index % len(self.subtypes)]

        if subtype == "exponential_decay":
            k = float(rng.uniform(0.1, 2.0))
            y0 = [float(rng.uniform(1.0, 10.0))]
            t_query = float(rng.uniform(0.5, 4.0))
            params = {"k": k}
        elif subtype == "exponential_growth":
            r = float(rng.uniform(0.05, 0.8))
            y0 = [float(rng.uniform(0.5, 5.0))]
            t_query = float(rng.uniform(0.5, 3.0))
            params = {"r": r}
        elif subtype == "logistic_growth":
            r = float(rng.uniform(0.2, 1.5))
            K = float(rng.uniform(10.0, 50.0))
            y0 = [float(rng.uniform(0.5, K * 0.3))]
            t_query = float(rng.uniform(0.5, 6.0))
            params = {"r": r, "K": K}
        elif subtype == "linear_ode":
            a = float(rng.uniform(-1.5, -0.1))
            b = float(rng.uniform(-2.0, 2.0))
            y0 = [float(rng.uniform(-5.0, 5.0))]
            t_query = float(rng.uniform(0.5, 4.0))
            params = {"a": a, "b": b}
        elif subtype == "coupled_linear_system":
            # Sample a stable-ish 2x2 linear system (negative trace bias)
            a = float(rng.uniform(-1.0, -0.1))
            d = float(rng.uniform(-1.0, -0.1))
            b = float(rng.uniform(-0.5, 0.5))
            c = float(rng.uniform(-0.5, 0.5))
            y0 = [float(rng.uniform(-2.0, 2.0)), float(rng.uniform(-2.0, 2.0))]
            t_query = float(rng.uniform(0.5, 3.0))
            params = {"a": a, "b": b, "c": c, "d": d}
        else:
            raise ValueError(subtype)

        reference_y = _analytic_reference(subtype, params, y0, t_query)

        problem_id = f"mathn_ode_{seed:06d}_{index:03d}"
        public = {
            "equation": _describe(subtype, params),
            "subtype": subtype,
            "params": params,
            "y0": y0,
            "t0": 0.0,
            "t_query": t_query,
            "query": "compute y(t_query) for the component of interest (index 0)"
            if subtype != "coupled_linear_system"
            else "compute x(t_query), i.e. the first component of the system",
        }
        protected = {"reference_y": float(reference_y)}
        return Problem(
            problem_id=problem_id,
            family="ode",
            subtype=subtype,
            seed=seed,
            public=public,
            protected=protected,
            metadata={"tags": ["ode", subtype]},
        )


def _describe(subtype: str, params: dict) -> str:
    if subtype == "exponential_decay":
        return f"dy/dt = -{params['k']:.6g} * y"
    if subtype == "exponential_growth":
        return f"dy/dt = {params['r']:.6g} * y"
    if subtype == "logistic_growth":
        return f"dy/dt = {params['r']:.6g} * y * (1 - y/{params['K']:.6g})"
    if subtype == "linear_ode":
        return f"dy/dt = {params['a']:.6g} * y + {params['b']:.6g}"
    if subtype == "coupled_linear_system":
        return (
            f"dx/dt = {params['a']:.6g} x + {params['b']:.6g} y; "
            f"dy/dt = {params['c']:.6g} x + {params['d']:.6g} y"
        )
    raise ValueError(subtype)


def build_rhs(problem: Problem):
    """Public helper so verifiers/generators can independently reconstruct
    the right-hand side from the problem's public parameters."""
    return _rhs_for(problem.subtype, problem.public["params"])
