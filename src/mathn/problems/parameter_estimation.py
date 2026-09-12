"""
Problem family C: parameter estimation.

    hidden true parameters
            |
    forward mathematical model
            |
    synthetic (noisy) observations  -> given to the candidate generator
            |
    candidate parameter estimate
            |
    independent forward simulation  (done by the verifier)
            |
    comparison against the SAME observations

The dataset retains the true parameters (under ``protected``) purely so
the evaluation harness can report parameter recovery error as a
diagnostic. They are never exposed to the candidate generator, and the
verifier's PASS/FAIL decision is based on simulation error against the
observations, not on directly comparing to the hidden true parameters.

V1 keeps both subtypes simple enough to be identifiable from the sampled
observation grid (see README limitations on identifiability).
"""

from __future__ import annotations

import numpy as np

from mathn.core.models import Problem
from mathn.problems.base import ProblemGenerator, register_problem_generator

SUBTYPES = ["exponential_decay_fit", "logistic_growth_fit"]


def forward_model(subtype: str, params: dict, t: np.ndarray, fixed: dict) -> np.ndarray:
    """The ground-truth forward model. Both the dataset generator (to
    create observations) and the verifier (to independently check a
    candidate's parameters) call this same function -- it IS the problem
    definition, analogous to f(x) in the root-finding family.
    """
    t = np.asarray(t, dtype=float)
    if subtype == "exponential_decay_fit":
        A, k = params["A"], params["k"]
        return A * np.exp(-k * t)
    if subtype == "logistic_growth_fit":
        r, K = params["r"], params["K"]
        y0 = fixed["y0"]
        return K / (1.0 + ((K - y0) / y0) * np.exp(-r * t))
    raise ValueError(f"Unknown parameter-estimation subtype {subtype!r}")


@register_problem_generator
class ParameterEstimationProblemGenerator(ProblemGenerator):
    family = "parameter_estimation"

    def __init__(self, subtypes=None, n_observations: int = 12, noise_std_frac: float = 0.02):
        self.subtypes = subtypes or SUBTYPES
        self.n_observations = n_observations
        self.noise_std_frac = noise_std_frac

    def generate(self, seed: int, index: int = 0) -> Problem:
        rng = np.random.default_rng(seed)
        subtype = self.subtypes[index % len(self.subtypes)]

        t = np.linspace(0.2, 6.0, self.n_observations)
        fixed = {}
        if subtype == "exponential_decay_fit":
            A = float(rng.uniform(2.0, 10.0))
            k = float(rng.uniform(0.2, 1.5))
            true_params = {"A": A, "k": k}
            query = "estimate A and k"
        elif subtype == "logistic_growth_fit":
            r = float(rng.uniform(0.3, 1.2))
            K = float(rng.uniform(10.0, 40.0))
            y0 = float(rng.uniform(0.5, K * 0.2))
            fixed = {"y0": y0}
            true_params = {"r": r, "K": K}
            query = "estimate r and K (y0 is given and fixed)"
        else:
            raise ValueError(subtype)

        clean = forward_model(subtype, true_params, t, fixed)
        noise_std = float(self.noise_std_frac * np.mean(np.abs(clean)))
        noise = rng.normal(0.0, noise_std, size=clean.shape)
        observed = clean + noise

        problem_id = f"mathn_param_{seed:06d}_{index:03d}"
        public = {
            "subtype": subtype,
            "t_observations": t.tolist(),
            "y_observations": observed.tolist(),
            "fixed": fixed,
            "query": query,
            "parameter_names": list(true_params.keys()),
        }
        protected = {
            "true_parameters": true_params,
            "noise_std": noise_std,
            "clean_observations": clean.tolist(),
        }
        return Problem(
            problem_id=problem_id,
            family="parameter_estimation",
            subtype=subtype,
            seed=seed,
            public=public,
            protected=protected,
            metadata={"tags": ["parameter_estimation", subtype], "identifiable": True},
        )
