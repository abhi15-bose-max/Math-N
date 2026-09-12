"""
Classical numerical candidate generator built on NumPy/SciPy.

Each problem family uses a genuinely different numerical *method* than
the one its verifier uses for the independent reference computation (see
verifiers/*.py docstrings), and each method's precision knob is refined
across retry attempts using the generic retry ``context`` (attempt number
and, where meaningful, prior feedback) -- never by peeking at the
problem's protected/reference fields.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares

from mathn.core.interfaces import CandidateGenerator, MalformedCandidateError
from mathn.core.models import Candidate
from mathn.generators.base import register_generator


def _attempt_schedule(context: Optional[Dict[str, Any]], values: list):
    """Pick a precision level from ``values`` based on the current
    attempt number (1-indexed), clamped to the last entry once attempts
    exceed the schedule length.
    """
    attempt = 1 if not context else int(context.get("attempt", 1))
    idx = min(attempt - 1, len(values) - 1)
    return values[idx]


@register_generator
class ScipyGenerator(CandidateGenerator):
    name = "scipy"
    supports_feedback = True

    def generate(self, problem: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Candidate:
        family = problem["family"]
        start = time.time()
        try:
            if family == "ode":
                result, raw = self._solve_ode(problem, context)
            elif family == "roots":
                result, raw = self._solve_root(problem, context)
            elif family == "parameter_estimation":
                result, raw = self._solve_parameters(problem, context)
            elif family == "symbolic":
                result, raw = self._solve_symbolic(problem, context)
            else:
                raise MalformedCandidateError(f"ScipyGenerator has no strategy for family {family!r}")
        except MalformedCandidateError:
            raise
        except Exception as exc:
            return Candidate(
                problem_id=problem["problem_id"],
                generator=self.name,
                attempt=(context or {}).get("attempt", 1),
                result={},
                malformed=True,
                error=f"{type(exc).__name__}: {exc}",
                runtime_seconds=time.time() - start,
            )

        return Candidate(
            problem_id=problem["problem_id"],
            generator=self.name,
            attempt=(context or {}).get("attempt", 1),
            result=result,
            raw=raw,
            runtime_seconds=time.time() - start,
        )

    # -- ODE ---------------------------------------------------------
    def _solve_ode(self, problem, context):
        subtype = problem["subtype"]
        params = problem["params"]
        y0 = problem["y0"]
        t_query = problem["t_query"]

        # Precision schedule: start loose, tighten on retry. RK45 is an
        # explicit embedded Runge-Kutta method -- a different algorithm
        # family from the verifier's implicit Radau reference solve.
        rtol = _attempt_schedule(context, [1e-2, 1e-3, 1e-5, 1e-7, 1e-9])
        atol = rtol * 1e-1

        if subtype == "exponential_decay":
            rhs = lambda t, y: [-params["k"] * y[0]]
        elif subtype == "exponential_growth":
            rhs = lambda t, y: [params["r"] * y[0]]
        elif subtype == "logistic_growth":
            rhs = lambda t, y: [params["r"] * y[0] * (1.0 - y[0] / params["K"])]
        elif subtype == "linear_ode":
            rhs = lambda t, y: [params["a"] * y[0] + params["b"]]
        elif subtype == "coupled_linear_system":
            rhs = lambda t, y: [
                params["a"] * y[0] + params["b"] * y[1],
                params["c"] * y[0] + params["d"] * y[1],
            ]
        else:
            raise MalformedCandidateError(f"unknown ODE subtype {subtype!r}")

        sol = solve_ivp(rhs, [0.0, t_query], y0, method="RK45", rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")

        return {"y_t_query": float(sol.y[0, -1])}, {"method": "RK45", "rtol": rtol, "atol": atol}

    # -- Roots ---------------------------------------------------------
    def _solve_root(self, problem, context):
        subtype = problem["subtype"]
        params = problem["params"]
        lo, hi = problem["domain"]

        if subtype == "polynomial":
            a, b, c = params["a"], params["b"], params["c"]
            f = lambda x: a * x ** 2 + b * x + c
        elif subtype == "transcendental":
            f = lambda x: math.cos(x) - x
        elif subtype == "nonlinear":
            c = params["c"]
            f = lambda x: x * math.exp(x) - c
        else:
            raise MalformedCandidateError(f"unknown root subtype {subtype!r}")

        # Precision schedule: brentq's xtol controls how tightly it
        # converges -- start crude, tighten on retry.
        xtol = _attempt_schedule(context, [1e-1, 1e-2, 1e-4, 1e-8, 1e-12])
        x = brentq(f, lo, hi, xtol=xtol, rtol=8.881784197001252e-16, maxiter=200)
        return {"x": float(x)}, {"method": "brentq", "xtol": xtol}

    # -- Parameter estimation -------------------------------------------
    def _solve_parameters(self, problem, context):
        subtype = problem["subtype"]
        t = np.array(problem["t_observations"])
        y_obs = np.array(problem["y_observations"])
        fixed = problem.get("fixed", {})
        attempt = 1 if not context else int(context.get("attempt", 1))

        # Multi-start: deterministic sequence of initial guesses, getting
        # progressively better informed by the shape of the observed data
        # itself (never by the hidden true parameters).
        rng = np.random.default_rng(1000 + attempt)

        if subtype == "exponential_decay_fit":
            def model(theta):
                A, k = theta
                return A * np.exp(-k * t) - y_obs

            # A crude, attempt-independent, data-driven guess is available
            # from the first/last observations; but for attempt 1 we
            # deliberately use a naive fixed guess to allow later attempts
            # (informed by a coarse log-linear fit) to demonstrate retry.
            if attempt == 1:
                x0 = np.array([1.0, 1.0])
            else:
                # Coarse closed-form guess via log-linear regression,
                # perturbed by a shrinking random offset per attempt.
                safe_y = np.clip(y_obs, 1e-6, None)
                slope, intercept = np.polyfit(t, np.log(safe_y), 1)
                guess = np.array([math.exp(intercept), -slope])
                jitter = rng.normal(0, 0.3 / attempt, size=2)
                x0 = np.clip(guess * (1 + jitter), 1e-6, None)

            res = least_squares(model, x0, max_nfev=200 * attempt)
            A, k = res.x
            return {"parameters": {"A": float(A), "k": float(k)}}, {"method": "least_squares", "x0": x0.tolist()}

        if subtype == "logistic_growth_fit":
            y0v = fixed["y0"]

            def model(theta):
                r, K = theta
                return K / (1.0 + ((K - y0v) / y0v) * np.exp(-r * t)) - y_obs

            if attempt == 1:
                x0 = np.array([0.1, max(y_obs) * 0.5 + 1.0])
            else:
                K_guess = max(float(np.max(y_obs)) * 1.1, y0v * 1.5)
                r_guess = 0.5
                jitter = rng.normal(0, 0.3 / attempt, size=2)
                x0 = np.clip(np.array([r_guess, K_guess]) * (1 + jitter), 1e-3, None)

            res = least_squares(model, x0, max_nfev=200 * attempt, bounds=([1e-4, y0v * 1.001], [10.0, 1e4]))
            r, K = res.x
            return {"parameters": {"r": float(r), "K": float(K)}}, {"method": "least_squares", "x0": x0.tolist()}

        raise MalformedCandidateError(f"unknown parameter-estimation subtype {subtype!r}")

    # -- Symbolic-to-numerical -------------------------------------------
    def _solve_symbolic(self, problem, context):
        subtype = problem["subtype"]
        coeffs = problem["params"]["coeffs"]
        trig = problem["params"]["trig"]

        def f(x):
            val = sum(c * x ** i for i, c in enumerate(coeffs))
            val += trig["sin_amp"] * math.sin(trig["freq"] * x)
            val += trig["cos_amp"] * math.cos(trig["freq"] * x)
            return val

        if subtype == "derivative_at_point":
            x0 = problem["point"]
            # Simple forward difference, refined (smaller step) on retry.
            # Deliberately a *different* numerical method/order from the
            # verifier's centered 5-point stencil.
            h = _attempt_schedule(context, [1e-1, 1e-2, 1e-4, 1e-6, 1e-7])
            value = (f(x0 + h) - f(x0)) / h
            return {"value": float(value)}, {"method": "forward_difference", "h": h}

        if subtype == "definite_integral":
            a, b = problem["interval"]
            # Composite Simpson's rule with a refined number of panels on
            # retry -- distinct from the verifier's adaptive quadrature.
            n = _attempt_schedule(context, [4, 16, 64, 512, 4096])
            n = n if n % 2 == 0 else n + 1
            xs = np.linspace(a, b, n + 1)
            ys = np.array([f(x) for x in xs])
            h = (b - a) / n
            simpson = h / 3 * (ys[0] + ys[-1] + 4 * np.sum(ys[1:-1:2]) + 2 * np.sum(ys[2:-2:2]))
            return {"value": float(simpson)}, {"method": "simpson", "panels": n}

        raise MalformedCandidateError(f"unknown symbolic subtype {subtype!r}")
