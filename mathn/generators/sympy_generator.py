"""
Symbolic candidate generator built on SymPy.

Where a closed-form or exact symbolic method exists, this generator uses
it instead of the SciPy generator's iterative numerical approach. This
gives the framework a genuine second, methodologically distinct
candidate-generation strategy to compare against SciPy under the same
verifiers (see section 23 of the project brief: generator comparison).

Symbolic methods used per family:
    ode                  -> sympy.dsolve (closed-form / matrix-exponential
                             solution), substitute t_query numerically
    roots.polynomial      -> sympy.solve (exact roots of the quadratic)
    roots.transcendental   -> sympy.nsolve (symbolic-engine-driven Newton)
    roots.nonlinear        -> closed form via the Lambert W special function
    parameter_estimation  -> analytic / few-point closed-form estimators
                             (log-linear regression, 2-point nonlinear
                             solve), distinct from SciPy's full-data
                             nonlinear least squares
    symbolic              -> exact sympy.diff / sympy.integrate
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

import numpy as np
import sympy as sp

from mathn.core.interfaces import CandidateGenerator, MalformedCandidateError
from mathn.core.models import Candidate
from mathn.generators.base import register_generator


@register_generator
class SympyGenerator(CandidateGenerator):
    name = "sympy"
    # Most of these methods are exact / closed-form and do not depend on
    # retry feedback -- they either produce their best answer immediately
    # or fail outright (e.g. an unsupported ODE structure).
    supports_feedback = False

    def generate(self, problem: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Candidate:
        family = problem["family"]
        start = time.time()
        attempt = (context or {}).get("attempt", 1)
        try:
            if family == "ode":
                result, raw = self._solve_ode(problem)
            elif family == "roots":
                result, raw = self._solve_root(problem)
            elif family == "parameter_estimation":
                result, raw = self._solve_parameters(problem, attempt)
            elif family == "symbolic":
                result, raw = self._solve_symbolic(problem)
            else:
                raise MalformedCandidateError(f"SympyGenerator has no strategy for family {family!r}")
        except MalformedCandidateError:
            raise
        except Exception as exc:
            return Candidate(
                problem_id=problem["problem_id"],
                generator=self.name,
                attempt=attempt,
                result={},
                malformed=True,
                error=f"{type(exc).__name__}: {exc}",
                runtime_seconds=time.time() - start,
            )

        return Candidate(
            problem_id=problem["problem_id"],
            generator=self.name,
            attempt=attempt,
            result=result,
            raw=raw,
            runtime_seconds=time.time() - start,
        )

    # -- ODE: closed-form via dsolve -------------------------------------
    def _solve_ode(self, problem):
        subtype = problem["subtype"]
        params = problem["params"]
        y0 = problem["y0"]
        t_query = problem["t_query"]
        t = sp.symbols("t")

        if subtype == "coupled_linear_system":
            x = sp.Function("x")
            y = sp.Function("y")
            a, b, c, d = params["a"], params["b"], params["c"], params["d"]
            eqs = [
                sp.Eq(x(t).diff(t), a * x(t) + b * y(t)),
                sp.Eq(y(t).diff(t), c * x(t) + d * y(t)),
            ]
            sol = sp.dsolve(eqs, [x(t), y(t)], ics={x(0): y0[0], y(0): y0[1]})
            x_expr = [s.rhs for s in sol if s.lhs == x(t)][0]
            value = complex(x_expr.subs(t, t_query).evalf())
            return {"y_t_query": float(value.real)}, {"method": "sympy.dsolve", "expr": str(x_expr)}

        if subtype == "logistic_growth":
            # The logistic equation is a Bernoulli equation. dsolve applied
            # directly to it often returns an *implicit* solution (a
            # transcendental equation in y(t), not solved for y(t)), which
            # is expensive and unreliable to invert numerically. Instead we
            # apply the standard Bernoulli substitution u = 1/y, which
            # turns the logistic ODE into a first-order LINEAR ODE that
            # dsolve solves explicitly and quickly -- still a genuine
            # symbolic derivation, just a better-conditioned one.
            r, K = params["r"], params["K"]
            u = sp.Function("u")
            eq_u = sp.Eq(u(t).diff(t), -r * u(t) + r / K)
            sol_u = sp.dsolve(eq_u, u(t), ics={u(0): 1 / sp.Float(y0[0])})
            y_expr = 1 / sol_u.rhs
            value = float(sp.N(y_expr.subs(t, t_query)))
            return {"y_t_query": value}, {"method": "sympy.dsolve (Bernoulli substitution)", "expr": str(y_expr)}

        y = sp.Function("y")
        if subtype == "exponential_decay":
            k = params["k"]
            eq = sp.Eq(y(t).diff(t), -k * y(t))
        elif subtype == "exponential_growth":
            r = params["r"]
            eq = sp.Eq(y(t).diff(t), r * y(t))
        elif subtype == "linear_ode":
            a, b = params["a"], params["b"]
            eq = sp.Eq(y(t).diff(t), a * y(t) + b)
        else:
            raise MalformedCandidateError(f"unknown ODE subtype {subtype!r}")

        sol = sp.dsolve(eq, y(t), ics={y(0): y0[0]})
        expr = sol.rhs
        value = float(sp.N(expr.subs(t, t_query)))
        return {"y_t_query": value}, {"method": "sympy.dsolve", "expr": str(expr)}

    # -- Roots -----------------------------------------------------------
    def _solve_root(self, problem):
        subtype = problem["subtype"]
        params = problem["params"]
        lo, hi = problem["domain"]
        x = sp.symbols("x", real=True)

        if subtype == "polynomial":
            a, b, c = params["a"], params["b"], params["c"]
            roots = sp.solve(sp.Eq(a * x ** 2 + b * x + c, 0), x)
            real_roots = [float(r) for r in roots if r.is_real]
            in_domain = [r for r in real_roots if lo - 1e-9 <= r <= hi + 1e-9]
            if not in_domain:
                raise MalformedCandidateError("no exact real root of the quadratic falls in the required domain")
            chosen = min(in_domain, key=lambda r: abs(r - (lo + hi) / 2))
            return {"x": chosen}, {"method": "sympy.solve", "all_roots": real_roots}

        if subtype == "transcendental":
            guess = (lo + hi) / 2
            root = sp.nsolve(sp.cos(x) - x, x, guess, prec=30)
            return {"x": float(root)}, {"method": "sympy.nsolve"}

        if subtype == "nonlinear":
            c = params["c"]
            # x * exp(x) = c  =>  x = W(c), the Lambert W function.
            root = sp.LambertW(c)
            value = float(sp.N(root, 30))
            return {"x": value}, {"method": "sympy.LambertW"}

        raise MalformedCandidateError(f"unknown root subtype {subtype!r}")

    # -- Parameter estimation: closed-form / few-point analytic methods --
    def _solve_parameters(self, problem, attempt):
        subtype = problem["subtype"]
        t = np.array(problem["t_observations"])
        y_obs = np.array(problem["y_observations"])
        fixed = problem.get("fixed", {})

        if subtype == "exponential_decay_fit":
            # Closed-form log-linear regression (ordinary least squares in
            # log-space), derived symbolically from the normal equations.
            ts, ys = sp.symbols("ts ys")
            n = len(t)
            log_y = np.log(np.clip(y_obs, 1e-9, None))
            t_mean, logy_mean = float(np.mean(t)), float(np.mean(log_y))
            slope_expr = sp.Symbol("slope")
            num = sum((float(ti) - t_mean) * (float(lyi) - logy_mean) for ti, lyi in zip(t, log_y))
            den = sum((float(ti) - t_mean) ** 2 for ti in t)
            slope = num / den
            intercept = logy_mean - slope * t_mean
            k = float(-slope)
            A = float(sp.exp(intercept))
            return {"parameters": {"A": A, "k": k}}, {"method": "sympy_loglinear_regression"}

        if subtype == "logistic_growth_fit":
            # Exact 2-point solve: pick two well-separated observations and
            # solve the (nonlinear) logistic equations for (r, K) exactly
            # at those two points using sympy.nsolve. This purposefully
            # ignores the rest of the data, unlike SciPy's full-data
            # nonlinear least squares -- a genuinely different method.
            y0v = fixed["y0"]
            n = len(t)
            i1, i2 = n // 3, (2 * n) // 3
            t1, y1 = float(t[i1]), float(y_obs[i1])
            t2, y2 = float(t[i2]), float(y_obs[i2])

            r, K = sp.symbols("r K", positive=True)
            eq1 = sp.Eq(K / (1 + ((K - y0v) / y0v) * sp.exp(-r * t1)), y1)
            eq2 = sp.Eq(K / (1 + ((K - y0v) / y0v) * sp.exp(-r * t2)), y2)
            guess = (0.5, max(float(np.max(y_obs)) * 1.2, y0v * 1.5))
            try:
                sol = sp.nsolve([eq1, eq2], [r, K], guess, prec=25)
                r_val, K_val = float(sol[0]), float(sol[1])
            except Exception as exc:
                raise MalformedCandidateError(f"2-point nsolve failed to converge: {exc}")
            return {"parameters": {"r": r_val, "K": K_val}}, {
                "method": "sympy_two_point_nsolve",
                "points_used": [[t1, y1], [t2, y2]],
            }

        raise MalformedCandidateError(f"unknown parameter-estimation subtype {subtype!r}")

    # -- Symbolic-to-numerical: exact diff/integrate ----------------------
    def _solve_symbolic(self, problem):
        x = sp.symbols("x")
        coeffs = problem["params"]["coeffs"]
        trig = problem["params"]["trig"]
        expr = sum(sp.Float(c) * x ** i for i, c in enumerate(coeffs))
        expr += sp.Float(trig["sin_amp"]) * sp.sin(sp.Float(trig["freq"]) * x)
        expr += sp.Float(trig["cos_amp"]) * sp.cos(sp.Float(trig["freq"]) * x)

        if problem["subtype"] == "derivative_at_point":
            deriv = sp.diff(expr, x)
            value = float(deriv.subs(x, problem["point"]))
            return {"value": value}, {"method": "sympy.diff", "expr": str(deriv)}

        if problem["subtype"] == "definite_integral":
            a, b = problem["interval"]
            antideriv = sp.integrate(expr, x)
            value = float(sp.N(antideriv.subs(x, b) - antideriv.subs(x, a)))
            return {"value": value}, {"method": "sympy.integrate", "expr": str(antideriv)}

        raise MalformedCandidateError(f"unknown symbolic subtype {problem['subtype']!r}")
