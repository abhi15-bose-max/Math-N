"""
Verifier for ODE / initial-value problems.

Independence strategy: regardless of how the candidate was produced
(closed-form analytic substitution, SciPy's solve_ivp with loose
tolerances, a symbolic dsolve, etc.), the verifier ALWAYS performs its
own high-accuracy numerical integration (Radau, very tight rtol/atol) to
obtain a reference value, then compares.

    absolute_error <= atol + rtol * |reference|
"""

from __future__ import annotations

import math
import time

from scipy.integrate import solve_ivp

from mathn.core.models import Candidate, Problem, VerificationResult
from mathn.problems.ode import build_rhs
from mathn.verifiers.base import NumericalVerifier, STATUS


class ODEVerifier(NumericalVerifier):
    def __init__(self, atol: float = 1e-6, rtol: float = 1e-5):
        self.atol = atol
        self.rtol = rtol

    def verify(self, problem: Problem, candidate: Candidate) -> VerificationResult:
        start = time.time()

        if candidate.malformed or "y_t_query" not in candidate.result:
            return VerificationResult(
                verified=False,
                status=STATUS.MALFORMED_CANDIDATE,
                details={"reason": "candidate missing required field 'y_t_query'"},
                runtime_seconds=time.time() - start,
                message="Malformed candidate.",
            )

        try:
            y_candidate = float(candidate.result["y_t_query"])
        except (TypeError, ValueError):
            return VerificationResult(
                verified=False,
                status=STATUS.MALFORMED_CANDIDATE,
                details={"reason": "y_t_query is not a real number"},
                runtime_seconds=time.time() - start,
            )

        if not math.isfinite(y_candidate):
            return VerificationResult(
                verified=False,
                status=STATUS.NUMERICAL_MISMATCH,
                details={"reason": "candidate is not finite", "candidate": y_candidate},
                runtime_seconds=time.time() - start,
            )

        try:
            rhs = build_rhs(problem)
            y0 = problem.public["y0"]
            t_query = problem.public["t_query"]
            sol = solve_ivp(
                rhs, [0.0, t_query], y0, method="Radau",
                rtol=1e-12, atol=1e-14, dense_output=False,
            )
            if not sol.success:
                return VerificationResult(
                    verified=False,
                    status=STATUS.SOLVER_FAILURE,
                    details={"reason": "independent reference integration failed", "solver_message": sol.message},
                    runtime_seconds=time.time() - start,
                )
            reference = float(sol.y[0, -1])
        except Exception as exc:  # pragma: no cover - defensive
            return VerificationResult(
                verified=False,
                status=STATUS.SOLVER_FAILURE,
                details={"reason": f"independent integration raised: {exc}"},
                runtime_seconds=time.time() - start,
            )

        abs_err = abs(y_candidate - reference)
        rel_err = abs_err / (abs(reference) + 1e-300)
        threshold = self.atol + self.rtol * abs(reference)
        verified = abs_err <= threshold

        return VerificationResult(
            verified=verified,
            status=STATUS.VERIFIED if verified else STATUS.NUMERICAL_MISMATCH,
            details={
                "candidate": y_candidate,
                "reference": reference,
                "absolute_error": abs_err,
                "relative_error": rel_err,
                "atol": self.atol,
                "rtol": self.rtol,
                "threshold": threshold,
            },
            runtime_seconds=time.time() - start,
            message=None if verified else "Candidate exceeds tolerance against independently integrated reference.",
        )
