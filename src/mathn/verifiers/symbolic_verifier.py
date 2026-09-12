"""
Verifier for symbolic-to-numerical problems (derivative / integral value).

Independence strategy: the candidate is typically produced symbolically
(SymPy diff/integrate then substitute). The verifier instead uses purely
numerical methods that never touch SymPy:

    derivative_at_point -> 5-point central finite difference
    definite_integral   -> scipy.integrate.quad (adaptive quadrature)

f itself is rebuilt as a plain Python callable independent of the SymPy
expression tree (mathn.problems.symbolic.build_callable).
"""

from __future__ import annotations

import math
import time

from scipy.integrate import quad

from mathn.core.models import Candidate, Problem, VerificationResult
from mathn.problems.symbolic import build_callable
from mathn.verifiers.base import NumericalVerifier, STATUS


def _central_difference_5pt(f, x0: float, h: float = 1e-4) -> float:
    return (
        -f(x0 + 2 * h) + 8 * f(x0 + h) - 8 * f(x0 - h) + f(x0 - 2 * h)
    ) / (12 * h)


class SymbolicVerifier(NumericalVerifier):
    def __init__(self, atol: float = 1e-4, rtol: float = 1e-4):
        self.atol = atol
        self.rtol = rtol

    def verify(self, problem: Problem, candidate: Candidate) -> VerificationResult:
        start = time.time()

        if candidate.malformed or "value" not in candidate.result:
            return VerificationResult(
                verified=False,
                status=STATUS.MALFORMED_CANDIDATE,
                details={"reason": "candidate missing required field 'value'"},
                runtime_seconds=time.time() - start,
            )

        try:
            value = float(candidate.result["value"])
        except (TypeError, ValueError):
            return VerificationResult(
                verified=False,
                status=STATUS.MALFORMED_CANDIDATE,
                details={"reason": "value is not a real number"},
                runtime_seconds=time.time() - start,
            )

        if not math.isfinite(value):
            return VerificationResult(
                verified=False,
                status=STATUS.NUMERICAL_MISMATCH,
                details={"reason": "candidate value is not finite"},
                runtime_seconds=time.time() - start,
            )

        f = build_callable(problem)

        try:
            if problem.subtype == "derivative_at_point":
                reference = _central_difference_5pt(f, problem.public["point"])
            elif problem.subtype == "definite_integral":
                a, b = problem.public["interval"]
                reference, _quad_err = quad(f, a, b, epsabs=1e-10, epsrel=1e-10)
            else:
                return VerificationResult(
                    verified=False,
                    status=STATUS.OTHER,
                    details={"reason": f"unknown symbolic subtype {problem.subtype!r}"},
                    runtime_seconds=time.time() - start,
                )
        except Exception as exc:  # pragma: no cover - defensive
            return VerificationResult(
                verified=False,
                status=STATUS.SOLVER_FAILURE,
                details={"reason": f"independent numerical evaluation raised: {exc}"},
                runtime_seconds=time.time() - start,
            )

        abs_err = abs(value - reference)
        rel_err = abs_err / (abs(reference) + 1e-300)
        threshold = self.atol + self.rtol * abs(reference)
        verified = abs_err <= threshold

        return VerificationResult(
            verified=verified,
            status=STATUS.VERIFIED if verified else STATUS.NUMERICAL_MISMATCH,
            details={
                "candidate": value,
                "reference": reference,
                "absolute_error": abs_err,
                "relative_error": rel_err,
                "atol": self.atol,
                "rtol": self.rtol,
                "threshold": threshold,
                "reference_method": "finite_difference" if problem.subtype == "derivative_at_point" else "quad",
            },
            runtime_seconds=time.time() - start,
            message=None if verified else "Candidate exceeds tolerance against independent numerical reference.",
        )
