"""
Verifier for root-finding problems: f(x) = 0.

Independence strategy: the verifier evaluates f(x_candidate) via a plain
Python callable rebuilt directly from the problem's stored parameters
(mathn.problems.roots.build_callable) -- it does not re-run whatever
search procedure the candidate generator used (brentq, Newton, symbolic
solve, ...), it simply checks the residual and domain membership.
"""

from __future__ import annotations

import math
import time

from mathn.core.models import Candidate, Problem, VerificationResult
from mathn.problems.roots import build_callable
from mathn.verifiers.base import NumericalVerifier, STATUS


class RootVerifier(NumericalVerifier):
    def __init__(self, residual_tolerance: float = 1e-6):
        self.residual_tolerance = residual_tolerance

    def verify(self, problem: Problem, candidate: Candidate) -> VerificationResult:
        start = time.time()

        if candidate.malformed or "x" not in candidate.result:
            return VerificationResult(
                verified=False,
                status=STATUS.MALFORMED_CANDIDATE,
                details={"reason": "candidate missing required field 'x'"},
                runtime_seconds=time.time() - start,
            )

        try:
            x = float(candidate.result["x"])
        except (TypeError, ValueError):
            return VerificationResult(
                verified=False,
                status=STATUS.MALFORMED_CANDIDATE,
                details={"reason": "x is not a real number"},
                runtime_seconds=time.time() - start,
            )

        if not math.isfinite(x):
            return VerificationResult(
                verified=False,
                status=STATUS.NUMERICAL_MISMATCH,
                details={"reason": "x is not finite"},
                runtime_seconds=time.time() - start,
            )

        lo, hi = problem.public["domain"]
        # Allow a small tolerance around the reported domain in case a
        # generator's bracket lands exactly on the boundary.
        margin = 1e-9 * max(1.0, abs(hi - lo))
        if not (lo - margin <= x <= hi + margin):
            return VerificationResult(
                verified=False,
                status=STATUS.INVALID_DOMAIN,
                details={"x": x, "domain": [lo, hi]},
                runtime_seconds=time.time() - start,
                message="Candidate root falls outside the required domain.",
            )

        try:
            f = build_callable(problem)
            residual = f(x)
        except Exception as exc:  # pragma: no cover - defensive
            return VerificationResult(
                verified=False,
                status=STATUS.SOLVER_FAILURE,
                details={"reason": f"independent function evaluation raised: {exc}"},
                runtime_seconds=time.time() - start,
            )

        abs_residual = abs(residual)
        verified = abs_residual <= self.residual_tolerance

        return VerificationResult(
            verified=verified,
            status=STATUS.VERIFIED if verified else STATUS.RESIDUAL_TOO_LARGE,
            details={
                "x": x,
                "residual": residual,
                "absolute_residual": abs_residual,
                "tolerance": self.residual_tolerance,
                "domain": [lo, hi],
            },
            runtime_seconds=time.time() - start,
            message=None if verified else "|f(x)| exceeds tolerance under independent evaluation.",
        )
