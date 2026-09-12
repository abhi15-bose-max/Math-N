"""
Convenience re-export so verifier implementations can do
``from mathn.verifiers.base import NumericalVerifier`` instead of reaching
into ``mathn.core.interfaces`` directly. Also defines the canonical set
of status codes used across all verifiers (see evaluation/failure_analysis
for how these are grouped and reported).
"""

from __future__ import annotations

from mathn.core.interfaces import NumericalVerifier
from mathn.core.models import VerificationResult

__all__ = ["NumericalVerifier", "VerificationResult", "STATUS"]


class STATUS:
    VERIFIED = "VERIFIED"
    NUMERICAL_MISMATCH = "NUMERICAL_MISMATCH"
    RESIDUAL_TOO_LARGE = "RESIDUAL_TOO_LARGE"
    INVALID_DOMAIN = "INVALID_DOMAIN"
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"
    SOLVER_FAILURE = "SOLVER_FAILURE"
    TIMEOUT = "TIMEOUT"
    MALFORMED_CANDIDATE = "MALFORMED_CANDIDATE"
    DUPLICATE_CANDIDATE = "DUPLICATE_CANDIDATE"
    OTHER = "OTHER"
