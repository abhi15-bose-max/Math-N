"""
Core abstractions for MATH-N.

These interfaces are intentionally generic. A ``CandidateGenerator`` is
*anything* that can look at a problem and propose a candidate solution:
a classical numerical routine, a symbolic solver, an optimizer, an MCMC
sampler, a neural network, an LLM, or something not yet invented. Nothing
in this module assumes AI is involved anywhere.

The framework's core claim is architectural, not algorithmic:

    GENERATION and VERIFICATION are separate, and a candidate generator
    is never trusted to grade its own output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from mathn.core.models import Candidate, Problem, VerificationResult


class MalformedCandidateError(Exception):
    """Raised by a CandidateGenerator when it cannot produce a well-formed
    candidate at all (as opposed to producing a wrong-but-well-formed one).
    """


class CandidateGenerator(ABC):
    """Abstract base class for anything that proposes candidate solutions.

    Subclasses must NOT assume they are being called by an LLM-driven
    pipeline; ``context`` is a plain dictionary of retry/feedback state
    that classical, non-adaptive generators are free to ignore entirely.
    """

    #: Short machine-readable name, e.g. "scipy", "sympy", "optimizer".
    name: str = "base"

    #: Whether this generator can make use of feedback from a previous
    #: failed attempt (``context["feedback_history"]``). The retry
    #: controller does not require this -- generators that ignore feedback
    #: are fully supported (e.g. a deterministic classical method that
    #: would just produce the same answer every time).
    supports_feedback: bool = False

    @abstractmethod
    def generate(self, problem: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Candidate:
        """Produce a Candidate for ``problem`` (the generator-facing view,
        i.e. ``Problem.for_generator()`` -- protected fields are never
        passed in). ``context`` may include:

            attempt              -- 1-indexed attempt number
            max_attempts         -- configured retry budget
            feedback_history     -- list of prior VerificationResult.details

        Implementations should raise ``MalformedCandidateError`` if they
        cannot produce a structurally valid candidate at all.
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.__class__.__name__} name={self.name!r}>"


class NumericalVerifier(ABC):
    """Abstract base class for independent verification.

    A verifier must not simply trust the candidate generator's own
    computation. Wherever practical it should recompute a reference
    quantity via a different computational path (a different solver,
    method, or direct function evaluation) and compare against the
    candidate within an explicit, configured tolerance.
    """

    @abstractmethod
    def verify(self, problem: Problem, candidate: Candidate) -> VerificationResult:
        """Independently check ``candidate`` against ``problem``.

        Receives the *full* Problem (including protected fields) because
        the verifier -- unlike the generator -- is trusted infrastructure.
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.__class__.__name__}>"
