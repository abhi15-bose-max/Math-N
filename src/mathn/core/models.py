"""
Core data models for MATH-N.

These dataclasses define the common vocabulary shared by every problem
family, every candidate generator, and every verifier:

    Problem              -- a mathematical problem instance
    Candidate            -- a proposed solution to a Problem
    VerificationResult   -- the outcome of independently checking a Candidate
    Attempt              -- one (candidate, verification) pair in a trajectory
    Trajectory           -- the full generate -> verify -> retry history

None of these types know anything about *how* a candidate was produced
(numerical algorithm, symbolic solver, optimizer, or, in a future version,
an AI model). They only describe the data that flows through the loop.
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


def _json_safe(value: Any) -> Any:
    """Recursively convert numpy scalars / arrays and other non-JSON-native
    objects into plain Python types so trajectories can always be serialized
    with the standard ``json`` module.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dependency in practice
        np = None

    if np is not None:
        if isinstance(value, np.ndarray):
            return [_json_safe(v) for v in value.tolist()]
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, complex):
        return {"re": value.real, "im": value.imag}
    return value


@dataclass
class Problem:
    """A single mathematical problem instance.

    ``public`` contains everything a candidate generator is allowed to see:
    the query, known parameters, observed data, domain constraints, etc.

    ``protected`` contains information that must NOT be given to the
    candidate generator: ground-truth reference values, true hidden
    parameters used to synthesize observations, etc. Verifiers may use
    ``protected`` for diagnostic purposes, but a verifier's PASS/FAIL
    decision should whenever possible be based on an independent
    recomputation rather than a bare comparison against a stored value.
    """

    problem_id: str
    family: str
    subtype: str
    seed: int
    public: Dict[str, Any] = field(default_factory=dict)
    protected: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def for_generator(self) -> Dict[str, Any]:
        """The view of this problem that a candidate generator is allowed
        to see. Never includes ``protected`` fields.
        """
        view = {
            "problem_id": self.problem_id,
            "family": self.family,
            "subtype": self.subtype,
        }
        view.update(copy.deepcopy(self.public))
        return view

    def to_dict(self, include_protected: bool = True) -> Dict[str, Any]:
        d = {
            "problem_id": self.problem_id,
            "family": self.family,
            "subtype": self.subtype,
            "seed": self.seed,
            "public": _json_safe(self.public),
            "metadata": _json_safe(self.metadata),
        }
        if include_protected:
            d["protected"] = _json_safe(self.protected)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Problem":
        return Problem(
            problem_id=d["problem_id"],
            family=d["family"],
            subtype=d["subtype"],
            seed=d.get("seed", 0),
            public=d.get("public", {}),
            protected=d.get("protected", {}),
            metadata=d.get("metadata", {}),
        )


@dataclass
class Candidate:
    """A proposed solution to a Problem, produced by some candidate
    generator. The generator abstraction is deliberately not tied to any
    particular method (classical numerical, symbolic, optimization,
    probabilistic, or, in later versions, AI-based).
    """

    problem_id: str
    generator: str
    attempt: int
    result: Dict[str, Any] = field(default_factory=dict)
    raw: Optional[Any] = None
    runtime_seconds: Optional[float] = None
    malformed: bool = False
    error: Optional[str] = None
    timed_out: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["result"] = _json_safe(d["result"])
        d["raw"] = _json_safe(d["raw"]) if d["raw"] is not None else None
        return d


@dataclass
class VerificationResult:
    """The outcome of independently checking a Candidate against a
    Problem. ``verified`` is the binary PASS/FAIL signal; ``status`` gives
    a machine-readable failure category (see evaluation/failure_analysis.py
    for the canonical category list); ``details`` carries whatever
    numerical evidence (errors, residuals, thresholds) supports the
    decision.
    """

    verified: bool
    status: str
    details: Dict[str, Any] = field(default_factory=dict)
    runtime_seconds: Optional[float] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["details"] = _json_safe(d["details"])
        return d


@dataclass
class Attempt:
    attempt: int
    candidate: Candidate
    verification: VerificationResult
    duplicate: bool = False
    duplicate_of_attempt: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt": self.attempt,
            "candidate": self.candidate.to_dict(),
            "verification": self.verification.to_dict(),
            "duplicate": self.duplicate,
            "duplicate_of_attempt": self.duplicate_of_attempt,
        }


@dataclass
class Trajectory:
    problem_id: str
    generator: str
    family: str
    subtype: str
    configuration: Dict[str, Any] = field(default_factory=dict)
    attempts: List[Attempt] = field(default_factory=list)
    final_status: str = "UNKNOWN"
    attempts_used: int = 0
    total_runtime_seconds: float = 0.0
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "problem_id": self.problem_id,
            "generator": self.generator,
            "family": self.family,
            "subtype": self.subtype,
            "created_at": self.created_at,
            "configuration": _json_safe(self.configuration),
            "attempts": [a.to_dict() for a in self.attempts],
            "final_status": self.final_status,
            "attempts_used": self.attempts_used,
            "total_runtime_seconds": self.total_runtime_seconds,
        }

    @property
    def verified(self) -> bool:
        return self.final_status == "VERIFIED"

    @property
    def first_attempt_verified(self) -> bool:
        return bool(self.attempts) and self.attempts[0].verification.verified
