"""
The generic retry controller.

This is the heart of the verifier-driven loop described in the project
README:

    problem -> generator -> candidate -> verifier -> PASS/FAIL
                   ^                                     |
                   |________________ feedback ___________|
                              (on FAIL, up to max_attempts)

The controller does not assume the generator supports feedback -- a
purely deterministic classical generator that ignores ``context`` entirely
is fully supported (it will just keep proposing the same candidate, which
the duplicate-detection logic will flag, and the loop will terminate at
max_attempts).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mathn.core.interfaces import CandidateGenerator, MalformedCandidateError, NumericalVerifier
from mathn.core.models import Attempt, Candidate, Problem, Trajectory, VerificationResult, _json_safe
from mathn.core.timeout import run_with_timeout
from mathn.verifiers.base import STATUS


def _canonical_key(result: Dict[str, Any]) -> str:
    """A stable hash of a candidate's result dict, used for exact
    duplicate-candidate detection."""
    safe = _json_safe(result)
    blob = json.dumps(safe, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class RetryConfig:
    max_attempts: int = 5
    generator_timeout_seconds: float = 30.0
    verifier_timeout_seconds: float = 30.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "generator_timeout_seconds": self.generator_timeout_seconds,
            "verifier_timeout_seconds": self.verifier_timeout_seconds,
        }


class RetryController:
    """Drives the generate -> verify -> retry loop for a single problem."""

    def __init__(self, generator: CandidateGenerator, verifier: NumericalVerifier, config: Optional[RetryConfig] = None):
        self.generator = generator
        self.verifier = verifier
        self.config = config or RetryConfig()

    def run(self, problem: Problem) -> Trajectory:
        start_time = time.time()
        attempts: List[Attempt] = []
        seen_candidates: Dict[str, int] = {}
        feedback_history: List[Dict[str, Any]] = []
        final_status = "UNKNOWN"

        generator_view = problem.for_generator()

        for attempt_num in range(1, self.config.max_attempts + 1):
            context = {
                "attempt": attempt_num,
                "max_attempts": self.config.max_attempts,
                # Snapshot the feedback accumulated so far -- generators
                # must see a frozen view of prior failures, not a live
                # reference that keeps growing as later attempts run.
                "feedback_history": list(feedback_history),
            }

            candidate, verification, duplicate, duplicate_of = self._run_one_attempt(
                problem, generator_view, context, attempt_num, seen_candidates
            )

            attempts.append(
                Attempt(
                    attempt=attempt_num,
                    candidate=candidate,
                    verification=verification,
                    duplicate=duplicate,
                    duplicate_of_attempt=duplicate_of,
                )
            )

            final_status = verification.status
            if verification.verified:
                final_status = STATUS.VERIFIED
                break

            feedback_history.append(verification.details)

            # A generator that keeps producing the exact same (wrong)
            # candidate cannot possibly converge by retrying further --
            # stop early rather than burning the full attempt budget, but
            # never loop forever regardless (max_attempts always bounds us).
            if duplicate and not self.generator.supports_feedback:
                break

        return Trajectory(
            problem_id=problem.problem_id,
            generator=self.generator.name,
            family=problem.family,
            subtype=problem.subtype,
            configuration=self.config.to_dict(),
            attempts=attempts,
            final_status=final_status,
            attempts_used=len(attempts),
            total_runtime_seconds=time.time() - start_time,
        )

    def _run_one_attempt(self, problem, generator_view, context, attempt_num, seen_candidates):
        # --- candidate generation ---
        try:
            candidate = run_with_timeout(
                self.generator.generate,
                args=(generator_view,),
                kwargs={"context": context},
                timeout_seconds=self.config.generator_timeout_seconds,
            )
        except TimeoutError:
            candidate = Candidate(
                problem_id=problem.problem_id,
                generator=self.generator.name,
                attempt=attempt_num,
                result={},
                timed_out=True,
            )
            verification = VerificationResult(
                verified=False,
                status=STATUS.TIMEOUT,
                details={"reason": "candidate generation exceeded the configured timeout", "phase": "generation"},
            )
            return candidate, verification, False, None
        except MalformedCandidateError as exc:
            candidate = Candidate(
                problem_id=problem.problem_id,
                generator=self.generator.name,
                attempt=attempt_num,
                result={},
                malformed=True,
                error=str(exc),
            )
            verification = VerificationResult(
                verified=False,
                status=STATUS.MALFORMED_CANDIDATE,
                details={"reason": str(exc)},
            )
            return candidate, verification, False, None
        except Exception as exc:  # pragma: no cover - defensive catch-all
            candidate = Candidate(
                problem_id=problem.problem_id,
                generator=self.generator.name,
                attempt=attempt_num,
                result={},
                malformed=True,
                error=f"{type(exc).__name__}: {exc}",
            )
            verification = VerificationResult(
                verified=False,
                status=STATUS.MALFORMED_CANDIDATE,
                details={"reason": f"generator raised an unhandled exception: {exc}"},
            )
            return candidate, verification, False, None

        # --- duplicate detection ---
        duplicate = False
        duplicate_of = None
        if not candidate.malformed:
            key = _canonical_key(candidate.result)
            if key in seen_candidates:
                duplicate = True
                duplicate_of = seen_candidates[key]
            else:
                seen_candidates[key] = attempt_num

        # --- independent verification ---
        try:
            verification = run_with_timeout(
                self.verifier.verify,
                args=(problem, candidate),
                timeout_seconds=self.config.verifier_timeout_seconds,
            )
        except TimeoutError:
            verification = VerificationResult(
                verified=False,
                status=STATUS.TIMEOUT,
                details={"reason": "verification exceeded the configured timeout", "phase": "verification"},
            )

        if duplicate and verification.verified is False:
            # Preserve the verifier's own status but make the duplicate
            # flag visible in the details for downstream analysis too.
            verification.details = dict(verification.details)
            verification.details["duplicate_candidate"] = True

        return candidate, verification, duplicate, duplicate_of
