import time
import unittest
from typing import Any, Dict, Optional

from mathn.core.interfaces import CandidateGenerator, MalformedCandidateError, NumericalVerifier
from mathn.core.models import Candidate, Problem, VerificationResult
from mathn.core.runner import RetryConfig, RetryController
from mathn.verifiers.base import STATUS

TOY_PROBLEM = Problem(
    problem_id="toy_001",
    family="toy",
    subtype="toy",
    seed=0,
    public={"target": 42.0},
    protected={},
)


class ToyVerifier(NumericalVerifier):
    """Independently checks candidate['value'] against problem.public['target']."""

    def verify(self, problem: Problem, candidate: Candidate) -> VerificationResult:
        if candidate.malformed or "value" not in candidate.result:
            return VerificationResult(verified=False, status=STATUS.MALFORMED_CANDIDATE, details={})
        value = candidate.result["value"]
        target = problem.public["target"]
        err = abs(value - target)
        verified = err <= 1e-6
        return VerificationResult(
            verified=verified,
            status=STATUS.VERIFIED if verified else STATUS.NUMERICAL_MISMATCH,
            details={"absolute_error": err, "value": value, "target": target},
        )


class AlwaysCorrectGenerator(CandidateGenerator):
    name = "always_correct"
    supports_feedback = False

    def generate(self, problem, context=None):
        return Candidate(problem_id=problem["problem_id"], generator=self.name, attempt=(context or {}).get("attempt", 1), result={"value": problem["target"]})


class AlwaysWrongGenerator(CandidateGenerator):
    """Deterministic and wrong -- every attempt produces the exact same
    incorrect candidate, so it should trigger duplicate-based early stop."""

    name = "always_wrong"
    supports_feedback = False

    def generate(self, problem, context=None):
        return Candidate(problem_id=problem["problem_id"], generator=self.name, attempt=(context or {}).get("attempt", 1), result={"value": 0.0})


class ImprovingGenerator(CandidateGenerator):
    """Gets closer to the target with each retry, using only the attempt
    number from context (never peeking at the target itself directly --
    it approaches it via a fixed convergent schedule)."""

    name = "improving"
    supports_feedback = True

    def generate(self, problem, context=None):
        attempt = (context or {}).get("attempt", 1)
        target = problem["target"]
        # Converge geometrically: error shrinks by 10x each attempt.
        error = 10.0 ** (-attempt)
        return Candidate(problem_id=problem["problem_id"], generator=self.name, attempt=attempt, result={"value": target + error})


class NeverConvergesGenerator(CandidateGenerator):
    """Uses feedback (so it isn't stopped by duplicate detection) but never
    actually reaches the target -- exercises max_attempts enforcement."""

    name = "never_converges"
    supports_feedback = True

    def generate(self, problem, context=None):
        attempt = (context or {}).get("attempt", 1)
        target = problem["target"]
        return Candidate(problem_id=problem["problem_id"], generator=self.name, attempt=attempt, result={"value": target + 1.0 + 0.001 * attempt})


class AlwaysRaisesGenerator(CandidateGenerator):
    name = "always_raises"
    supports_feedback = False

    def generate(self, problem, context=None):
        raise MalformedCandidateError("cannot produce a candidate for this toy problem")


class SlowGenerator(CandidateGenerator):
    name = "slow"
    supports_feedback = False

    def generate(self, problem, context=None):
        time.sleep(2.0)
        return Candidate(problem_id=problem["problem_id"], generator=self.name, attempt=1, result={"value": 42.0})


class TestRetryController(unittest.TestCase):
    def setUp(self):
        self.verifier = ToyVerifier()

    def test_first_attempt_success(self):
        rc = RetryController(AlwaysCorrectGenerator(), self.verifier, RetryConfig(max_attempts=5))
        traj = rc.run(TOY_PROBLEM)
        self.assertEqual(traj.final_status, "VERIFIED")
        self.assertEqual(traj.attempts_used, 1)
        self.assertTrue(traj.first_attempt_verified)

    def test_duplicate_detection_stops_early_without_feedback_support(self):
        rc = RetryController(AlwaysWrongGenerator(), self.verifier, RetryConfig(max_attempts=5))
        traj = rc.run(TOY_PROBLEM)
        self.assertNotEqual(traj.final_status, "VERIFIED")
        self.assertLess(traj.attempts_used, 5, "should stop before exhausting the retry budget")
        self.assertTrue(traj.attempts[-1].duplicate)
        self.assertEqual(traj.attempts[-1].duplicate_of_attempt, 1)

    def test_retry_improves_and_eventually_verifies(self):
        rc = RetryController(ImprovingGenerator(), self.verifier, RetryConfig(max_attempts=6))
        traj = rc.run(TOY_PROBLEM)
        self.assertEqual(traj.final_status, "VERIFIED")
        self.assertGreater(traj.attempts_used, 1)
        errors = [a.verification.details["absolute_error"] for a in traj.attempts]
        self.assertEqual(errors, sorted(errors, reverse=True), "error should shrink monotonically across retries")

    def test_max_attempts_enforced(self):
        rc = RetryController(NeverConvergesGenerator(), self.verifier, RetryConfig(max_attempts=4))
        traj = rc.run(TOY_PROBLEM)
        self.assertNotEqual(traj.final_status, "VERIFIED")
        self.assertEqual(traj.attempts_used, 4)

    def test_malformed_candidate_recorded_every_attempt(self):
        rc = RetryController(AlwaysRaisesGenerator(), self.verifier, RetryConfig(max_attempts=3))
        traj = rc.run(TOY_PROBLEM)
        self.assertEqual(traj.final_status, "MALFORMED_CANDIDATE")
        self.assertEqual(traj.attempts_used, 3)
        for a in traj.attempts:
            self.assertTrue(a.candidate.malformed)

    def test_timeout_is_recorded_not_treated_as_math_failure(self):
        rc = RetryController(
            SlowGenerator(), self.verifier,
            RetryConfig(max_attempts=1, generator_timeout_seconds=0.3),
        )
        traj = rc.run(TOY_PROBLEM)
        self.assertEqual(traj.final_status, "TIMEOUT")
        self.assertTrue(traj.attempts[0].candidate.timed_out)

    def test_feedback_history_passed_to_generator(self):
        seen_contexts = []

        class RecordingGenerator(CandidateGenerator):
            name = "recording"
            supports_feedback = True

            def generate(self, problem, context=None):
                seen_contexts.append(dict(context or {}))
                attempt = (context or {}).get("attempt", 1)
                target = problem["target"]
                value = target if attempt >= 2 else target + 5.0
                return Candidate(problem_id=problem["problem_id"], generator=self.name, attempt=attempt, result={"value": value})

        rc = RetryController(RecordingGenerator(), self.verifier, RetryConfig(max_attempts=3))
        rc.run(TOY_PROBLEM)
        self.assertEqual(len(seen_contexts), 2)
        self.assertEqual(seen_contexts[0]["feedback_history"], [])
        self.assertEqual(len(seen_contexts[1]["feedback_history"]), 1)
        self.assertIn("absolute_error", seen_contexts[1]["feedback_history"][0])


if __name__ == "__main__":
    unittest.main()
