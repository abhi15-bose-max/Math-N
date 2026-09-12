import unittest

from mathn.evaluation.failure_analysis import analyze_failures
from mathn.evaluation.metrics import compute_metrics


def _traj(problem_id, family, generator, final_status, attempts):
    """Build a minimal trajectory dict. ``attempts`` is a list of
    (verified: bool, status: str, details: dict) tuples.
    """
    atts = []
    for i, (verified, status, details) in enumerate(attempts, start=1):
        atts.append(
            {
                "attempt": i,
                "candidate": {"malformed": status == "MALFORMED_CANDIDATE", "runtime_seconds": 0.01},
                "verification": {"verified": verified, "status": status, "details": details, "runtime_seconds": 0.02},
                "duplicate": status == "DUPLICATE_CANDIDATE",
            }
        )
    return {
        "problem_id": problem_id,
        "family": family,
        "generator": generator,
        "final_status": final_status,
        "attempts_used": len(atts),
        "attempts": atts,
    }


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.trajectories = [
            _traj("p1", "ode", "scipy", "VERIFIED", [(True, "VERIFIED", {"absolute_error": 1e-9})]),
            _traj(
                "p2", "ode", "scipy", "VERIFIED",
                [
                    (False, "NUMERICAL_MISMATCH", {"absolute_error": 0.5}),
                    (True, "VERIFIED", {"absolute_error": 1e-8}),
                ],
            ),
            _traj("p3", "roots", "scipy", "RESIDUAL_TOO_LARGE", [(False, "RESIDUAL_TOO_LARGE", {"absolute_residual": 0.2})]),
            _traj("p4", "roots", "sympy", "VERIFIED", [(True, "VERIFIED", {"absolute_residual": 0.0})]),
            _traj("p5", "parameter_estimation", "scipy", "TIMEOUT", [(False, "TIMEOUT", {"reason": "timeout"})]),
        ]

    def test_totals(self):
        m = compute_metrics(self.trajectories)
        self.assertEqual(m["total_problems"], 5)
        self.assertEqual(m["final_successes"], 3)
        self.assertAlmostEqual(m["final_success_rate"], 3 / 5)

    def test_first_attempt_vs_final_success(self):
        m = compute_metrics(self.trajectories)
        # p1 and p4 verified on attempt 1; p2 needed a retry.
        self.assertEqual(m["first_attempt_successes"], 2)
        self.assertEqual(m["final_successes"], 3)

    def test_average_and_median_attempts(self):
        m = compute_metrics(self.trajectories)
        attempts_used = [1, 2, 1, 1, 1]
        self.assertAlmostEqual(m["average_attempts"], sum(attempts_used) / len(attempts_used))
        self.assertEqual(m["median_attempts"], 1)

    def test_timeout_count(self):
        m = compute_metrics(self.trajectories)
        self.assertEqual(m["timeout_count"], 1)

    def test_success_after_attempt_buckets(self):
        m = compute_metrics(self.trajectories)
        buckets = m["success_after_attempt"]
        # p1, p4 verified at attempt 1; p2 verified at attempt 2.
        self.assertEqual(buckets["success_after_attempt_1"], 2)
        self.assertEqual(buckets["success_after_attempt_2"], 3)
        self.assertEqual(buckets["success_after_attempt_5"], 3)

    def test_per_family_breakdown(self):
        m = compute_metrics(self.trajectories)
        self.assertEqual(m["by_family"]["ode"]["total"], 2)
        self.assertEqual(m["by_family"]["ode"]["final_successes"], 2)
        self.assertEqual(m["by_family"]["roots"]["total"], 2)
        self.assertEqual(m["by_family"]["roots"]["final_successes"], 1)

    def test_per_generator_breakdown(self):
        m = compute_metrics(self.trajectories)
        self.assertEqual(m["by_generator"]["scipy"]["total"], 4)
        self.assertEqual(m["by_generator"]["sympy"]["total"], 1)

    def test_empty_input(self):
        m = compute_metrics([])
        self.assertEqual(m["total_problems"], 0)


class TestFailureAnalysis(unittest.TestCase):
    def setUp(self):
        self.trajectories = [
            _traj("p1", "ode", "scipy", "VERIFIED", [(True, "VERIFIED", {"absolute_error": 1e-9})]),
            _traj(
                "p2", "ode", "scipy", "VERIFIED",
                [
                    (False, "NUMERICAL_MISMATCH", {"absolute_error": 0.5}),
                    (True, "VERIFIED", {"absolute_error": 1e-8}),
                ],
            ),
            _traj("p3", "roots", "scipy", "RESIDUAL_TOO_LARGE", [(False, "RESIDUAL_TOO_LARGE", {"absolute_residual": 0.2})]),
            _traj("p5", "parameter_estimation", "scipy", "TIMEOUT", [(False, "TIMEOUT", {"reason": "timeout"})]),
        ]

    def test_category_counts(self):
        r = analyze_failures(self.trajectories)
        self.assertEqual(r["category_counts"].get("NUMERICAL_MISMATCH"), 1)
        self.assertEqual(r["category_counts"].get("RESIDUAL_TOO_LARGE"), 1)
        self.assertEqual(r["category_counts"].get("TIMEOUT"), 1)

    def test_unsolved_problems(self):
        r = analyze_failures(self.trajectories)
        unsolved_ids = {u["problem_id"] for u in r["unsolved_problems"]}
        self.assertEqual(unsolved_ids, {"p3", "p5"})
        self.assertEqual(r["num_unsolved"], 2)

    def test_convergence_after_retry(self):
        r = analyze_failures(self.trajectories)
        self.assertEqual(r["num_converged_after_retry"], 1)
        self.assertEqual(r["convergence_after_retry"][0]["problem_id"], "p2")

    def test_largest_errors_sorted_descending(self):
        r = analyze_failures(self.trajectories)
        errors = [e["absolute_error"] for e in r["largest_errors_top10"]]
        self.assertEqual(errors, sorted(errors, reverse=True))


if __name__ == "__main__":
    unittest.main()
