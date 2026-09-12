import json
import tempfile
import unittest

from mathn.core.models import Attempt, Candidate, Trajectory, VerificationResult
from mathn.logging.trajectory import load_trajectories, load_trajectory, save_trajectory


def _make_trajectory():
    c1 = Candidate(problem_id="p1", generator="scipy", attempt=1, result={"y_t_query": 0.51}, runtime_seconds=0.001)
    v1 = VerificationResult(
        verified=False,
        status="NUMERICAL_MISMATCH",
        details={"candidate": 0.51, "reference": 0.44626, "absolute_error": 0.06374},
        runtime_seconds=0.002,
    )
    c2 = Candidate(problem_id="p1", generator="scipy", attempt=2, result={"y_t_query": 0.44626}, runtime_seconds=0.001)
    v2 = VerificationResult(verified=True, status="VERIFIED", details={"absolute_error": 1.2e-8}, runtime_seconds=0.002)

    return Trajectory(
        problem_id="p1",
        generator="scipy",
        family="ode",
        subtype="exponential_decay",
        configuration={"max_attempts": 5, "atol": 1e-6, "rtol": 1e-5},
        attempts=[
            Attempt(attempt=1, candidate=c1, verification=v1),
            Attempt(attempt=2, candidate=c2, verification=v2),
        ],
        final_status="VERIFIED",
        attempts_used=2,
        total_runtime_seconds=0.006,
    )


class TestTrajectorySerialization(unittest.TestCase):
    def test_to_dict_has_required_top_level_fields(self):
        traj = _make_trajectory()
        d = traj.to_dict()
        for key in ["trajectory_id", "problem_id", "generator", "configuration", "attempts", "final_status", "attempts_used"]:
            self.assertIn(key, d)
        self.assertEqual(len(d["attempts"]), 2)
        # never save only the final answer -- both attempts must be present
        self.assertEqual(d["attempts"][0]["candidate"]["result"]["y_t_query"], 0.51)
        self.assertEqual(d["attempts"][1]["candidate"]["result"]["y_t_query"], 0.44626)

    def test_json_round_trip_via_json_dumps(self):
        traj = _make_trajectory()
        blob = json.dumps(traj.to_dict())
        reloaded = json.loads(blob)
        self.assertEqual(reloaded["final_status"], "VERIFIED")
        self.assertEqual(reloaded["attempts_used"], 2)

    def test_save_and_load_from_disk(self):
        traj = _make_trajectory()
        with tempfile.TemporaryDirectory() as tmp:
            path = save_trajectory(traj, tmp)
            self.assertTrue(path.endswith(".json"))
            reloaded = load_trajectory(path)
            self.assertEqual(reloaded["problem_id"], "p1")
            self.assertEqual(reloaded["final_status"], "VERIFIED")

            all_loaded = load_trajectories(tmp)
            self.assertEqual(len(all_loaded), 1)

    def test_numpy_scalars_are_json_safe(self):
        import numpy as np

        c = Candidate(problem_id="p2", generator="scipy", attempt=1, result={"x": np.float64(1.23)})
        v = VerificationResult(verified=True, status="VERIFIED", details={"residual": np.float64(1e-10)})
        traj = Trajectory(
            problem_id="p2", generator="scipy", family="roots", subtype="polynomial",
            attempts=[Attempt(attempt=1, candidate=c, verification=v)],
            final_status="VERIFIED", attempts_used=1,
        )
        # should not raise
        json.dumps(traj.to_dict())


if __name__ == "__main__":
    unittest.main()
