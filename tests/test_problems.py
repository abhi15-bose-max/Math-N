import math
import unittest

from mathn.problems.ode import ODEProblemGenerator, SUBTYPES as ODE_SUBTYPES
from mathn.problems.roots import RootProblemGenerator, SUBTYPES as ROOT_SUBTYPES, build_callable as root_callable
from mathn.problems.parameter_estimation import (
    ParameterEstimationProblemGenerator,
    SUBTYPES as PARAM_SUBTYPES,
    forward_model,
)
from mathn.problems.symbolic import SymbolicProblemGenerator, SUBTYPES as SYMBOLIC_SUBTYPES, build_callable as sym_callable


class TestODEProblems(unittest.TestCase):
    def test_all_subtypes_generate(self):
        gen = ODEProblemGenerator()
        for i, subtype in enumerate(ODE_SUBTYPES):
            p = gen.generate(seed=1, index=i)
            self.assertEqual(p.family, "ode")
            self.assertEqual(p.subtype, subtype)
            self.assertIn("t_query", p.public)
            self.assertIn("reference_y", p.protected)
            self.assertTrue(math.isfinite(p.protected["reference_y"]))

    def test_reproducibility(self):
        gen = ODEProblemGenerator()
        p1 = gen.generate(seed=123, index=0)
        p2 = gen.generate(seed=123, index=0)
        self.assertEqual(p1.public, p2.public)
        self.assertEqual(p1.protected, p2.protected)
        self.assertEqual(p1.problem_id, p2.problem_id)

    def test_different_seed_differs(self):
        gen = ODEProblemGenerator()
        p1 = gen.generate(seed=1, index=0)
        p2 = gen.generate(seed=2, index=0)
        self.assertNotEqual(p1.public["params"], p2.public["params"])

    def test_generator_view_hides_protected(self):
        gen = ODEProblemGenerator()
        p = gen.generate(seed=1, index=0)
        view = p.for_generator()
        self.assertNotIn("reference_y", view)
        self.assertNotIn("protected", view)


class TestDatasetCrossProcessReproducibility(unittest.TestCase):
    """Regression test: dataset generation must be reproducible ACROSS
    process runs, not just within one Python process. This guards against
    accidentally depending on Python's randomized string hash() (PEP 456)
    for anything seed-related.
    """

    def test_build_v1_dataset_is_stable_across_fresh_interpreters(self):
        import subprocess
        import sys

        script = (
            "from mathn.dataset import build_v1_dataset;"
            "ps = build_v1_dataset(problems_per_family=3, base_seed=42);"
            "print([p.problem_id for p in ps])"
        )
        out1 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout
        out2 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout
        self.assertEqual(out1, out2)


class TestRootProblems(unittest.TestCase):
    def test_all_subtypes_have_sign_change_or_root_in_domain(self):
        gen = RootProblemGenerator()
        for i, subtype in enumerate(ROOT_SUBTYPES):
            p = gen.generate(seed=3, index=i)
            f = root_callable(p)
            lo, hi = p.public["domain"]
            root = p.protected["constructed_root"]
            self.assertTrue(lo - 1e-6 <= root <= hi + 1e-6, f"{subtype}: root {root} not in domain {[lo, hi]}")
            self.assertAlmostEqual(f(root), 0.0, places=6)

    def test_reproducibility(self):
        gen = RootProblemGenerator()
        p1 = gen.generate(seed=55, index=1)
        p2 = gen.generate(seed=55, index=1)
        self.assertEqual(p1.public, p2.public)


class TestParameterEstimationProblems(unittest.TestCase):
    def test_all_subtypes_generate_observations(self):
        gen = ParameterEstimationProblemGenerator(n_observations=8)
        for i, subtype in enumerate(PARAM_SUBTYPES):
            p = gen.generate(seed=9, index=i)
            self.assertEqual(len(p.public["t_observations"]), 8)
            self.assertEqual(len(p.public["y_observations"]), 8)
            self.assertIn("true_parameters", p.protected)

    def test_true_parameters_not_exposed_to_generator(self):
        gen = ParameterEstimationProblemGenerator()
        p = gen.generate(seed=9, index=0)
        view = p.for_generator()
        self.assertNotIn("true_parameters", view)
        self.assertNotIn("clean_observations", view)

    def test_observations_are_noisy_perturbation_of_forward_model(self):
        gen = ParameterEstimationProblemGenerator(noise_std_frac=0.0)
        p = gen.generate(seed=42, index=0)
        # zero noise => observations should exactly equal the forward model
        # evaluated at the true parameters (sanity check on generation).
        import numpy as np

        t = np.array(p.public["t_observations"])
        y = np.array(p.public["y_observations"])
        predicted = forward_model(p.subtype, p.protected["true_parameters"], t, p.public["fixed"])
        self.assertTrue(np.allclose(y, predicted, atol=1e-8))


class TestSymbolicProblems(unittest.TestCase):
    def test_all_subtypes_generate(self):
        gen = SymbolicProblemGenerator()
        for i, subtype in enumerate(SYMBOLIC_SUBTYPES):
            p = gen.generate(seed=13, index=i)
            f = sym_callable(p)
            self.assertTrue(math.isfinite(f(0.1)))

    def test_reproducibility(self):
        gen = SymbolicProblemGenerator()
        p1 = gen.generate(seed=13, index=0)
        p2 = gen.generate(seed=13, index=0)
        self.assertEqual(p1.public, p2.public)


if __name__ == "__main__":
    unittest.main()
