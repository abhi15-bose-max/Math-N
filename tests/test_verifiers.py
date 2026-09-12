import math
import unittest

from mathn.core.models import Candidate
from mathn.problems.ode import ODEProblemGenerator
from mathn.problems.parameter_estimation import ParameterEstimationProblemGenerator
from mathn.problems.roots import RootProblemGenerator
from mathn.problems.symbolic import SymbolicProblemGenerator
from mathn.verifiers.ode_verifier import ODEVerifier
from mathn.verifiers.parameter_verifier import ParameterVerifier
from mathn.verifiers.root_verifier import RootVerifier
from mathn.verifiers.symbolic_verifier import SymbolicVerifier


def _cand(problem_id, **result):
    return Candidate(problem_id=problem_id, generator="test", attempt=1, result=result)


class TestODEVerifier(unittest.TestCase):
    def setUp(self):
        self.v = ODEVerifier(atol=1e-6, rtol=1e-5)
        self.p = ODEProblemGenerator().generate(seed=1, index=0)  # exponential_decay

    def test_correct_candidate_verified(self):
        reference = self.p.protected["reference_y"]
        c = _cand(self.p.problem_id, y_t_query=reference)
        res = self.v.verify(self.p, c)
        self.assertTrue(res.verified)
        self.assertEqual(res.status, "VERIFIED")

    def test_incorrect_candidate_rejected_with_feedback(self):
        reference = self.p.protected["reference_y"]
        wrong = reference + 10.0  # far outside tolerance
        c = _cand(self.p.problem_id, y_t_query=wrong)
        res = self.v.verify(self.p, c)
        self.assertFalse(res.verified)
        self.assertEqual(res.status, "NUMERICAL_MISMATCH")
        self.assertIn("absolute_error", res.details)
        self.assertIn("reference", res.details)
        self.assertAlmostEqual(res.details["absolute_error"], 10.0, places=6)

    def test_malformed_candidate(self):
        c = _cand(self.p.problem_id)  # missing y_t_query
        res = self.v.verify(self.p, c)
        self.assertFalse(res.verified)
        self.assertEqual(res.status, "MALFORMED_CANDIDATE")

    def test_tolerance_boundary(self):
        reference = self.p.protected["reference_y"]
        threshold = self.v.atol + self.v.rtol * abs(reference)
        just_inside = reference + threshold * 0.5
        just_outside = reference + threshold * 2.0
        self.assertTrue(self.v.verify(self.p, _cand(self.p.problem_id, y_t_query=just_inside)).verified)
        self.assertFalse(self.v.verify(self.p, _cand(self.p.problem_id, y_t_query=just_outside)).verified)


class TestRootVerifier(unittest.TestCase):
    def setUp(self):
        self.v = RootVerifier(residual_tolerance=1e-6)
        self.p = RootProblemGenerator().generate(seed=2, index=1)  # transcendental: cos(x)-x

    def test_correct_root_verified(self):
        root = self.p.protected["constructed_root"]
        res = self.v.verify(self.p, _cand(self.p.problem_id, x=root))
        self.assertTrue(res.verified)

    def test_wrong_root_rejected(self):
        res = self.v.verify(self.p, _cand(self.p.problem_id, x=0.1))
        self.assertFalse(res.verified)
        self.assertEqual(res.status, "RESIDUAL_TOO_LARGE")

    def test_out_of_domain_rejected(self):
        lo, hi = self.p.public["domain"]
        res = self.v.verify(self.p, _cand(self.p.problem_id, x=hi + 100.0))
        self.assertFalse(res.verified)
        self.assertEqual(res.status, "INVALID_DOMAIN")

    def test_nonfinite_rejected(self):
        res = self.v.verify(self.p, _cand(self.p.problem_id, x=float("nan")))
        self.assertFalse(res.verified)


class TestParameterVerifier(unittest.TestCase):
    def setUp(self):
        self.v = ParameterVerifier(rmse_threshold=0.05)
        self.p = ParameterEstimationProblemGenerator(noise_std_frac=0.0).generate(seed=3, index=0)

    def test_true_parameters_pass_with_zero_noise(self):
        true_params = self.p.protected["true_parameters"]
        res = self.v.verify(self.p, _cand(self.p.problem_id, parameters=true_params))
        self.assertTrue(res.verified)
        self.assertLess(res.details["simulation_rmse"], 1e-6)

    def test_wildly_wrong_parameters_rejected(self):
        wrong = {k: v * 100 + 50 for k, v in self.p.protected["true_parameters"].items()}
        res = self.v.verify(self.p, _cand(self.p.problem_id, parameters=wrong))
        self.assertFalse(res.verified)

    def test_missing_parameter_is_malformed(self):
        res = self.v.verify(self.p, _cand(self.p.problem_id, parameters={}))
        self.assertEqual(res.status, "MALFORMED_CANDIDATE")

    def test_reports_parameter_error_diagnostic(self):
        true_params = self.p.protected["true_parameters"]
        res = self.v.verify(self.p, _cand(self.p.problem_id, parameters=true_params))
        self.assertIsNotNone(res.details["parameter_error"])
        for err in res.details["parameter_error"].values():
            self.assertLess(err, 1e-4)


class TestSymbolicVerifier(unittest.TestCase):
    def setUp(self):
        self.v = SymbolicVerifier(atol=1e-4, rtol=1e-4)

    def test_derivative_correct(self):
        p = SymbolicProblemGenerator().generate(seed=4, index=0)  # derivative_at_point
        from mathn.problems.symbolic import sympy_expression
        import sympy as sp

        x, expr = sympy_expression(p)
        deriv = float(sp.diff(expr, x).subs(x, p.public["point"]))
        res = self.v.verify(p, _cand(p.problem_id, value=deriv))
        self.assertTrue(res.verified)

    def test_integral_correct(self):
        p = SymbolicProblemGenerator().generate(seed=4, index=1)  # definite_integral
        from mathn.problems.symbolic import sympy_expression
        import sympy as sp

        x, expr = sympy_expression(p)
        a, b = p.public["interval"]
        antideriv = sp.integrate(expr, x)
        value = float(sp.N(antideriv.subs(x, b) - antideriv.subs(x, a)))
        res = self.v.verify(p, _cand(p.problem_id, value=value))
        self.assertTrue(res.verified)

    def test_wrong_value_rejected(self):
        p = SymbolicProblemGenerator().generate(seed=4, index=0)
        res = self.v.verify(p, _cand(p.problem_id, value=1e6))
        self.assertFalse(res.verified)


if __name__ == "__main__":
    unittest.main()
