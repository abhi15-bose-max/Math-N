import math
import unittest

from mathn.core.interfaces import MalformedCandidateError
from mathn.generators.scipy_generator import ScipyGenerator
from mathn.generators.sympy_generator import SympyGenerator
from mathn.problems.ode import ODEProblemGenerator
from mathn.problems.parameter_estimation import ParameterEstimationProblemGenerator
from mathn.problems.roots import RootProblemGenerator
from mathn.problems.symbolic import SymbolicProblemGenerator

GENERATORS = [ScipyGenerator(), SympyGenerator()]


class TestScipyGeneratorWellFormed(unittest.TestCase):
    def setUp(self):
        self.gen = ScipyGenerator()

    def test_ode_result_shape(self):
        p = ODEProblemGenerator().generate(seed=1, index=0)
        c = self.gen.generate(p.for_generator(), {"attempt": 1})
        self.assertFalse(c.malformed)
        self.assertIn("y_t_query", c.result)
        self.assertTrue(math.isfinite(c.result["y_t_query"]))

    def test_root_result_shape(self):
        p = RootProblemGenerator().generate(seed=1, index=0)
        c = self.gen.generate(p.for_generator(), {"attempt": 1})
        self.assertFalse(c.malformed)
        self.assertIn("x", c.result)

    def test_parameter_estimation_result_shape(self):
        p = ParameterEstimationProblemGenerator().generate(seed=1, index=0)
        c = self.gen.generate(p.for_generator(), {"attempt": 1})
        self.assertFalse(c.malformed)
        self.assertIn("parameters", c.result)
        for name in p.public["parameter_names"]:
            self.assertIn(name, c.result["parameters"])

    def test_symbolic_result_shape(self):
        p = SymbolicProblemGenerator().generate(seed=1, index=0)
        c = self.gen.generate(p.for_generator(), {"attempt": 1})
        self.assertFalse(c.malformed)
        self.assertIn("value", c.result)

    def test_precision_improves_with_attempt_number(self):
        # ODE: rtol schedule should tighten with attempt number, and the
        # resulting candidate should generally get closer to the truth.
        p = ODEProblemGenerator().generate(seed=77, index=0)  # exponential_decay
        reference = p.protected["reference_y"]
        errors = []
        for attempt in [1, 5]:
            c = self.gen.generate(p.for_generator(), {"attempt": attempt})
            errors.append(abs(c.result["y_t_query"] - reference))
        self.assertLessEqual(errors[1], errors[0])

    def test_unknown_family_raises_malformed(self):
        with self.assertRaises(MalformedCandidateError):
            self.gen.generate({"problem_id": "x", "family": "not_a_family", "subtype": "y"}, {"attempt": 1})


class TestSympyGeneratorWellFormed(unittest.TestCase):
    def setUp(self):
        self.gen = SympyGenerator()

    def test_ode_all_subtypes(self):
        gen_p = ODEProblemGenerator()
        for i in range(5):
            p = gen_p.generate(seed=21, index=i)
            c = self.gen.generate(p.for_generator(), {"attempt": 1})
            self.assertFalse(c.malformed, f"{p.subtype}: {c.error}")
            self.assertIn("y_t_query", c.result)

    def test_root_all_subtypes(self):
        gen_p = RootProblemGenerator()
        for i in range(3):
            p = gen_p.generate(seed=21, index=i)
            c = self.gen.generate(p.for_generator(), {"attempt": 1})
            self.assertFalse(c.malformed, f"{p.subtype}: {c.error}")
            self.assertIn("x", c.result)

    def test_symbolic_all_subtypes(self):
        gen_p = SymbolicProblemGenerator()
        for i in range(2):
            p = gen_p.generate(seed=21, index=i)
            c = self.gen.generate(p.for_generator(), {"attempt": 1})
            self.assertFalse(c.malformed, f"{p.subtype}: {c.error}")
            self.assertIn("value", c.result)

    def test_generator_does_not_see_protected_fields(self):
        # SympyGenerator.generate receives only the dict produced by
        # Problem.for_generator(); assert that dict truly lacks protected
        # information regardless of which family is used.
        p = ParameterEstimationProblemGenerator().generate(seed=3, index=0)
        view = p.for_generator()
        c = self.gen.generate(view, {"attempt": 1})
        self.assertNotIn("true_parameters", view)
        self.assertFalse(c.malformed)


if __name__ == "__main__":
    unittest.main()
