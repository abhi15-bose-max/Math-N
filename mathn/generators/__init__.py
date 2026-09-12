"""Candidate generators. Importing this package registers the built-in
generators (scipy, sympy) with the generator registry."""

from mathn.generators import scipy_generator  # noqa: F401
from mathn.generators import sympy_generator  # noqa: F401
