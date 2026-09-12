"""Problem families. Importing this package registers the built-in
problem generators (ode, roots, parameter_estimation, symbolic) with the
problem registry."""

from mathn.problems import ode  # noqa: F401
from mathn.problems import roots  # noqa: F401
from mathn.problems import parameter_estimation  # noqa: F401
from mathn.problems import symbolic  # noqa: F401
