"""
Base abstraction for problem families.

A ProblemGenerator turns a seed into a reproducible Problem instance. The
same seed must always produce the same problem so experiments are
reproducible and auditable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Type

from mathn.core.models import Problem


class ProblemGenerator(ABC):
    """Abstract base for a synthetic problem-generation family."""

    #: Family identifier, e.g. "ode", "roots", "parameter_estimation", "symbolic".
    family: str = "base"

    @abstractmethod
    def generate(self, seed: int, index: int = 0) -> Problem:
        """Produce one reproducible Problem for the given seed.

        ``index`` is included in the problem_id to disambiguate multiple
        problems generated from a shared base seed within one dataset.
        """
        raise NotImplementedError

    def generate_many(self, n: int, base_seed: int = 0) -> List[Problem]:
        return [self.generate(seed=base_seed + i, index=i) for i in range(n)]


_REGISTRY: Dict[str, Type[ProblemGenerator]] = {}


def register_problem_generator(cls: Type[ProblemGenerator]) -> Type[ProblemGenerator]:
    """Class decorator that registers a ProblemGenerator under its
    ``family`` name (or, for generators covering several subtypes, its
    class name) so the CLI / config can refer to it by string.
    """
    key = getattr(cls, "registry_key", None) or cls.family
    _REGISTRY[key] = cls
    return cls


def get_problem_generator(key: str) -> Type[ProblemGenerator]:
    if key not in _REGISTRY:
        raise KeyError(f"Unknown problem generator '{key}'. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def available_problem_generators() -> List[str]:
    return sorted(_REGISTRY)
