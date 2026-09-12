"""
Convenience re-exports and a small registry so the CLI can look up
candidate generators by name (e.g. "scipy", "sympy") without importing
every module explicitly.
"""

from __future__ import annotations

from typing import Dict, List, Type

from mathn.core.interfaces import CandidateGenerator, MalformedCandidateError

__all__ = ["CandidateGenerator", "MalformedCandidateError", "register_generator", "get_generator", "available_generators"]

_REGISTRY: Dict[str, Type[CandidateGenerator]] = {}


def register_generator(cls: Type[CandidateGenerator]) -> Type[CandidateGenerator]:
    _REGISTRY[cls.name] = cls
    return cls


def get_generator(name: str) -> Type[CandidateGenerator]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown candidate generator '{name}'. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_generators() -> List[str]:
    return sorted(_REGISTRY)
