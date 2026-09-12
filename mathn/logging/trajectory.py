"""
Trajectory persistence.

Every problem execution produces a full Trajectory (all attempts, all
candidates, all verification results) which is saved as JSON. We never
save only the final answer -- the point of this framework is that the
full generate/verify/retry history is the experimental evidence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List

from mathn.core.models import Trajectory


def save_trajectory(trajectory: Trajectory, directory: str) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{trajectory.problem_id}__{trajectory.generator}.json")
    with open(path, "w") as f:
        json.dump(trajectory.to_dict(), f, indent=2, sort_keys=False)
    return path


def load_trajectory(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_trajectories(directory: str) -> List[dict]:
    paths = sorted(Path(directory).glob("*.json"))
    return [load_trajectory(str(p)) for p in paths]


def save_trajectories(trajectories: Iterable[Trajectory], directory: str) -> List[str]:
    return [save_trajectory(t, directory) for t in trajectories]
