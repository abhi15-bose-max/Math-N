"""
Basic failure analysis over a set of trajectories.

The goal here is diagnosis, not ranking: which failure categories occur,
which problems never got verified, how much retrying helped, and where
the largest numerical errors showed up.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List

from mathn.evaluation.metrics import _extract_abs_error

CATEGORIES = [
    "NUMERICAL_MISMATCH",
    "RESIDUAL_TOO_LARGE",
    "INVALID_DOMAIN",
    "CONSTRAINT_VIOLATION",
    "SOLVER_FAILURE",
    "TIMEOUT",
    "MALFORMED_CANDIDATE",
    "DUPLICATE_CANDIDATE",
    "OTHER",
]


def analyze_failures(trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
    category_counts = Counter()
    category_by_family = defaultdict(Counter)
    category_by_generator = defaultdict(Counter)

    unsolved_problems = []
    problems_requiring_retries = []
    convergence_after_retry = []
    largest_errors = []

    for traj in trajectories:
        attempts = traj.get("attempts", [])
        family = traj.get("family", "unknown")
        generator = traj.get("generator", "unknown")
        final_status = traj.get("final_status")
        attempts_used = traj.get("attempts_used", len(attempts))

        if final_status != "VERIFIED":
            unsolved_problems.append(
                {
                    "problem_id": traj.get("problem_id"),
                    "family": family,
                    "generator": generator,
                    "final_status": final_status,
                    "attempts_used": attempts_used,
                }
            )

        if attempts_used > 1:
            problems_requiring_retries.append(traj.get("problem_id"))
            if final_status == "VERIFIED":
                convergence_after_retry.append(
                    {
                        "problem_id": traj.get("problem_id"),
                        "family": family,
                        "generator": generator,
                        "attempts_used": attempts_used,
                    }
                )

        for a in attempts:
            ver = a.get("verification", {})
            status = ver.get("status", "OTHER")
            if status == "VERIFIED":
                continue
            cat = status if status in CATEGORIES else "OTHER"
            category_counts[cat] += 1
            category_by_family[family][cat] += 1
            category_by_generator[generator][cat] += 1

        final_attempt = attempts[-1] if attempts else None
        if final_attempt is not None and final_status != "VERIFIED":
            details = final_attempt["verification"].get("details", {}) or {}
            abs_err = _extract_abs_error(details)
            if abs_err is not None:
                largest_errors.append(
                    {
                        "problem_id": traj.get("problem_id"),
                        "family": family,
                        "generator": generator,
                        "absolute_error": abs_err,
                        "status": final_attempt["verification"].get("status"),
                    }
                )

    largest_errors.sort(key=lambda d: d["absolute_error"], reverse=True)

    return {
        "category_counts": dict(category_counts),
        "category_by_family": {k: dict(v) for k, v in category_by_family.items()},
        "category_by_generator": {k: dict(v) for k, v in category_by_generator.items()},
        "unsolved_problems": unsolved_problems,
        "num_unsolved": len(unsolved_problems),
        "problems_requiring_retries": problems_requiring_retries,
        "num_requiring_retries": len(problems_requiring_retries),
        "convergence_after_retry": convergence_after_retry,
        "num_converged_after_retry": len(convergence_after_retry),
        "largest_errors_top10": largest_errors[:10],
    }
