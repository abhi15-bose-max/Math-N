"""
Aggregate metrics over a set of trajectories.

Trajectories are consumed as plain dicts (i.e. ``Trajectory.to_dict()``
or JSON loaded straight off disk from ``logging/trajectory.py``) so this
module has no dependency on how the trajectories were produced.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional


def _extract_abs_error(details: Dict[str, Any]) -> Optional[float]:
    for key in ("absolute_error", "absolute_residual", "simulation_rmse"):
        if key in details and details[key] is not None:
            try:
                return float(details[key])
            except (TypeError, ValueError):
                return None
    return None


def _extract_rel_error(details: Dict[str, Any]) -> Optional[float]:
    for key in ("relative_error", "relative_rmse"):
        if key in details and details[key] is not None:
            try:
                return float(details[key])
            except (TypeError, ValueError):
                return None
    return None


def compute_metrics(trajectories: List[Dict[str, Any]], max_attempts_tracked: int = 5) -> Dict[str, Any]:
    total = len(trajectories)
    if total == 0:
        return {"total_problems": 0}

    first_attempt_successes = 0
    final_successes = 0
    attempts_used_list = []
    abs_errors = []
    rel_errors = []
    timeout_count = 0
    malformed_count = 0
    duplicate_count = 0
    total_gen_runtime = 0.0
    total_ver_runtime = 0.0
    success_after_attempt = {k: 0 for k in range(1, max_attempts_tracked + 1)}

    per_family: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"total": 0, "final_successes": 0, "abs_errors": []})
    per_generator: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"total": 0, "final_successes": 0})

    for traj in trajectories:
        attempts = traj.get("attempts", [])
        attempts_used = traj.get("attempts_used", len(attempts))
        attempts_used_list.append(attempts_used)

        is_final_success = traj.get("final_status") == "VERIFIED"
        if is_final_success:
            final_successes += 1

        if attempts and attempts[0]["verification"].get("verified"):
            first_attempt_successes += 1

        # success_after_attempt_N: did the trajectory reach VERIFIED status
        # by attempt N (inclusive)?
        verified_attempt_index = None
        for a in attempts:
            if a["verification"].get("verified"):
                verified_attempt_index = a["attempt"]
                break
        if verified_attempt_index is not None:
            for n in range(verified_attempt_index, max_attempts_tracked + 1):
                success_after_attempt[n] += 1

        for a in attempts:
            cand = a.get("candidate", {})
            ver = a.get("verification", {})
            details = ver.get("details", {}) or {}

            if ver.get("status") == "TIMEOUT":
                timeout_count += 1
            if cand.get("malformed"):
                malformed_count += 1
            if a.get("duplicate"):
                duplicate_count += 1

            if cand.get("runtime_seconds"):
                total_gen_runtime += cand["runtime_seconds"]
            if ver.get("runtime_seconds"):
                total_ver_runtime += ver["runtime_seconds"]

        final_attempt = attempts[-1] if attempts else None
        if final_attempt is not None:
            details = final_attempt["verification"].get("details", {}) or {}
            abs_err = _extract_abs_error(details)
            rel_err = _extract_rel_error(details)
            if abs_err is not None:
                abs_errors.append(abs_err)
            if rel_err is not None:
                rel_errors.append(rel_err)

            family = traj.get("family", "unknown")
            per_family[family]["total"] += 1
            if is_final_success:
                per_family[family]["final_successes"] += 1
            if abs_err is not None:
                per_family[family]["abs_errors"].append(abs_err)

        generator = traj.get("generator", "unknown")
        per_generator[generator]["total"] += 1
        if is_final_success:
            per_generator[generator]["final_successes"] += 1

    def _safe_mean(xs):
        return float(statistics.mean(xs)) if xs else None

    def _safe_median(xs):
        return float(statistics.median(xs)) if xs else None

    family_summary = {}
    for family, d in per_family.items():
        family_summary[family] = {
            "total": d["total"],
            "final_successes": d["final_successes"],
            "final_success_rate": d["final_successes"] / d["total"] if d["total"] else None,
            "mean_absolute_error": _safe_mean(d["abs_errors"]),
        }

    generator_summary = {}
    for generator, d in per_generator.items():
        generator_summary[generator] = {
            "total": d["total"],
            "final_successes": d["final_successes"],
            "final_success_rate": d["final_successes"] / d["total"] if d["total"] else None,
        }

    return {
        "total_problems": total,
        "first_attempt_successes": first_attempt_successes,
        "final_successes": final_successes,
        "first_attempt_success_rate": first_attempt_successes / total,
        "final_success_rate": final_successes / total,
        "average_attempts": _safe_mean(attempts_used_list),
        "median_attempts": _safe_median(attempts_used_list),
        "mean_absolute_error": _safe_mean(abs_errors),
        "mean_relative_error": _safe_mean(rel_errors),
        "timeout_count": timeout_count,
        "malformed_candidate_count": malformed_count,
        "duplicate_count": duplicate_count,
        "total_generation_runtime_seconds": total_gen_runtime,
        "total_verification_runtime_seconds": total_ver_runtime,
        "success_after_attempt": {f"success_after_attempt_{k}": v for k, v in success_after_attempt.items()},
        "by_family": family_summary,
        "by_generator": generator_summary,
    }
