"""
Verifier for parameter-estimation problems.

Independence strategy: given candidate parameters, independently run the
SAME forward model used to generate the dataset (this is the problem
definition, not the candidate's method) and compare the simulated curve
to the observed data via RMSE. The PASS/FAIL decision is based entirely
on this simulation-vs-observation comparison.

The dataset also retains true hidden parameters (see
problems/parameter_estimation.py). We additionally report
``parameter_error`` as a diagnostic (it is not used to decide PASS/FAIL,
since a general verifier cannot assume ground-truth parameters are always
available -- only V1's synthetic datasets happen to retain them).
"""

from __future__ import annotations

import math
import time

import numpy as np

from mathn.core.models import Candidate, Problem, VerificationResult
from mathn.problems.parameter_estimation import forward_model
from mathn.verifiers.base import NumericalVerifier, STATUS


class ParameterVerifier(NumericalVerifier):
    def __init__(self, rmse_threshold: float = 0.05):
        # NOTE: rmse_threshold is interpreted as a *relative* RMSE
        # (fraction of the mean |observation|), so it is meaningful across
        # subtypes with very different observation scales.
        self.rmse_threshold = rmse_threshold

    def verify(self, problem: Problem, candidate: Candidate) -> VerificationResult:
        start = time.time()

        if candidate.malformed or "parameters" not in candidate.result:
            return VerificationResult(
                verified=False,
                status=STATUS.MALFORMED_CANDIDATE,
                details={"reason": "candidate missing required field 'parameters'"},
                runtime_seconds=time.time() - start,
            )

        params = candidate.result["parameters"]
        expected_names = problem.public["parameter_names"]
        try:
            params = {k: float(params[k]) for k in expected_names}
        except (KeyError, TypeError, ValueError):
            return VerificationResult(
                verified=False,
                status=STATUS.MALFORMED_CANDIDATE,
                details={"reason": f"parameters must include finite values for {expected_names}", "got": params},
                runtime_seconds=time.time() - start,
            )

        if not all(math.isfinite(v) for v in params.values()):
            return VerificationResult(
                verified=False,
                status=STATUS.NUMERICAL_MISMATCH,
                details={"reason": "non-finite parameter value", "parameters": params},
                runtime_seconds=time.time() - start,
            )

        t = np.array(problem.public["t_observations"])
        y_obs = np.array(problem.public["y_observations"])
        fixed = problem.public.get("fixed", {})

        try:
            y_sim = forward_model(problem.subtype, params, t, fixed)
        except Exception as exc:
            return VerificationResult(
                verified=False,
                status=STATUS.SOLVER_FAILURE,
                details={"reason": f"independent forward simulation raised: {exc}", "parameters": params},
                runtime_seconds=time.time() - start,
            )

        if not np.all(np.isfinite(y_sim)):
            return VerificationResult(
                verified=False,
                status=STATUS.NUMERICAL_MISMATCH,
                details={"reason": "forward simulation produced non-finite values", "parameters": params},
                runtime_seconds=time.time() - start,
            )

        rmse = float(np.sqrt(np.mean((y_sim - y_obs) ** 2)))
        scale = float(np.mean(np.abs(y_obs))) + 1e-300
        relative_rmse = rmse / scale
        verified = relative_rmse <= self.rmse_threshold

        true_params = problem.protected.get("true_parameters")
        parameter_error = None
        if true_params:
            parameter_error = {k: abs(params[k] - true_params[k]) for k in expected_names if k in true_params}

        return VerificationResult(
            verified=verified,
            status=STATUS.VERIFIED if verified else STATUS.NUMERICAL_MISMATCH,
            details={
                "parameters": params,
                "simulation_rmse": rmse,
                "relative_rmse": relative_rmse,
                "threshold": self.rmse_threshold,
                "parameter_error": parameter_error,  # diagnostic only, not decisive
            },
            runtime_seconds=time.time() - start,
            message=None if verified else "Simulated curve does not match observations within tolerance.",
        )
