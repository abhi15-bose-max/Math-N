# MATH-N: Verifier-Driven Numerical Mathematics

MATH-N is a minimal, extensible research framework for studying one
architectural idea:

> **Generation and verification are separate.** A candidate solution to a
> mathematical problem is never trusted to grade its own correctness. An
> independent numerical/symbolic computation verifies it instead.

```
MATHEMATICAL PROBLEM
        |
CANDIDATE GENERATOR
        |
  NUMERICAL/SYMBOLIC CANDIDATE
        |
 INDEPENDENT VERIFIER
        |
   PASS / FAIL
     /      \
  PASS      FAIL
    |          |
 finish    feedback -> next candidate -> retry -> trajectory -> evaluation
```

This is **V1**: classical numerical and symbolic methods (NumPy / SciPy /
SymPy) generating candidates, independently checked by SciPy-based
verifiers, with a generic retry loop, full trajectory logging, and
aggregate evaluation. Nothing here is specific to AI. The
`CandidateGenerator` interface is deliberately general enough to later
host an LLM, an SLM, an MCMC sampler, or a hybrid method (see
[Roadmap](#roadmap)) without touching the core architecture.

## Table of contents

- [Why generation != verification](#why-generation--verification)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Problem families](#problem-families)
- [Candidate generators](#candidate-generators)
- [Verifiers and independence](#verifiers-and-independence)
- [Retry loop, feedback, duplicates, timeouts](#retry-loop-feedback-duplicates-timeouts)
- [Trajectories and metrics](#trajectories-and-metrics)
- [V1 results](#v1-results)
- [Known limitations](#known-limitations)
- [Scientific positioning](#scientific-positioning)
- [Roadmap](#roadmap)
- [Testing](#testing)

## Why generation != verification

A candidate generator's job is to *propose* an answer. It is not asked
whether its own answer is correct -- that determination is made by a
separate, independently-configured verifier that recomputes a reference
quantity via a different method wherever practical (see
[Verifiers and independence](#verifiers-and-independence) for exactly how,
and where perfect independence isn't possible).

This separation is what makes MATH-N useful as a research instrument
rather than a demo: every "PASS" in this repository means *something
else, computed a different way, agrees with the candidate within an
explicit tolerance* -- not merely "the algorithm that produced this also
believes it".

## Repository layout

```
math-n/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml            # every numeric knob lives here
├── datasets/v1/                 # generated synthetic problems (JSON, one per problem)
├── src/mathn/
│   ├── core/
│   │   ├── interfaces.py       # CandidateGenerator, NumericalVerifier (AI-agnostic)
│   │   ├── models.py           # Problem, Candidate, VerificationResult, Attempt, Trajectory
│   │   ├── runner.py           # RetryController: generate -> verify -> retry loop
│   │   └── timeout.py          # SIGALRM-based timeout enforcement
│   ├── generators/
│   │   ├── base.py             # registry
│   │   ├── scipy_generator.py  # classical numerical methods
│   │   └── sympy_generator.py  # symbolic / closed-form methods
│   ├── problems/
│   │   ├── base.py             # ProblemGenerator ABC + registry
│   │   ├── ode.py               # family A: initial-value problems
│   │   ├── roots.py             # family B: f(x) = 0
│   │   ├── parameter_estimation.py  # family C
│   │   └── symbolic.py          # family D: symbolic-to-numerical
│   ├── verifiers/
│   │   ├── base.py             # NumericalVerifier re-export + status codes
│   │   ├── ode_verifier.py
│   │   ├── root_verifier.py
│   │   ├── parameter_verifier.py
│   │   └── symbolic_verifier.py
│   ├── logging/
│   │   └── trajectory.py       # save/load full trajectories as JSON
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── failure_analysis.py
│   ├── dataset.py               # dataset generation/persistence across all families
│   └── cli.py                   # generate-dataset / run / evaluate / inspect
├── experiments/
│   └── run_v1.py                # one-command end-to-end benchmark driver
├── trajectories/                 # generator/*.json (gitignored, regenerated)
├── results/                       # v1_report.json (gitignored, regenerated)
├── notebooks/
│   └── MATH_N_V1_demo.ipynb     # Colab-ready walkthrough
└── tests/
    ├── test_problems.py
    ├── test_generators.py
    ├── test_verifiers.py
    ├── test_runner.py
    ├── test_trajectory.py
    └── test_metrics.py
```

## Installation

```bash
git clone <this-repo>
cd math-n
pip install -e .
```

Requires Python >= 3.10. Dependencies: `numpy`, `scipy`, `sympy`,
`pyyaml` (see `requirements.txt` / `pyproject.toml`). No GPU is required
anywhere in V1.

## Quickstart

```bash
# 1. Generate the synthetic V1 dataset (40 problems: 10 per family)
python -m mathn.cli generate-dataset --dataset datasets/v1

# 2. Run a candidate generator over the dataset
python -m mathn.cli run --dataset datasets/v1 --generator scipy --max-attempts 5
python -m mathn.cli run --dataset datasets/v1 --generator sympy --max-attempts 5

# 3. Evaluate: compute metrics + failure analysis over saved trajectories
python -m mathn.cli evaluate --trajectories trajectories/scipy --out results/scipy_report.json

# 4. Inspect a single trajectory
python -m mathn.cli inspect --trajectory trajectories/scipy/<problem_id>__scipy.json
```

Or run the whole pipeline (dataset -> both generators -> metrics ->
failure analysis -> `results/v1_report.json`) in one command:

```bash
python experiments/run_v1.py
```

## Problem families

All four families are synthetic, seeded, and reproducible (same seed +
index always produces the same problem). Each `Problem` exposes a
`public` view (everything the candidate generator may see) and a
`protected` view (ground truth / hidden parameters, used only by the
dataset and by evaluation diagnostics -- **never** passed to a
generator).

| Family | Module | Subtypes | Query |
|---|---|---|---|
| A: ODE / IVP | `problems/ode.py` | exponential decay/growth, logistic growth, linear ODE, coupled linear system | compute `y(t_query)` |
| B: Root-finding | `problems/roots.py` | polynomial, transcendental (`cos x - x`), nonlinear (`x e^x = c`) | find `x` with `f(x) = 0` in a domain |
| C: Parameter estimation | `problems/parameter_estimation.py` | exponential-decay fit, logistic-growth fit | estimate hidden parameters from noisy observations |
| D: Symbolic-to-numerical | `problems/symbolic.py` | derivative at a point, definite integral | compute a numeric quantity derived from a symbolic expression |

## Candidate generators

Two generators ship with V1, chosen specifically so their **methods**
differ from each other and from their verifiers (see next section):

- **`ScipyGenerator`** (`generators/scipy_generator.py`) -- classical
  iterative numerical methods: `solve_ivp` (RK45) for ODEs, `brentq` for
  roots, `least_squares` multi-start for parameter estimation, and manual
  forward-difference / composite Simpson's rule for the symbolic family.
  Its precision knob (solver tolerance, step size, panel count) starts
  loose and tightens across retry attempts.
- **`SympyGenerator`** (`generators/sympy_generator.py`) -- exact /
  closed-form / few-point analytic methods: `dsolve` for ODEs (with a
  Bernoulli substitution for the logistic case, since raw `dsolve` returns
  an expensive implicit form there), exact `solve` / the Lambert-W special
  function / `nsolve` for roots, log-linear regression / two-point
  `nsolve` for parameter estimation, and exact `diff` / `integrate` for
  the symbolic family. Being (mostly) exact, it rarely benefits from
  retrying -- see [V1 results](#v1-results) for where that hurts it.

Both implement the same `CandidateGenerator` interface
(`core/interfaces.py`): `generate(problem, context) -> Candidate`, where
`problem` is always the generator-facing view (`Problem.for_generator()`)
and `context` carries the attempt number and prior feedback. Nothing in
the interface assumes AI is involved, and a generator is free to ignore
`context` entirely.

## Verifiers and independence

Section 33 of the design brief singles this out as one of the most
important requirements, so here is exactly what each verifier does and
does not trust:

| Family | Verifier | Independent recomputation | What it is *not* trusted to say |
|---|---|---|---|
| ODE | `ODEVerifier` | Re-integrates with `solve_ivp(..., method="Radau", rtol=1e-12, atol=1e-14)` regardless of what produced the candidate | Never reads a candidate's own solver metadata |
| Roots | `RootVerifier` | Rebuilds `f(x)` from the problem's stored coefficients (a plain Python callable, not the generator's expression object) and checks `\|f(x)\| <= tol` plus domain membership | Never re-runs the generator's search procedure |
| Parameter estimation | `ParameterVerifier` | Independently forward-simulates the candidate's parameters through the same physical model and compares to the observed data via relative RMSE | Does **not** decide PASS/FAIL by comparing to the hidden true parameters (that comparison is logged separately as a `parameter_error` diagnostic only, since a general verifier can't assume ground truth is always available) |
| Symbolic | `SymbolicVerifier` | 5-point central finite difference for derivatives; adaptive quadrature (`scipy.integrate.quad`) for integrals -- neither touches SymPy | Never asks SymPy to confirm its own symbolic result |

**Where perfect independence is not possible:** for the root-finding and
symbolic families, the function `f(x)` itself is shared between generator
and verifier -- that is the *problem statement*, not the candidate's
method, so sharing it is not circularity. For parameter estimation, the
forward model is likewise shared (it is what "observations" means). What
is always independent is the *method used to search for or derive the
answer*.

## Retry loop, feedback, duplicates, timeouts

`core/runner.py::RetryController` drives:

```
generate -> verify -> PASS -> done
                 \-> FAIL -> feedback appended -> retry (up to max_attempts)
```

- **Duplicate detection**: candidates are hashed by their result dict. If
  a generator that does not declare `supports_feedback` repeats its exact
  previous (wrong) answer, the loop stops early rather than burning the
  full attempt budget -- it cannot possibly converge by retrying
  identically. Generators that *do* use feedback are allowed to repeat a
  value (e.g. while exploring) without early termination.
- **Timeouts**: enforced per generation/verification call via
  `SIGALRM` (`core/timeout.py`). A timeout is recorded as `TIMEOUT`, not
  as a mathematical failure -- it means the computation didn't finish
  within budget, not that it was wrong.
- **Malformed candidates**: a generator that raises `MalformedCandidateError`
  (or crashes) produces a candidate flagged `malformed=True` with status
  `MALFORMED_CANDIDATE`, distinct from a wrong-but-well-formed answer.
- **`max_attempts`** always bounds the loop (default 5), regardless of
  duplicates, feedback support, or malformed candidates.

## Trajectories and metrics

Every run produces a full `Trajectory` (never just the final answer):
every attempt's candidate, its verification result, whether it was a
duplicate, and timing -- saved as JSON under `trajectories/<generator>/`.

`evaluation/metrics.py` computes (see `results/v1_report.json` for the
live numbers): total/first-attempt/final success counts and rates,
average/median attempts, mean absolute/relative error, timeout/malformed/
duplicate counts, generation/verification runtime totals,
`success_after_attempt_1..5`, and breakdowns by problem family and by
generator.

`evaluation/failure_analysis.py` groups failures into categories
(`NUMERICAL_MISMATCH`, `RESIDUAL_TOO_LARGE`, `INVALID_DOMAIN`,
`SOLVER_FAILURE`, `TIMEOUT`, `MALFORMED_CANDIDATE`, ...), and reports
unsolved problems, which problems needed retries, which converged after
retrying, and the largest remaining errors.

## V1 results

Benchmark: 40 problems (10 per family), `max_attempts=5`, run on this
machine (Python 3.12.3, NumPy 2.4.4, SciPy 1.17.1, SymPy 1.14.0). Full
machine-readable output: `results/v1_report.json`. Reproduce with
`python experiments/run_v1.py`.

| Generator | Final success rate | First-attempt success rate | Avg. attempts | Mean \|error\| | Malformed | Wall time (40 problems) |
|---|---|---|---|---|---|---|
| `scipy` | **100.0%** (40/40) | 47.5% (19/40) | 2.25 | 3.9e-2 | 0 | 1.67s |
| `sympy` | 90.0% (36/40) | 90.0% (36/40) | 1.18 | 3.7e-1 | 5 attempts (1 problem) | 3.49s |

Per-family final success rate:

| Family | scipy | sympy |
|---|---|---|
| ODE | 10/10 | 10/10 |
| Roots | 10/10 | 10/10 |
| Parameter estimation | 10/10 | 6/10 |
| Symbolic | 10/10 | 10/10 |

**Reading these numbers correctly:** the *mean absolute error* column
mixes different units across families (ODE state values, root residuals,
RMSE) and is dominated by whichever family has the largest raw scale --
treat it as a rough index, not a single meaningful number; the
per-family breakdown in `results/v1_report.json` is the trustworthy view.

**What the numbers show:**

- **SciPy needs retries, but always eventually converges.** Its
  first-attempt success rate is only 47.5% by design -- it starts with
  loose solver tolerances (`rtol=1e-2` for ODEs, `xtol=1e-1` for roots,
  4-panel Simpson's rule for integrals) and tightens on each retry. Every
  one of the 40 problems reaches `VERIFIED` by attempt 4
  (`success_after_attempt_3 = 30/40`, `success_after_attempt_4 = 40/40`).
  This is the retry loop doing real work, not a formality.
- **SymPy is usually exact on the first try, but fails outright rather
  than partially on its weak spot.** Closed-form ODE solutions, exact
  polynomial roots, the Lambert-W closed form, and exact
  differentiation/integration all verify on attempt 1 essentially every
  time (ODE, roots, and symbolic all 10/10). Its failures are
  concentrated entirely in `parameter_estimation` (6/10 vs SciPy's
  10/10): the two-point `nsolve` approach used for logistic growth is
  sensitive to noise at the two sampled points -- most failures converged
  to a fit outside tolerance (`NUMERICAL_MISMATCH`), and one didn't
  converge at all (`MALFORMED_CANDIDATE`, an `nsolve` convergence
  failure). This is a genuine, reproducible finding: a few-point analytic
  estimator is less robust to observation noise than SciPy's full-dataset
  nonlinear least squares, illustrating exactly the kind of
  generator-comparison signal MATH-N is meant to surface (see
  [Primary research question](#why-generation--verification)).
- **Neither generator ever produced a false PASS.** All 76 successful
  trajectories across both generators were independently verified via a
  different computational path than the one that produced the candidate
  (see [Verifiers and independence](#verifiers-and-independence)).

## Known limitations

- **Malformed-candidate loops are not deduplicated.** Duplicate detection
  only applies to well-formed candidates; a deterministic generator
  (like `SympyGenerator`, which does not use feedback) that fails with
  the *same* error every time will still burn its full `max_attempts`
  budget re-raising it, as seen in the `logistic_growth_fit` failure
  above (5 identical `nsolve` failures). A V2 improvement would extend
  duplicate detection to cover repeated identical error messages.
- **Identifiability is assumed, not verified.** Parameter-estimation
  problems are constructed to be identifiable from the sampled
  observation grid (documented, not proven, per `problems/parameter_estimation.py`);
  MATH-N does not yet distinguish identifiable / weakly identifiable /
  non-identifiable problems (see project brief section 26).
- **The ODE verifier's "independence" is a different solver + far tighter
  tolerance, not a fully disjoint algorithm family** in every case --
  when the candidate generator also used `solve_ivp` (as `ScipyGenerator`
  does), independence rests on the method (RK45 vs Radau) and tolerance
  gap rather than a completely different numerical paradigm. This is
  documented, not hidden: see [Verifiers and independence](#verifiers-and-independence).
  Where a genuinely different paradigm was available (symbolic ODE
  solutions, closed-form roots), we used it.
- **Mean-error metrics mix units across problem families**; use the
  per-family breakdown, not the top-level `mean_absolute_error`, for any
  real comparison.
- **V1 is small by design** (40 problems, 2 generators, 4 families) --
  broad claims should wait for a larger benchmark (see
  [Roadmap](#roadmap)).

## Scientific positioning

This framework evaluates candidate numerical solutions against
independently computed reference quantities, under an explicit numerical
model, tolerance, and solver configuration. It does **not** prove that a
numerical answer is universally, absolutely, mathematically correct --
correctness here is always relative to the stated model, method, and
tolerance. A `VERIFIED` result means "an independent computation agrees
with this candidate within the configured tolerance," not "this is
provably the unique true mathematical answer under all interpretations."

## Roadmap

The core architecture (`CandidateGenerator` / `NumericalVerifier` /
`RetryController`) is meant to stay fixed while candidate-generation
strategies get progressively more sophisticated:

- **V1** (this repository): classical numerical/symbolic methods (SciPy, SymPy).
- **V2**: additional numerical/probabilistic methods (e.g. Bayesian
  inference, MCMC/HMC/ABC) under the same verifiers.
- **V3**: an AI candidate generator (e.g. an LLM), implementing the exact
  same `CandidateGenerator` interface.
- **V4**: a specialized numerical SLM, trained on failure-driven,
  targeted datasets derived from V1-V3 trajectories (not automatically --
  training is a deliberate future decision based on observed, systematic
  failure modes, not a default pipeline step).
- **V5**: hybrid classical + AI candidate generation.

**Recommended next experiment:** extend `parameter_estimation.py` with a
non-identifiable or weakly-identifiable subtype (e.g. a model with a
near-degenerate parameter combination) and confirm the verifier's
simulation-RMSE criterion correctly rejects parameter estimates that fit
the observations well but recover the wrong underlying parameters --
directly testing the identifiability caveat from section 26 of the
design brief, and giving `failure_analysis.py` a genuine
`CONSTRAINT_VIOLATION`-style case to report on beyond what V1's
by-construction-identifiable problems can surface.

## Testing

The sandbox this repository was built in has no network access, so
`pytest` isn't installed; all tests are written against the Python
standard library's `unittest` and run with zero extra dependencies:

```bash
python -m unittest discover -s tests -v
```

If you do have `pytest` available (see `requirements.txt`), it will also
happily collect and run these same `unittest.TestCase` classes:

```bash
pytest tests/
```

60 tests currently cover: reproducible problem generation (including a
cross-process regression test), protected-field hiding, well-formed
candidate shape per family, malformed-candidate handling, all four
verifiers (correct/incorrect/boundary/domain cases), the retry
controller (first-attempt success, duplicate-triggered early stop,
feedback-driven convergence, `max_attempts` enforcement, timeout
handling, malformed-candidate retries, feedback-history plumbing),
trajectory JSON round-tripping (including NumPy-scalar safety), and
metrics/failure-analysis aggregation.
