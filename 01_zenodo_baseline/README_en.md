# 01_zenodo_baseline

[한국어](README.md) | [日本語](README_ja.md) | English (this document)

The very first project in this repository: a Bayesian/Fisher-based
analysis pipeline for superconducting-qubit avoided-crossing (S21)
spectroscopy. It starts with mock-data validation of the pipeline and
ends with an application to real Zenodo data ("Emergent Macroscopic
Bistability Induced by a Single Superconducting Qubit," DOI
10.5281/zenodo.10518320 — see `../DATA_SOURCES.md` for details). The
folder captures this 4-stage evolution in chronological order.

## Folder Structure (4 stages, in order)

```
01_zenodo_baseline/
├── v0_first_prototype/         # Stage 1: single scripts, before modularization
│   ├── anticrossing_full_flux_sweep.py   # first batch MCMC fitting
│   ├── generate_noisy_mock_data.py       # mock data with 1/f noise + drift + outliers
│   ├── threshold_methods_comparison.py   # 3 methods for stability-threshold detection
│   └── delta_vs_g_uncertainty.py         # |Δ| vs g-uncertainty diagnostic
│
├── core/                        # Stage 2: reusable modular components
│   ├── models.py                # forward model (avoided-crossing S21)
│   ├── likelihood.py            # prior/likelihood (gaussian/robust)
│   ├── mcmc_pipeline.py         # emcee runner (single/batch, warm-start)
│   ├── diagnostics.py           # noise estimation, threshold detection
│   ├── bayesian_toolkit.py      # credible interval, R-hat, AIC/BIC, etc.
│   └── fisher_matrix.py         # Fisher-matrix based uncertainty (MCMC alternative)
│
├── mock_data_validation/        # Stage 3: pipeline validation with mock data
│   ├── mock_data.py             # clean-signal + noise-injection generator
│   ├── example_run.py           # example combining all 6 core modules
│   ├── example_run_bay.py       # + deeper bayesian_toolkit analysis
│   └── compare_gaussian_vs_robust.py  # accuracy comparison
│
└── real_data_application/       # Stage 4: applied to real Zenodo data
    ├── zenodo_loader.py, zenodo_load_real_data.py
    ├── zenodo_models.py, zenodo_likelihood.py, zenodo_mcmc_pipeline.py
    ├── zenodo_bayesian_toolkit.py, zenodo_diagnostics.py, zenodo_fisher_matrix.py
    ├── zenodo_fit_and_compare.py, zenodo_joint_fit.py
    ├── zenodo_multi_file_check.py, zenodo_verify_parser_equivalence.py
```

## Design Principles

**Parts that change (per physical system)**
- `core/models.py`: add one function per new physical system
- `mock_data_validation/mock_data.py`'s clean-signal generation logic
- `core/likelihood.py`'s `_theta_to_kwargs` (mapping changes when parameter set changes)

**Parts that stay mostly fixed (the skeleton)**
- `core/mcmc_pipeline.py`: sampler execution, warm-start, convergence diagnostics
- `core/diagnostics.py`: noise estimation, threshold-detection methodology
- `core/likelihood.py`'s factory-function structure (`make_uniform_log_prior`, `make_log_probability`)

## Extension Scenarios

| Situation | Where to change |
|---|---|
| Different physical system (e.g. 3-qubit) | add a function to `core/models.py`, replace STEP 1–4 in `example_run.py` |
| Outlier-robust analysis | set `aa_likelihood_type = 'robust'` in `example_run.py` |
| Different noise characteristics | tune `aa_flicker_level`, `aa_drift_amplitude`, `aa_outlier_probability` in `example_run.py` |
| Hierarchical model (shared parameters) | extend `core/mcmc_pipeline.py` with a new function that fits multiple slices jointly |
| A different sampler (nested sampling) | add a parallel function such as `run_single_nested()` to `core/mcmc_pipeline.py` |

## Usage

```bash
cd 01_zenodo_baseline
# so that the core modules are importable, run from inside
# mock_data_validation or real_data_application, or add core
# to PYTHONPATH
PYTHONPATH=core python mock_data_validation/example_run.py
```

Adjust the `aa_` variables at the top of `example_run.py` to toggle
noise types on/off or switch the likelihood (gaussian/robust).

## Validation History

- `core/models.py`'s `s21_anticrossing_model`: recovered fr, g, and
  kappa within their error bars against mock-data ground truth (the
  `1j` convention was confirmed — using `2j` instead causes the
  coupling strength to be underestimated by a factor of 2)
- Comparison of the three threshold-detection methods in
  `core/diagnostics.py`: the crossing method and the inflection
  method — despite being conceptually different — converge on the
  same threshold (|Δ| ≈ 0.29 GHz), while the discrete-label method
  (method A) was confirmed to be sensitive to outliers

## Data Source

See `../DATA_SOURCES.md` at the repository root for the source of the
real data used in `real_data_application/`.
