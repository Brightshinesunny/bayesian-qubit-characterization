# Superconducting Qubit Data Analysis — Master Index

[한국어](README.md) | English (this document) | [日本語](README_ja.md)

A project-by-project index of all work carried out between 8/21 and
8/29 (130+ files). **If you're not sure where to start, start here.**

---

## 0. Read First — 4 Summary Documents

| File | Content |
|---|---|
| `09_summaries/SUMMARY_fano_avoided_tls_physics_and_stats.py` | Physics + statistics mapping across the three physical systems (Fano / avoided-crossing / TLS) |
| `09_summaries/SUMMARY_statistical_toolkit_playbook.py` | Tool-by-tool notes on Gaussian/Robust/MCMC/Fisher matrix, and the 8-step standard workflow |
| `09_summaries/SUMMARY_catalog_to_raw_mapping_journey.m` | The full TLS catalog↔raw mapping journey (v1–v4, final verdict on columns 4 and 17) |
| `09_summaries/SUMMARY_robots_performance_and_full_journey.py` | Real-world performance review of the 8 validation robots + full-journey synthesis |

---

## 01_zenodo_baseline — Foundational Pipeline

Goal: establish a Bayesian pipeline for avoided-crossing (fr, g,
kappa) using mock data, then apply it to real Zenodo data (Sett et
al., PRX Quantum 5, 010327 (2024) — see `DATA_SOURCES.md`). **The
folder contains 4 chronological stages: v0 (first single scripts) →
core (modularized) → mock_data_validation (validation) →
real_data_application (applied to real data). See
`01_zenodo_baseline/README.md` for details.**

**Conclusion**: fixed a `1j`-convention bug (using `2j` underestimates
the coupling strength by a factor of 2). Established the
Gaussian/Robust comparison pipeline → became the backbone for every
later project.

---

## 02_fano_resonator — Fano Resonators, Circle Fit vs. Bayesian/Fisher

Goal: estimate Qi/Qc/Ql for 9 resonators (overcoupled + undercoupled).

**Key files** (the remaining `fano_*.py` files are mostly intermediate trial-and-error):
- `fano_final_pipeline.py`, `fano_final_estimation_with_corner.py`: the final, confirmed pipeline
- `fano_undercoupled_summary.py`: coupling-ratio (Ql/Qc) vs. credible-interval-direction analysis across 8 undercoupled resonators
- `fano_noise_diagnosis.py`, `fano_correlated_noise*.py`: residual-autocorrelation diagnostics (AR(1)/compound-symmetry attempts, ultimately abandoned)

**Final conclusion**: circle fit serves only as an initial-value safety net; Qi
should be estimated via Bayesian/Fisher methods. 5% outlier trimming
+ Gaussian/Robust cross-validation is the most stable approach. For
undercoupled resonators, the true determinant of the credible-interval
pattern is the Ql/Qc ratio (coupling strength), not simply whether the
resonator is "undercoupled or not."

---

## 03_avoided_crossing_tls — Avoided-Crossing Revisited + TLS Extension

Goal: apply the day-1 avoided-crossing model to 2D TLS swap-spectroscopy data.

```
tls_avcross_models.py       # V0-recentered 2D avoided-crossing model
tls_avcross_likelihood.py   # generic real-valued 2D likelihood functions
tls_avcross_fit_seg12.py    # seg12 fitting in practice (multi-start + MCMC)
tls_avcross_visual_check.py # visual comparison of 3 candidates (pixel-tracking/STEP2/STEP3)
tls_avcross_pipeline.py     # reusable function library + alternative horizontal-lines model
```
**Final conclusion**: visual inspection showed seg12 is not a diagonal
avoided-crossing trace at all, but a horizontal multi-band structure —
the original hypothesis itself was wrong. This resolves the
100×-discrepancy between pixel-tracking (109 MHz/V) and the 2D-model
MCMC result (0.6–1.7 MHz/V): the assumption, not the method, was at fault.

---

## 04_tls_data_structure — Decoding the TLS Data Structure

From first exploration of the HDF5 (v7.3) structure to full decoding
of `tabledata2` (13 rows).
- `tls_decode_tabledata2.py`: reverse-engineering row meanings via scale analysis
- `tls_catalog_final_estimation.py`, `tls_noise_removal_bayesian_full.py`: Gaussian/Robust + Fisher-matrix analysis of the 20-entry TLS catalog
- `tls_position_quick_attempt.py`: a simplified weighted-centroid position estimate, without simulation

**Confirmed**: rows 0, 2, 9, 11 = coupling strength (4 electrodes); row
4 = TLS frequency. Rows 1, 3, 10, 12 are **not** voltage (confirmed in
a later section).

---

## 05_tls_noise_diagnosis_dip_tracking — Systematic-Error Diagnosis + Dip Tracking

Five successive rounds of fixes to the pixel dip-tracking algorithm
(indexing → nonlinear mapping → crossing point → stall detection).
`tls_seg12_v5_precrossing_only.py` is the final, successful version
(0.1% agreement between Gaussian and Robust estimates). Four
systematic artifacts confirmed: 3 flux-based ones (0.045, −0.030,
−0.065) plus one at 5.168 GHz.

---

## 06_tls_multi_file_integration — Combining Three TLS Files

Segments 1–200 / 200–400 / 400–640 (59 TLS catalog entries in
total). `tls_heterogeneity_check.py`: tested for heterogeneity
between files → rejected (the files are homogeneous; the earlier
apparent difference was ordinary statistical fluctuation).

---

## 07_catalog_to_raw_mapping — Catalog↔Raw Reverse Mapping

**Important: the process of narrowing down the failure mode from
v1 through v4 is itself the key learning here.**
- v1: assumed rows 1/3/10/12 = voltage → produced physically impossible values (e.g. −151 V) → assumption abandoned
- v2: matched by frequency + dominant electrode → the filter turned out to be essentially meaningless (whole-map quality score got contaminated, producing a false top candidate)
- v3: separated local vs. global quality scores + introduced an A/B/C case classification → correctly filtered out the false top candidate
- v4: checked all 4 electrodes systematically → column 4 had zero A-cases across all 4 electrodes and all 639 segments
- `tls_multi_column_scan.py`: scanned all 19 catalog entries → only 15.8% matched anything in the raw data; column 17 dominated
- `tls_column17_visual_check.py`: continuity tracking showed column 17 is also essentially horizontal (0.36 MHz/V, unrelated to the catalog's 452 MHz/V)

**Final conclusion** (see Part 6 of `SUMMARY_catalog_to_raw_mapping_journey.m`):
most catalog entries (84%) fall within the original authors'
multi-segment-stitching algorithm — something our single-frame search
cannot reproduce. The 5.096 GHz band at column 17 is more likely a
systematic structure (though not conclusively confirmed).

---

## 08_validation_pipeline — Validation Pipeline

Eight validation tools (prior sanity check / parameter recovery / multi-start
consistency / parameter correlation / residual autocorrelation /
Gaussian-vs-Robust cross-check / quality score / full diagnostic
suite). Real-world performance is reviewed in Part 1 of
`09_summaries/SUMMARY_robots_performance_and_full_journey.py` — 6
worked correctly on first design; 2 (the Gaussian-vs-Robust check and
the quality-score robot) needed redesign once actually deployed
(notably, the quality-score tool's failure to distinguish "global"
from "local" was the root cause of the catalog-mapping failure,
fixed in v3).

---

## 10_mathematica_fisher_pipeline — A Separate Mathematica-Based Fisher Pipeline

While 01–09 are entirely Python, this folder documents the same
GW-derived Bayesian/Fisher methodology **independently implemented in
Mathematica (Wolfram Language)**. It applies Fisher-matrix analysis to
a damped-oscillation signal model (T2 decay + Rabi oscillation),
expanding through 7 stages: single qubit → multi-qubit → PSD / 2-qubit
cross-coupling → prior-informed 4D MAP → inclusion of systematic bias.
See `10_mathematica_fisher_pipeline/README.md` for details.

---

## 11_results_figures — Real-Data Re-run Result Figures (added later)

Three figures reproducing the core findings of `02_fano_resonator` and
`05_tls_noise_diagnosis_dip_tracking` using the actual original data
(9 real Fano resonator npz files, TLS raw mat files): |Qi| divergence
near critical coupling across all 9 resonators, a circle-fit example
of the most extreme divergence case, and systematic-error bands in a
TLS raw segment. Includes KO/JA README.

## 12_bugfixes_pending — Reviewed Bug Fixes Awaiting Re-run

Includes a fix for the Fano uncertainty calculation bug in
`02_fano_resonator` (uncertainty was silently reported as zero when
the underlying assumption was violated), a fix for the Fano-correction
model bug in `01`/`03` (signal was shrunk up to 5x), 11 robustness
improvements to the avoided-crossing stress-test pipeline, and a
comprehensive injection-test report (injection_test_JP.md) documenting
the results. Not yet merged into the main code.

## Quick Navigation When There Are Too Many Files

1. **For the full physics/statistics picture**: read the 4 SUMMARY documents in order.
2. **Reusable code**: `08_validation_pipeline/tls_avcross_pipeline.py` (avoided-crossing),
   `08_validation_pipeline/validation_pipeline.py` (validation),
   `02_fano_resonator/fano_final_pipeline.py` (Fano).
3. **Curious about "why it ended up this way"?** Follow each section's
   v1→v4-style sequence — the trial-and-error logic is preserved as-is.
