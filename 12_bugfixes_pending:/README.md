# Bug Fixes Not Yet Merged (reviewed, awaiting re-run)

These four files fix real bugs found in the existing code. They have not replaced the main repository code yet, and earlier conclusions have not all been re-verified. This note records what changed, why, and what it affects, for the next time the project is reopened.

---

## 1. fano_models_v2.py → replaces the file in `02_fano_resonator/`

**FIX-1 (most important).** At the exact moment the Fano isolation assumption broke down, the uncertainty range (`Qi_min`/`Qi_max`) was being reported as zero. The code looked most confident when its assumption failed, which made this a dangerous bug. A `fano_assumption_violated` flag was added, and the function now returns NaN when the assumption is violated.

**New: `qi_range_vs_isolation()`.** Shows how the Qi range changes when the isolation assumption is set to 10/15/20/25 dB. Isolation is an assumption, not a measurement, so it is more honest to report this sensitivity table than to commit to a single range.

**New: `CircleFitError` exception.** Circle-fit failures (no root of the characteristic equation, or an unphysical radius) now raise a clear exception, so that one failed resonator does not kill a loop over many resonators.

**What it affects.** The Qi *point estimate* (`result['Qi']`) comes from a separate calculation step (STEP D) and is **not affected**. Only the **confidence interval / uncertainty range** (`Qi_min`, `Qi_max`) is affected. Any statement in `02_fano_resonator` about the interval being wide or narrow needs to be rechecked.

**fig1 and fig2 in `11_results_figures` are not affected**, because both use only the Qi point estimate.

---

## 2. models_v2.py → replaces `01_zenodo_baseline/core/models.py` and files in `03_avoided_crossing_tls/`

**FANO-1 (most important).** The old Fano-correction function shrank the signal unphysically by up to a factor of 5. A standard model fit to such data no longer measures "bias from model misspecification"; the signal cannot be matched by any parameters at all.

**FANO-2.** The definition of x was tied to the frequency grid (number of points). It is now normalized by the linewidth (kappa). The old version gave different results when `n_freq` changed, so it was not reproducible.

**FANO-3.** The multiplicative correction (which can give |S21| > 1, so is unphysical) was replaced by the complex-phase (complex Qc) method, the standard approach of Khalil et al. 2012 and Probst et al. 2015. Call it with `method='phase'`.

**What it affects.** Any mock data with Fano distortion generated or fitted in `01_zenodo_baseline` or `03_avoided_crossing_tls` came from the buggy version and should be rechecked. The three figures in `11_results_figures` do not use this file.

---

## 3. stress_test_documented.py → replaces the file in `03_avoided_crossing_tls/`

A robustness test script for the avoided-crossing MCMC pipeline. It contains 11 changes relative to the original `stress_test.py` (prior upper bounds, walker initialization, dip-search algorithm, fixed MCMC seed, automatic avoided-crossing flux search, per-scenario individual sweeps, and others).

**New: three-stage signal separation.** The old version had two stages, "clean" and "noisy". Now `s21_ideal` (true values) → `s21_distorted` (distortion only, no noise) → `s21_noisy` (distortion + noise, the actual fit target) are kept separate. Plotting them as gray (ideal), green (distorted), and red (fit) shows in one figure what the distortion changed and how far the model followed.

**New: `sigma_ratio` tracking.** How many times the estimated noise exceeds the true noise is stored explicitly in the result dictionary. It is already 2.42x at baseline and inflates to 6.06x when all three distortions are on. Overestimated noise loosens the likelihood and statistically hides model misspecification, which makes this an important diagnostic.

**New: physics and statistics glossary sections.** Key concepts (avoided crossing, Hopfield coefficient, dispersive regime, autocorrelation time tau, MAP vs. median) are explained in the code to make it reusable.

**Key finding (ALL scenario).** Fano moves kappa by −7.3% (underestimate) and TLS by +8.1% (overestimate), but with all three on (ALL) kappa appears off by only +3.3%. That is not even the simple sum (+0.8%); the effects partially cancel nonlinearly. Looking only at ALL, without individual sweeps, would have underestimated the size of each effect by more than half.

---

## 4. injection_test.md — summary report of the results above

A report on the `stress_test_documented.py` results, from the physical background (avoided crossing, Hopfield coefficient, dispersive regime) through seven conclusions to follow-up directions.

**Seven conclusions in brief.** (1) The choice of operating point decides parameter identifiability: at the avoided crossing the error is within 1.2%, while in the dispersive regime g is degenerate and set by the prior. (2) Model misspecification concentrates in kappa: f_r and g stay near 1%, while kappa is off by more than 5x as much. (3) Effects can cancel. (4) The parameter that looks most accurate (f_r) is the least sensitive one. (5) The model's degrees of freedom decide where the distortion goes. (6) The noise estimate itself is contaminated by misspecification, in the direction that hides it. (7) Convergence and accuracy of the estimate are separate questions.

---

## Next steps (not done yet)

- [ ] Recheck any statement in `02_fano_resonator` that mentions Qi confidence intervals
- [ ] Re-run any mock-data results in `01`/`03` that used the Fano-correction model, with `models_v2.py`
- [ ] If nothing breaks, replace the old files with these fixed versions and delete this folder
- [ ] (from `injection_test.md`) Re-run ALL with 40000 steps and check tau
- [ ] (from `injection_test.md`) Repeat with 5–10 seeds to confirm that the 7–8% kappa error is clearly above the baseline error floor (1.5%)
- [ ] (from `injection_test.md`) Fit the same data with an extended model that includes Fano/TLS/tilt, check whether the true values are recovered, and compare with the FCV formalism
