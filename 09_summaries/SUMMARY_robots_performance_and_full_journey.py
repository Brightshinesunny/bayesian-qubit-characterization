"""
SUMMARY_robots_performance_and_full_journey.py
=================================================================
[Consolidated summary] Practical performance evaluation of validation_pipeline.py + the final version of the whole journey
(Fano/avoided crossing/TLS/catalog mapping). A pure write-up,
not executable code.


=================================================================
Part 1. The 8 validation tools - practical performance table
=================================================================
Building the tools was not the end; this summarizes whether their performance was actually verified by deploying them in several situations.
The principle that "built" and "actually usable" are different matters applies equally to the
tools themselves.

  Tool 1 (prior check)
    Practical performance: O (immediate success)
    - Exactly reproduced and caught the real gamma_stark bug (-0.01 to 0.01 excluded the answer 0.109).
      Passed without any modification.

  Tool 2 (parameter recovery)
    Practical performance: O (immediate success)
    - Clearly separated the normal pipeline (90% pass) from the V0-bug pipeline (3.3% pass).
      No further modification needed.

  Tool 3 (multi-start consistency)
    Practical performance: O (immediate success)
    - On a normal linear model, the 4 starting points converged perfectly with relative deviation ~0.

  Tool 4 (parameter correlation)
    Practical performance: O (immediate success)
    - Exactly reproduced the real V0 bug (correlation coefficient 1.0) and even output
      the recentering suggestion correctly.

  Tool 5 (residual autocorrelation)
    Practical performance: O (success without modification, though the test data were reselected)
    - The first attempt (a periodic signal) changed sign with the lag, making the verdict
      ambiguous -> when the test data were changed to a random-walk type, closer to what was actually encountered,
      a smooth decay from lag=1 (0.999) to lag=500 (-0.349) was confirmed,
      and the warning fired correctly.

  Tool 6 (Gaussian vs Robust)
    Practical performance: Δ (design change was needed)
    - The initial design (soft_l1 loss) did not exceed the threshold (0.2) even with severe outliers
      inserted (7 points, size 15-20) - lack of sensitivity confirmed.
    - Replaced with the Student-t log-likelihood + minimize approach that had been verified in real use throughout the work,
      and the warning fired correctly for extreme outliers (size 50-60).
      The only tool that "needed a redesign in practice".

  Tool 7 (quality score/SNR)
  Practical performance: Δ (the synthetic validation succeeded, but an important
  limitation was found in practical use - ultimately improved through v3)
  - The simple synthetic-data test (strong signal 5.2 vs noise 0.8) passed immediately.
  - However, in the catalog -> raw reverse mapping, using it as is with the "whole-map minimum" criterion
  led to wrongly adopting a background unrelated to the target frequency (a false number one)
  (the raw_index=503 incident).
  - Only after redesigning it as a "local vs global comparison" (v3) did it separate
  cases A/B/C exactly.
  - Lesson: even for the same tool, "the minimum over everything" and "the minimum inside the region of interest"
  answer completely different questions - when reusing in a different context it must always be re-verified.

  Tool 8 (comprehensive diagnosis)
    Practical performance: O (immediate success)
    - Separated exactly scenario A (normal) = "trustworthy" (0 issues) and scenario B (the real
      V0+prior bug combination) = "hard to trust" (3 issues).

[Part 1 overall assessment]
Six of the 8 worked correctly immediately after design; two (Gaussian/Robust comparison, quality
score) showed their limits only after being deployed in practice and needed redesign.
This is also a miniature of the whole project - rather than "designing everything correctly in advance",
the core was the repetition of "build it, use it in practice, and when it is wrong, diagnose why and fix it".


=================================================================
Part 2. Summary of the whole journey (physics + statistics, in chronological order)
=================================================================

--- Fano resonator ---
Physics: 1/Ql = 1/Qi + 1/Qc. Separate the external coupling (Qc) and the internal loss (Qi).
Problem: near critical coupling, the Qi calculation blows up as a division. The circle fit gave
  Qi_max=inf.
Solution: switch to the Bayesian/Fisher approach. Establish Gaussian vs Robust cross-validation.
  Residual autocorrelation diagnostics (looking at short lags only can mislead - wide lags are needed).

--- Avoided crossing (qubit-TLS coupling) ---
Physics: two systems repel near resonance and create hybrid modes (2 new eigenfrequencies).
  At the resonance point the gap = 2g.
Problems: the intercept (f_TLS0) was at a reference point (V=0) far from the measurement range, so it was
  completely entangled with the slope (correlation 1.0). Unit mismatch (Hz vs GHz). The prior range
  excluded the real answer (0.109) from the start.
Solutions: V0 recentering (correlation 1.0 -> 0.0 verified). multi-start. Real
  MCMC to check for multiple peaks (both the small-gamma and large-gamma chains converged to small values
  -> 0.109 was not a real well).
Final twist: visual verification showed that the assumption of a "diagonal" itself was wrong -
  the real map had a horizontal multiple-band structure.

--- Decoding the TLS data structure ---
raw (639 segments, round-robin over 4 electrodes) / qset (settings) / tdat (fit results)
tabledata2 (13 rows x 200 columns): coupling strength (rows 0, 2, 9, 11) and frequency (row 4) established.
  Rows 1, 3, 10, 12 were guessed to be "symmetry-point voltages", but on actual checking
  gave physically impossible values such as -151 V, +1240 V -> that assumption was discarded (presumed to be
  derived values, identity unconfirmed).
4 systematic errors found (3 flux-based + 1 frequency-based), dip tracking
  fixed 5 times (indexing -> nonlinear mapping -> crossing point -> stall detection).

--- Catalog <-> raw reverse-mapping journey ---
v1 (voltage matching): failed. It established that the row-1 etc. values are not voltages.
v2 (frequency + main electrode): the filter hardly filtered anything (the frequency window is always the same across all
  segments) - the strongest background point was adopted as a false number one.
v3 (local vs global quality score + A/B/C classification): filtered out the false number one exactly.
  Column 4: 0 case-A hits in all 639 segments of the 4 electrodes.
v4 (systematic check of 4 electrodes) + multi-column scan (all 19 entries checked):
  only 3 of 19 (15.8%) confirmed as case A in raw. Column 17 (597 hits, nearly
  all) is this file's "truly dominant background". Column 4 and the rest were provisionally concluded to be in the realm of the original authors'
  more sophisticated methods (such as stitching several slices).
Final check of column 17: slope re-confirmed by continuity tracking -> 0.356 MHz/V
  (1/1300 of the catalog's 452 MHz/V) -> column 17 is also effectively a horizontal systematic error,
  possibly a background unrelated to the catalog value.

[Part 2 conclusion] The most important general principle found: "trying to match the catalog (fit results) and
raw (original data) 1:1 by single-frame pixel search alone" failed for most of this dataset
(16 of 19, 84%). That the original authors' own algorithm ("quin") uses a far more sophisticated method than ours (stitching several segments,
etc.) was confirmed consistently through several independent routes (visual inspection, pixel tracking, statistical scan,
continuity tracking).


=================================================================
Part 3. Methodological lessons running through the work (final version)
=================================================================
1. Always validate a new function/model first on synthetic data (repeated more than 30 times)
2. The validation tools themselves need re-validation (tools 6 and 7 were redesigned in practice)
3. "The maximum/minimum over everything" and "inside the region of interest" are different questions -
   when the context changes, the same code must be re-validated
4. A single failure (column 4) alone cannot separate "the method is wrong" from "this entry is unusual" -
   the pattern appears only by repeating over several entries systematically
5. Do not conclude from numbers (chi2, posterior) alone; always do visual verification
   - for both the avoided crossing and column 17, visual verification was the decisive moment that revealed that "the assumption
     itself was wrong"
6. It is normal for things to take disheartening amounts of time - the results that appear in papers (20/639
   segments, 3%) are the tiny fraction that survived all of this process
"""
