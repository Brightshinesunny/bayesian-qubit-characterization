"""
SUMMARY_fano_avoided_tls_physics_and_stats.py
=================================================================
A write-up of the three physical systems covered so far (Fano resonator / avoided crossing / TLS mapping).
Not executable code, a pure write-up - no need to import it.

=================================================================
Part 1. Relationship among the three physical systems
=================================================================
Fano resonator (linear system)
    - About a single resonator only. The goal is to obtain how strongly it is coupled to the outside (Qc) and its internal
      loss (Qi).

Avoided crossing (interaction of two systems)
    - What happens when a resonator/qubit meets something else (a TLS, another resonator).
      "Coupling", invisible with Fano alone, appears.

TLS mapping (tracking position through avoided crossings)
    - An application that aims to find "where" the avoided crossing occurs (where on the chip
      the TLS is).

In sum: Fano -> avoided crossing -> TLS
was a structure in which difficulty builds up in the order "understand one system -> understand the interaction of two systems -> use that to get down to position".


=================================================================
Part 2. Fano resonator - physics / data problems / statistical response
=================================================================
[Physics]
Obtain how strongly the resonator is connected to the outside (transmission line) (Qc), and the degree to which the
resonator itself stores energy without loss (Qi).
    1/Ql = 1/Qi + 1/Qc

[Problems seen in the data]
- When Ql and Qc are of similar size (near critical coupling), the Qi calculation
  blows up as a division (dividing by a value close to 0)
- The circle fit (the original authors' standard method) gives Qi_max=inf in this situation

[Statistical adjustments]
- Switched to the Bayesian/Fisher approach - instead of a geometric method that forces a fit to one circle,
  the parameter distribution is obtained directly from the likelihood function
- Gaussian vs Robust cross-validation - check whether outliers distort the mean
- Residual autocorrelation diagnostics - check whether the assumption "the noise is independent" holds.
  Several models such as AR(1) and compound symmetry were tried and all failed,
  and in the end "5% outlier trimming + Gaussian/Robust cross-validation" was the most stable

[Parts driven by intuition]
Which systematic-error model to try first (AR(1)? compound symmetry?)
relied on intuition, narrowed down through repeated failures.


=================================================================
Part 3. Avoided crossing - physics / data problems / statistical response
=================================================================
[Physics]
When two systems (qubit-resonator, qubit-TLS) come near the same frequency, they "do not meet but
repel", creating two new eigenfrequencies (hybrid modes).
    omega_± = (f1+f2)/2 ± sqrt((f1-f2)^2 + 4g^2)/2
The key point is that at the resonance point (f1=f2) the gap between the two modes is exactly 2g -
the standard way to measure the coupling strength directly.

[Problems seen in the data]
- The parameter intercept (f_TLS0) was defined at a reference point (V=0) far from the
  measurement range (V=-57 to -56 V), so intercept and slope became completely entangled
  (correlation coefficient 1.0)
- Unit mismatch (the calibration data were in Hz while the model assumed GHz)
- The prior range excluded the true answer (0.109) from the start - even when MCMC said it
  had "converged", it had searched only within a range that could never contain the answer, so the conclusion
  itself was void (a dangerous mistake)

[Statistical adjustments]
- V0 recentering - moved the reference point of the intercept to the center of the measurement range, reducing the correlation
  to 0 (verified directly on synthetic data, 1.0 -> 0.0)
- multi-start - trying from several initial values to mitigate the local-optimum problem
- Real MCMC to check "whether there is one well or several" - a situation that needed the whole
  shape of the posterior, which a single point estimate can never provide

[Parts driven by intuition]
The physical upper limit of g (30 MHz? 500 MHz?) and how narrow to make the frequency search window
were decided by intuition and trial and error.


=================================================================
Part 4. TLS mapping - physics / data problems / statistical response (the part with the most problems)
=================================================================
[Physics]
To find "where" an avoided crossing occurs, the position is back-computed by comparing how strongly
the electric fields of several electrodes shake the TLS (coupling strength gamma). The TLS resonance frequency
shifts with voltage (electric field) through the Stark shift.
    f_TLS(V) = f_TLS0 + gamma_stark * V

[Problems seen in the data]
1. 4 systematic errors (independent of electrode, independent of observable, horizontal lines that hardly change
   with voltage)
2. The dip-tracking algorithm kept getting trapped by systematic errors - 5 fixes were needed
   (indexing -> nonlinear mapping -> crossing point -> stall detection)
3. Finally, on visual inspection, the initial assumption of a "diagonal" itself was
   wrong - the real structure was horizontal multiple bands

[Statistical adjustments]
- Continuity-constrained dip tracking (look for the next dip only near the previous point) +
automatic stall detection (if motion stops, regard it as a systematic error and stop)
- Quality score (SNR) - an absolute criterion for separating real signal from noise
(contrast / background standard deviation)
- Weighted centroid - a simple way to estimate the rough direction without simulation
- Visual verification - what could not be judged from numbers (chi2, posterior) alone
was settled clearly with a single figure

[Parts driven by intuition - the most of them]
The systematic-error masking width (halfwidth), the stall-detection criterion (0.5 MHz), and the quality-score
thresholds (3.0, 3.5) were all tried as "reasonable-looking" values and adjusted
while watching the results.


=================================================================
Part 5. Common patterns running through the three physical systems
=================================================================
                    Fano              Avoided-crossing      TLS
Model mismatch found:    Fano -> the circle fit's geometric assumption breaks down (near critical coupling)
                                             Avoided crossing -> missed V0 recentering
                                             TLS mapping -> the assumption of a diagonal itself was wrong (visual check)

Noise/outlier problems:   Fano -> autocorrelation, several failures (compound symmetry, etc.)
                         Avoided crossing -> prior-range bug
                         TLS mapping -> 4 systematic errors, dip tracking failed 5 times

Final means of validation: Fano -> Gaussian vs Robust cross-check
                         Avoided crossing -> checking the posterior with real MCMC
                         TLS mapping -> visual verification (figures)

A pattern repeated throughout: "the first attempt was always wrong somewhere, and finding that error was
most of the real work". The tool for finding the error differed each time - statistical cross-validation for Fano,
MCMC multi-start validation for the avoided crossing, and in the end looking directly with the eyes for TLS.


=================================================================
Part 6. On "moving by intuition"
=================================================================
More precisely, the work "started" from intuition but did not "end" with intuition. The pattern was the same each time:
1. Choose thresholds/models by intuition (intuition)
2. Verify on synthetic data that the choice actually works (numbers)
3. Apply to real data
4. If the result looks odd -> go back to step 1 and retry with a different intuition
5. Finalize by visual/cross-validation

This is not a bad thing; it is exactly what the "hypothesis-test" scientific method looks like in practice.
Almost no threshold was derived purely from theory; most were a "reasonable
starting point + repeated verification" - this is how real data analysis works.
"""
