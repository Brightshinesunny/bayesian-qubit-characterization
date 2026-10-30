"""
SUMMARY_statistical_toolkit_playbook.py
=================================================================
Separately from the per-dataset write-up (SUMMARY_fano_avoided_tls...), a document organizing the
statistical tools used here by "the tools themselves". A pure write-up,
not executable code.

=================================================================
1. Gaussian (simple least squares) vs Robust (Student-t)
=================================================================
[What it does]
Both are rules that decide "how badly to treat the difference (residual) between model and data".
Gaussian penalizes residual^2 (being squared, it is extremely sensitive to large residuals).
Robust (Student-t) has the form log(1+residual^2/nu), so it is far more tolerant of large residuals.

[When to use]
- When you suspect outliers but are not sure: run both and compare
- If the two give almost the same answer (confirmed several times) -> trust that answer
- If the two differ greatly -> a sign that outliers are actually distorting the result

[Mathematical relationship]
Sending nu (degrees of freedom) to infinity makes Robust exactly Gaussian. That is,
Robust is a broader concept that "contains" Gaussian.

[Where it was actually used]
Mean estimation after removing outliers in the Fano data, statistics of the 20 TLS catalog values.


=================================================================
2. Fisher matrix / Laplace approximation
=================================================================
[What it does]
Takes the second derivative (curvature) of the log-likelihood (or log posterior) at the optimum and expresses "how precisely
this parameter is known" as a matrix. Inverting it gives the covariance, which holds
the uncertainty of each parameter and the correlations among parameters.

[A way to imitate MCMC without emcee]
Drawing samples from a "multivariate normal distribution" with the covariance obtained from the Fisher matrix gives,
under the assumption that "the posterior is exactly Gaussian-shaped", conceptually the same information as MCMC samples.
This method was used as a substitute whenever emcee was not available in the sandbox.

[Important limitation - encountered first-hand]
This method assumes that the posterior has "a single peak". If it actually has several
peaks (multi-modal), it gives a completely wrong picture - this is exactly the problem met in the avoided-crossing
analysis (a point estimate or Fisher matrix can never tell whether there is one well or several,
and real MCMC was needed).


=================================================================
3. Real MCMC (emcee) - how does it differ from the Fisher matrix
=================================================================
[What it does]
Directly explores the "whole shape" of the posterior. Several walkers move around the parameter
space, staying long in plausible places and leaving implausible ones quickly -
as a result the sample density becomes the shape of the posterior itself.

[What only MCMC can do, and the Fisher matrix cannot]
- Check whether there are several peaks (multi-modality)
- Compare the "weight" of a given peak (how plausible that answer is)
- Capture asymmetric or distorted posterior shapes accurately

[The key case met here - avoided crossing]
- Starting from the first (small-gamma) point: converged to a particular well
- Starting from the large-gamma (0.109) point: converged elsewhere, but run separately for confirmation
- Conclusion: both chains eventually drifted toward small values -> only MCMC could confirm that
  the large-gamma well was not real

[Important - a prior bug can completely invalidate an MCMC result]
If the prior range excludes the real answer from the start, then even if MCMC reports "converged"
the conclusion itself is meaningless. In the gamma_stark case, the prior was narrow at (-0.01,0.01),
so 0.109 could never be explored - the prior range must always be rechecked before running MCMC.


=================================================================
4. Prior design
=================================================================
[What it does]
Fixes in advance "the range this parameter can take". A uniform
prior is the most common - every value inside the range is considered equally possible.

[Key lesson]
The basic principle is to make the prior range "reasonably generous". If it is too narrow (as in the
gamma_stark case), it excludes the true answer from the start, and all the results that follow
(including whether MCMC converged) become void. Conversely, if it is too wide,
the search becomes inefficient or allows physically meaningless values (for example, a negative quality
factor) - it is important to tighten it moderately to a physically valid range.


=================================================================
5. Parameterization - V0 recentering
=================================================================
[Problem]
When a line (or gentle curve) model is written as y = a + b*x and the "intercept a" is defined as the value at
x=0, then if the actual measurement range is far from x=0, a and b become severely entangled
(the correlation coefficient approaches 1).

[Solution]
Write it as y = a + b*(x - x0) and redefine x0 as the "center" of the measurement range. Then
a becomes "the value at the middle of the measurement range" and is estimated almost independently of b (the slope).

[Numbers verified directly]
Without V0: correlation coefficient 1.0000 (completely entangled)
With V0:    correlation coefficient 0.0000 (completely separated)

[Generalization]
A general principle that applies to polynomial regression, splines, and any situation that "deals with a range
far from the origin". Here it was applied to f_TLS0 of the avoided crossing.


=================================================================
6. Multi-start optimization
=================================================================
[Problem]
Gradient-descent-type optimizers (curve_fit, etc.) only descend into the "nearest valley" to the starting point -
they cannot find the true optimum even if it lies in a different valley
(the local minimum problem).

[Solution]
Run the optimization from several different initial values and adopt the one with the lowest loss
(chi2, etc.).

[Situations met repeatedly]
- Fano data R4: FWHM-based initial values all failed; it was solved only after adding the circle-fit result
  as a candidate
- TLS dip tracking: several initial values of phi (phase) were needed
- Avoided crossing: the sign of gamma_stark was unknown, so both +/-109 were needed

[Limitation - multi-start is not perfect either]
If the chi2 values from several initial values are "similar to each other", this may be a sign not that the
local optima were all missed but that the true posterior is multi-modal - in that case do not stop at multi-start
and move on to MCMC (see 3 above).


=================================================================
7. Residual diagnostics
=================================================================
[Autocorrelation]
Checks whether the assumption that the residuals are "independent" holds. If neighboring residuals resemble each other
(autocorrelation departs from 0), the error estimate of simple least squares is inaccurate.
- A pitfall learned: looking only up to lag=29 made the correlation look as if it "hardly decreased",
  which almost led to a wrong judgment -> only when the lag was widened to 900 did the true decay
  structure (length scale) appear.

[Quality score / signal-to-noise ratio (SNR)]
An absolute criterion for judging whether "what was found is a real signal or noise". Pure noise can also
happen to make a "plausible pattern" (verified directly: a 21-point
track also arose from pure noise), so relative criteria such as length/monotonicity are not enough; an absolute criterion such as
dip depth relative to the background is needed.


=================================================================
8. The standard workflow settled on
=================================================================
1. Get safe initial/background values with a fast algebraic method (circle fit, etc.)
2. Lightly remove outliers (3-5%)
3. Gaussian vs Robust cross-validation - if the two agree, trust the point estimate
4. Confidence intervals from the Fisher matrix (first-order approximation, an alternative when emcee is not available)
5. Residual autocorrelation diagnostics (both short and long distances, a wide lag range)
6. For models with several entangled parameters (2D, etc.), use multi-start
7. If the chi2 values of the multi-start results are similar to each other -> check the whole posterior
   with real MCMC (rechecking the prior range is mandatory)
8. Finally, always do a visual check (overlay the model on the measured data) -
   never conclude from numbers alone. In the avoided crossing, this last step overturned
   the hypothesis (a diagonal) first set up.
"""
