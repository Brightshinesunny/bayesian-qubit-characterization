# Model-Misspecification Injection Test — Summary

A record that quantifies, on synthetic data with known true values, the **systematic bias that model misspecification introduces into parameter estimates** in superconducting resonator and qubit characterization.

---

## 0. The question this test addresses

Real measurements have no true values. Even when a fit converges and the residuals look small, there is no way to tell whether that is **because the model is right** or **because the model is wrong but only looks right**.

An injection test makes that distinction possible. A systematic effect is deliberately added to a signal with known true values, and the data are fitted with a model that **does not include** the effect. Then "how wrong the fit is" can be stated as a percentage.

This test looked at three things.

1. Which parameters are vulnerable to misspecification
2. Whether the contribution of each effect can be separated
3. Whether the usual diagnostics (convergence checks, residuals) catch the misspecification

---

## 1. Physical background

### 1.1 Avoided crossing

When a qubit and a resonator interact with coupling strength $g$, the two systems **repel instead of crossing directly** as their frequencies approach. This is level repulsion in quantum mechanics, a consequence of the Jaynes–Cummings model.

For detuning $\delta = f_q - f_r$, the frequencies of the two hybrid modes are

$$f_\pm = \frac{f_r + f_q}{2} \pm \frac{1}{2}\sqrt{\delta^2 + 4g^2}$$

At $\delta = 0$ the gap has its minimum value $2g$. This is the **standard way to measure the coupling strength directly**.

How much each hybrid mode mixes the original qubit and resonator components is given by the **Hopfield coefficients**, i.e. the mixing angle:

$$\sin^2\theta = \frac{1}{2}\left(1 - \frac{\delta}{\sqrt{\delta^2+4g^2}}\right), \qquad \cos^2\theta = 1 - \sin^2\theta$$

At $\delta = 0$ the mixing is an even $0.5/0.5$; for $|\delta| \gg g$ it is almost entirely one-sided.

### 1.2 Flux dependence of the transmon

Because the effective Josephson energy of the SQUID structure changes with the external flux, the qubit frequency is

$$f_q(\Phi) = (f_{q,\max} + E_C)\sqrt{|\cos(\pi\Phi)|} - E_C$$

$E_C$ is the charging energy. In this test, $f_{q,\max} = 5.150$ and $E_C = 0.250$ GHz.

### 1.3 Dispersive and resonant regimes

When $|\delta| \gg g$ the system is in the **dispersive regime**. The two systems do not exchange energy directly and only push each other's frequencies slightly. This regime is used for qubit readout, but it is the **worst operating point for measuring $g$**.

At the initial setting $\Phi = 0$ in this test,

$$f_q(0) = 5.150,\quad \delta = +0.150 = 3.75g$$

The Hopfield coefficients are lopsided at $0.941 / 0.059$, so **effectively only one dip is visible**. The only observable is then a single dip position, while there are two unknowns $(f_r, g)$, so they become **degenerate**. In this state the estimate of $g$ is decided by the prior, not by the data.

The actual avoided crossing is at

$$\sqrt{|\cos(\pi\Phi)|} = \frac{f_r + E_C}{f_{q,\max} + E_C} = \frac{5.25}{5.40} \;\Rightarrow\; \Phi = \pm 0.1059$$

and a scan selected $\Phi = -0.1062$ for this test.

### 1.4 Linewidth and quality factor

$\kappa$ is the decay rate of the resonator and corresponds to the half width at half maximum (HWHM) of the Lorentzian. Its relation to the quality factors is

$$Q_l = \frac{f_r}{\kappa_{\rm total}}, \qquad \frac{1}{Q_l} = \frac{1}{Q_i} + \frac{1}{Q_c}$$

$Q_i$ is the internal loss (material defects, etc.) and $Q_c$ is the external coupling loss. **An error in $\kappa$ propagates directly into the reported $Q$ values.**

### 1.5 The three injected systematic effects

**Fano resonance.** An ideal Lorentzian resonance assumes a single path into the resonator. In real wiring there is always a direct leakage path that bypasses the resonator, and when the two paths interfere, the lineshape becomes asymmetric. The concept was introduced in atomic physics to explain interference between a discrete state and a continuum.

The implementation uses the **complex coupling Q** method (Khalil et al. 2012, Probst et al. 2015):

$$\tilde{Q}_c = Q_c\cos\phi\, e^{-i\phi}, \qquad \phi = -\arctan(1/q)$$

Because the asymmetry enters as a phase rotation, the baseline is preserved.

> **Note.** The initial implementation multiplied the signal by $(q+x)^2/(1+x^2)$ and then divided by the maximum. This function already converges to 1 as $x \to \pm\infty$, so dividing again by the maximum $1+q^2$ **shrank the whole signal by a factor of $1/(1+q^2)$** (a factor of 5 for $q=2$). A standard model that assumes a baseline of 1 cannot match this for any parameter combination, which produced the 140% error in $g$ and the 233% error in $\kappa$ in the first test.

**TLS (two-level system).** A tiny defect in the superconducting chip surface or dielectric layer behaves like a small qubit and couples weakly to the resonator at a specific fixed frequency. It produces a narrow extra dip that is not in the theoretical model. If it is mistaken for a true avoided-crossing peak, the fit converges to completely different values.

**Baseline tilt.** A tilt of the transmission baseline, for example from the frequency-dependent loss of the cabling.

---

## 2. Statistical background

### 2.1 Bayesian inference

$$p(\theta \mid D) \;\propto\; p(D \mid \theta)\, p(\theta)$$

**Posterior** $\propto$ **likelihood** $\times$ **prior**.

This test used a uniform prior and a Gaussian likelihood. A Gaussian likelihood requires the noise $\sigma$, and **if this $\sigma$ is wrong, the width of the likelihood is wrong.** See §4.8.

### 2.2 MCMC and autocorrelation time

emcee is an ensemble sampler in which multiple walkers explore the posterior while referring to each other's positions.

The **autocorrelation time** $\tau$ is "the number of steps a chain needs to produce one independent sample". The effective sample size is $N_{\rm eff} \approx N/\tau$.

The standard emcee criterion is

$$N > 50\tau \quad\Longleftrightarrow\quad \tau < N/50$$

which means securing at least 50 effective samples.

**Burn-in** discards the initial transient, and **thinning** removes correlated samples.

### 2.3 MAP and median

**MAP (Maximum A Posteriori)** is the point where the posterior is largest, i.e. one of the actual samples.

The **median** is a component-wise median, so if the posterior is asymmetric or the parameters are strongly correlated, it can be a **point that matches none of the samples**.

In this test the `result` dictionary had no keys for the samples and log-probabilities, so the median was used instead. The posterior was fortunately symmetric and the median column and MAP column came out identical, but this is the result of falling back because MAP was not computed, and **it has not been verified**.

### 2.4 Identifiability and degeneracy

**Identifiability** is whether the data can determine the parameters uniquely.

When there are more unknowns than observables, or when several parameter combinations give the same prediction, the problem is **degenerate**. A degenerate problem has a singular Fisher matrix, no inverse exists, and MCMC drifts to the prior boundary.

The dispersive regime in §1.3 is exactly this situation.

### 2.5 Systematic bias versus statistical error

**Statistical error** comes from noise and decreases as $\propto 1/\text{SNR}$.
**Systematic bias** comes from model misspecification and **does not depend on SNR**.

So the more precise the measurement becomes, the more systematic bias dominates in relative terms. This is the starting point of the research direction.

### 2.6 Contact with the prior boundary

If an estimate sticks to a prior boundary, the value is not "what the data showed" but **"the edge of the search range"**. The error in that state cannot be interpreted as the size of the bias.

In the first test, $g = 0.09623$ and $\kappa = 0.10000$, and the upper bound was 0.1 for both. So **the numbers 140% and 233% were not measurements of bias but the walls of a box.**

---

## 3. Test setup

### True values

| Parameter | Value | Meaning |
|---|---|---|
| $f_r$ | 5.000 GHz | Resonator frequency |
| $g$ | 0.040 GHz | Qubit–resonator coupling strength |
| $\kappa$ | 0.030 GHz | Resonator decay rate |
| $\sigma$ | 0.015 | White noise |

$f_{q,\max} = 5.150$, $E_C = 0.250$, and cable delay $\tau = 0.12$ are fixed constants.

### Operating point

$\Phi = -0.1062$ was selected by a scan.

- Range where two dips are visible: $[-0.131, +0.131]$
- Minimum gap $0.0780$ GHz vs. $2g = 0.0800$ (a 2.5% difference, caused by the 1 MHz grid resolution and the pulling of the dip positions by the finite linewidth)
- Hopfield coefficients $0.500/0.500$, $g$ sensitivity $1.000$ (maximum)
- splitting/$\kappa = 2.67 > 1$, so the two dips are resolved by the linewidth

### Injected effects

| Effect | Parameters |
|---|---|
| Fano | $q = 2.0$ (complex-phase method) |
| TLS | offset $+0.005$ GHz, depth $0.4$, linewidth $0.006$ GHz |
| tilt | slope $0.15$ |

### MCMC

`nwalkers=32`, `nsteps=25000`, `burn_in=5000`, `thin=10`, `seed=7`
prior: $f_r \in (4.8, 5.2)$, $g, \kappa \in (0.001, 0.50)$

---

## 4. Results

<img width="2158" height="729" alt="ALL_clean version" src="https://github.com/user-attachments/assets/6f398fc6-f5b5-4f31-ba74-4f182c674955" />


### 4.1 Summary table

| Scenario | $f_r$ | $g$ | $\kappa$ | $\sigma$ ratio | prior | $\tau$ |
|---|---|---|---|---|---|---|
| baseline | 0.0 % | 0.1 % | 1.5 % | 2.42x | ok | ok |
| Fano | 0.2 % | 0.1 % | **7.3 %** | 5.52x | ok | ok |
| TLS | 0.0 % | 0.5 % | **8.1 %** | 2.43x | ok | ok |
| Tilt | 0.0 % | 0.1 % | 1.4 % | 3.02x | ok | ok |
| ALL | 0.3 % | 1.2 % | 3.3 % | 6.06x | ok | **FAIL** |

No scenario touched a prior boundary. These errors are therefore values the data showed, not the edge of the search range.

### 4.2 baseline — validating the pipeline

The true values were recovered to $0.0 / 0.1 / 1.5\%$. In the figure, the gray (ideal), green (distorted), and red (fit) curves overlap completely and cannot be told apart.

**Only because this row is clean are the other rows comparable.** It also means the pipeline's own error floor is about 1.5% (mainly in $\kappa$).

### 4.3 Robustness to chain length

Comparison of 8000 and 25000 steps.

| | 8000 | 25000 | Change |
|---|---|---|---|
| Fano $\kappa$ | 7.4 % | 7.3 % | −0.1 |
| TLS $\kappa$ | 8.0 % | 8.1 % | +0.1 |
| ALL $\kappa$ | 3.7 % | 3.3 % | −0.4 |

**Even at 8000 steps, which did not meet the convergence criterion, the estimates themselves were already stable.** Insufficient convergence makes the uncertainty estimate unreliable, but it did not shift the median much.

This shows the robustness of the result, and also that **convergence and accuracy of the estimate are separate questions**.

### 4.4 $f_r$ — looks the most accurate, but is the least sensitive

Within 0.3% in all scenarios. The resonator frequency is tied directly to a strong observable, the dip position, so it hardly moves even when the model is wrong.

**So one must not judge the success of a fit by the accuracy of $f_r$.** In the first (misconfigured) test, $f_r$ was accurate to 0.1% while $g$ and $\kappa$ were off by 140% and 233%. **The best-looking parameter is the least sensitive to misspecification.**

### 4.5 $g$ — robust when the operating point is right

Within 1.2% in all scenarios. At an avoided crossing, $g$ is observed directly as the dip spacing $2g$, so it resists distortion.

In the dispersive regime at $\Phi = 0$, this value was dominated by the prior. The lesson here is that **the choice of operating point decides identifiability**.

### 4.6 $\kappa$ — where the distortion collects

The only parameter above 5%. And **the direction is opposite for different effects**.

| Effect | $\kappa$ error | Direction |
|---|---|---|
| Fano | 7.3 % | Underestimate (0.02780) |
| TLS | 8.1 % | Overestimate (0.03242) |
| ALL | 3.3 % | Overestimate (0.03098) |

Because $\kappa$ is a linewidth parameter, it is the only path through which the model can absorb lineshape distortion that it does not contain.

**How Fano is absorbed.** The two dips in the data shifted left from $4.960/5.040$ to $4.956/5.037$, and their depths became asymmetric (left 0.49 / right 0.57).

The standard model has **only one way** to make the dip depths asymmetric: make the detuning nonzero so that the Hopfield coefficients are unequal. The fit indeed lowered $f_r$ to $4.98757$, creating $\delta = +0.0124$, which produced the asymmetry $\cos^2\theta = 0.577$ / $\sin^2\theta = 0.423$.

That is, **the Fano asymmetry was absorbed through the path $f_r$ shift → detuning → Hopfield-coefficient asymmetry.** This is also why $f_r$ moved by 0.2% (against 0.0% in the other scenarios). It is a case where the distortion propagated to an unexpected place through parameter correlation.

**How TLS is absorbed.** The narrow dip at $5.005$ GHz sits on the peak between the two main dips. The peak height is set by the overlap of two Lorentzians, so the model increases $\kappa$ to increase the overlap and lower the peak. In the figure this appears as a fitted peak of 0.855 against an ideal peak of 0.875.

**Why tilt was harmless.** The tilt is a slowly varying multiplicative factor, so it is almost constant near a dip. Since the local shape of the dip does not change, $\kappa$ was not affected. The residual remains only in the wings, and the inflated $\sigma$ (3.02x) tolerated it.

### 4.7 Cancellation — looking only at the simultaneous case underestimates

Fano is $-7.3\%$ and TLS is $+8.1\%$, yet ALL is $+3.3\%$.

The simple sum is $+0.8\%$, so **it is not linear superposition either.** The three effects mix nonlinearly and partially cancel.

Looking only at ALL would have **underestimated the size of each effect by more than half**. This is direct evidence that **a design that enables the effects one at a time is essential**.

For the planned mapping of the parameter space, it also means that **regions with cancellation exist**. In those regions the observed bias is small even though the individual systematic effects are large.

### 4.8 The noise estimate is also contaminated

| Scenario | Estimated $\sigma$ | Ratio to true |
|---|---|---|
| baseline | 0.0363 | 2.42x |
| Fano | 0.0829 | **5.52x** |
| TLS | 0.0365 | 2.43x |
| Tilt | 0.0453 | 3.02x |
| ALL | 0.0910 | **6.06x** |

The adaptive estimate absorbs the resonance structure and the distortion residuals as noise. Even the baseline is 2.4 times too large.

If $\sigma$ is overestimated, the Gaussian likelihood widens and **large residuals are statistically acceptable**. This is a secondary reason why misspecification does not show up in $\chi^2$ or residual metrics.

### 4.9 The model's degrees of freedom decide where distortion goes

In all five panels, the red curve converges exactly to $1.0$ at both ends of the scan, while the data do not.

| Scenario | Left end | Right end |
|---|---|---|
| baseline | 1.00 | 1.00 |
| Fano | 0.96 | 1.05 |
| Tilt | 0.93 | 1.07 |
| ALL | 0.89 | 1.10 |

The standard model `s21_anticrossing_model` has neither an amplitude scale $a$ nor a baseline-tilt parameter, so structurally $|S_{21}| \to 1$.

**This "inability to follow" paradoxically protected the accuracy of $f_r$ and $g$.** There was no freedom to absorb the baseline deviation, so it had to remain in the residuals and did not leak into the parameters.

Conversely, in a model that has $a$ and $\alpha$ (such as `Sij`), the same distortion would be absorbed by the parameters and **could produce a larger bias.** More freedom is not always safer.

This is an untested hypothesis and is left as a follow-up item in §7.

---

## 5. Assessment of the diagnostics

### 5.1 Convergence check

The four individual scenarios passed $\tau < N/50$ at 25000 steps. Only ALL fails.

```
ALL:  tau = [172.1, 123.2, 643.2],  N/50 = 500.0
```

Note that **$\tau$ for $\kappa$ actually increased from 426.6 at 8000 steps to 643.2 at 25000 steps.**

When $\tau$ grows with chain length, it is usually a signal that **a slow mode appeared only in the longer chain.** A short chain traverses that mode only a few times and underestimates $\tau$. So the 426.6 at 8000 steps was itself unreliable.

The overlap of three distortions may have created a narrow, elongated valley in the posterior along the $\kappa$ direction. A re-run with 40000 steps can check this, and if $\tau$ still grows there, the problem is the structure of the posterior rather than chain length, and the number of walkers needs to be increased.

### 5.2 Reliability of the convergence flag

`converged: True` was output for all scenarios, yet ALL does not meet the $\tau$ criterion. **The pipeline's convergence flag disagrees with the standard emcee criterion.** The flag's logic needs to be checked.

### 5.3 Residual-based metrics

As in §4.8, $\sigma$ inflates by 2.4–6.1 times, so $\chi^2$-type metrics cannot detect the distortion.

**Visual verification was decisive.** Overlaying gray, green, and red in the figure showed immediately where the fit deviates. Numbers alone could not provide that information.

---

## 6. Conclusions

1. **The choice of operating point decides parameter identifiability.** At the avoided crossing $g$ sensitivity is maximal and the recovery error was within 1.2% in all scenarios. In the dispersive regime $(f_r, g)$ are degenerate and the prior decides $g$.

2. **Model misspecification concentrates in $\kappa$.** $f_r$ and $g$ stay near 1%, while only $\kappa$ shows an error more than 5 times larger. Since $\kappa$ propagates directly to the reported $Q$ values, the practical impact is large.

3. **Effects can cancel.** With Fano ($-7.3\%$) and TLS ($+8.1\%$) together, the result looks like $+3.3\%$, a nonlinear mixture that is not even the simple sum ($+0.8\%$). A design that enables effects individually is essential.

4. **The parameter that looks most accurate is the least sensitive.** The accuracy of $f_r$ must not be used to judge the validity of a fit.

5. **The model's degrees of freedom decide where the distortion goes.** In a model without baseline freedom, the distortion stayed in the residuals and did not contaminate the parameters.

6. **The noise estimate itself is contaminated by misspecification, in the direction that hides it.**

7. **Convergence and accuracy of the estimate are separate questions.** The median was stable even without meeting the convergence criterion. Conversely, the convergence flag can be True while the standard criterion is not met.

---

## 7. Next steps

**Near term**

- Re-run ALL with 40000 steps and check $\tau$. If it still grows, increase walkers to 64
- Run 5–10 seeds to confirm that the 7–8% in $\kappa$ is clearly above the baseline error floor (1.5%)
- Check the keys of the `result` dictionary and actually compute MAP

**Main research direction**

- Fit the same data with an **extended model that includes** Fano, TLS, and tilt, and check whether the true values are recovered
- **Predict analytically with the Fisher–Cutler–Vallisneri formalism** the bias arising from the difference between the standard and extended models, and compare with the measured values in this test
- Sweep $q$, TLS depth and offset, and tilt slope to identify in parameter space the **conditions under which bias exceeds statistical error**, and locate the cancellation regions
- Also test realistic TLS depths ($0.02$–$0.15$). The $0.4$ used here is extreme
- Run the same injection on a model with an amplitude degree of freedom $a$ (the `Sij` family) and see whether the destination of the distortion changes
