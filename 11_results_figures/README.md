# 11_results_figures

The three figures in this folder reproduce key findings described in the repository README and `PORTFOLIO_DETAILED.md`, re-run on the actual original data. They are not constructed examples: they were recomputed from the measured data (9 Fano resonator `npz` files and a TLS raw-segment `mat` file) with code that ports the original authors' methodology (`fano_models.py`, the Rieger & Günzler / Probst circle fit).

---
<img width="1350" height="900" alt="fig1_qi_divergence" src="https://github.com/user-attachments/assets/dca3d209-3c1d-41cd-9cfb-4950ab55913e" />
## fig1_qi_divergence.png — Divergence of the internal Q near critical coupling

**Reproduces the main conclusion of `02_fano_resonator` on all 9 real resonators.**

For each of the 9 resonators (5 overcoupled, 4 undercoupled), a circle fit at the mid-power point gives Ql and Qc. |Qi| is then plotted on a log scale against the ratio Ql/Qc (closer to 1 means closer to critical coupling). The undercoupled resonators sit closer to critical coupling, and as Ql/Qc approaches 1 the real data show |Qi| diverging by orders of magnitude, from tens of thousands up to millions.


<img width="1950" height="825" alt="fig2_circle_fit_example" src="https://github.com/user-attachments/assets/904cacfc-30a6-4819-9df6-060921759f7a" />
## fig2_circle_fit_example.png — Circle-fit example (the most extreme divergence)

This is resonator_4 (undercoupled, Ql/Qc = 1.074), the case that diverges most in fig. 1. The left panel shows the circle fit in the complex plane (raw data points and the fitted circle); the right panel shows |S21| versus frequency with the fitted curve. Ql ≈ 181,500 and Qc ≈ 169,000 are very close to each other, which is why |Qi| shoots up to about 2.46 million.

## fig3_systematic_bands.png — Candidate systematic-error bands (TLS raw data)

A 2D amplitude map of one TLS raw segment (reference data inside `quadspec_Segments1to200_20TLSfitted.mat`). Three bright vertical bands stay at fixed frequencies (x) regardless of the voltage sweep axis (y). This is the same kind of pattern as the "gate-voltage-independent systematic bands" described in `03_avoided_crossing_tls` and `05_tls_noise_diagnosis_dip_tracking`. It is shown as an example of the phenomenon in real data. This re-run did not check whether these three bands are at exactly the same frequencies as the four systematic artifacts identified earlier.

---

## How to reproduce

For the original data and code, see the scripts in `01_zenodo_baseline/` and `02_fano_resonator/`. The figures were generated as follows:

```python
from fano_models import autofit  # circle-fit port in 02_fano_resonator/
import numpy as np

d = np.load('resonator_N_powersweep_XXX.npz', allow_pickle=True)
z_data = d['amplitude'][idx] * np.exp(1j * d['phase'][idx])
result = autofit(d['frequency'], z_data)
# use result['Ql'], result['Qc'], result['Qi']
```
