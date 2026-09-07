# 10_mathematica_fisher_pipeline

English (this document) | [日本語](README_ja.md)

If folders 01–09 are entirely Python (emcee/scipy), this folder is a
record of the **same idea — applying the Bayesian/Fisher-matrix
methodology from gravitational-wave physics to qubit signals —
implemented independently in Mathematica (Wolfram Language)**.

## Why redo it in a separate tool

Mathematica excels at symbolic computation and handling statistical
distributions, and it remains widely used in theory-focused research
groups. Implementing the same goal (Fisher-matrix-based parameter
uncertainty estimation) in both Python and Mathematica demonstrates a
tool-independent grasp of the underlying methodology.

## Signal Model Covered

Unlike the avoided-crossing model used in folders 01–03 and 07–09,
this pipeline works with a **damped-oscillation** model:

```
signalModel(t, A, gamma, omega) = A * Exp(-gamma*t) * Cos(omega*t)
```

This is the standard model for a qubit's **T2 decoherence decay +
Rabi oscillation**. The code comments still contain the phrase
"gravitational-wave strain data d(t) = h(t) + n(t)" — a direct trace
of transplanting the signal-plus-noise framework (d = h + n) used for
GW waveform fitting straight into the qubit-signal context.

## File Overview (reordered by content; original filenames were in Korean)

| File | Original filename | Content |
|---|---|---|
| `01_single_qubit_fisher_prototype.nb` | 초전도.nb | First prototype: single-qubit damped-oscillation signal + Fisher matrix |
| `02_qubit_fisher_pipeline_precision.nb` | (초전도 큐빗 정밀 피셔 파이프라인) | Turned into functions: `QubitFisherFixedSNR`, `RunQubitFisherPipeline`, etc. |
| `03_mock_data_fisher_validation.nb` | (가짜 데이터 Mock Data 생성 및 피셔행렬 검증) | Validation via `NonlinearModelFit`; the most extensive version |
| `04_mock_data_multiqubit_classification.nb` | (Mathematica 기반 Mock Data 생성 및 분류 전처리) | `MultiQubitFisherFixedSNR`, `RunMultiQubitPipeline` — extended to multiple qubits |
| `05_psd_two_qubit_cross_coupling.nb` | (확장 코드: PSD, 2-큐빗 교차 결합) | Power spectral density (PSD) + 2-qubit cross-coupling extension |
| `06_prior_informed_map_4d_fisher_report.nb` | (Prior가 적용된 MAP 4D 피셔 정밀 분석 리포트) | `sigmaGamma2Prior` — 4D MAP estimation incorporating prior information |
| `07_systematic_bias_fisher_pipeline.nb` | (초전도 큐빗 피셔 + Systematic Bias 연산 파이프라인) | Final, most extended version including systematic bias; the largest file |

## A Note on Viewing These Files

`.nb` files are plain text (ASCII, Wolfram source format), so they
can be diffed and code-reviewed on GitHub, but **GitHub does not
render a notebook preview for them** (unlike `.ipynb`). To view the
content as intended, open them in Mathematica or the free Wolfram
Player.

## Relationship to Today's Python Work

This Mathematica pipeline and the Python pipelines in folders 01–09
deal with **different signal models** (damped oscillation vs.
avoided-crossing), but share the **same core methodology** (Fisher
matrix, priors, multi-parameter uncertainty estimation). Together they
demonstrate the same statistical mindset applied consistently across
two different tools and two different signal models.
