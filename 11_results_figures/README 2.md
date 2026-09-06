# 11_results_figures

이 폴더의 그림 3개는 오늘 문서(README, PORTFOLIO_DETAILED.md)에서 서술한
핵심 발견들을 실제 원본 데이터로 다시 실행하여 얻은 결과입니다. 만들어낸
예시가 아니라, 업로드된 실측 데이터(Fano 공진기 9개 npz 파일, TLS
raw segment mat 파일)와 원저자 방법론을 그대로 이식한 코드
(`fano_models.py`, Rieger & Günzler / Probst circle-fit)를 이용해
직접 재계산한 것입니다.

---

## fig1_qi_divergence.png — 임계결합 근처 내부 Q값 발산

**02_fano_resonator 프로젝트의 핵심 결론을 실제 데이터 9개 전부로 재현.**

9개 공진기(overcoupled 5개, undercoupled 4개) 각각의 중간 power 지점에서
circle fit을 실행해 Ql, Qc를 얻고, Ql/Qc 비율(1에 가까울수록 임계결합)에
대해 |Qi|를 로그스케일로 그렸습니다. Undercoupled 공진기들이 임계결합에
더 가까웠고(Ql/Qc가 1에 근접), 그럴수록 |Qi|가 수만에서 수백만까지
기하급수적으로 발산하는 것이 실제 데이터에서 확인됩니다.

## fig2_circle_fit_example.png — Circle Fit 예시 (가장 극단적 발산 사례)

fig1에서 가장 크게 발산한 resonator_4(undercoupled, Ql/Qc=1.074)를 골라,
왼쪽엔 복소평면 위의 원 피팅(raw 데이터 점 + 피팅된 원), 오른쪽엔 주파수에
따른 |S21| 크기와 피팅 곡선을 나란히 그렸습니다. Ql≈181,500, Qc≈169,000으로
두 값이 서로 매우 가까워, |Qi|가 246만까지 치솟는 것을 시각적으로 확인할
수 있습니다.

## fig3_systematic_bands.png — 계통오차 후보 밴드 (TLS raw 데이터)

TLS raw segment 하나(`quadspec_Segments1to200_20TLSfitted.mat` 내부 참조
데이터)의 2D 진폭 지도를 그린 것입니다. 밝은 세로 밴드 세 개가 전압
스윕 축(y)과 무관하게 특정 주파수 지점(x)에 고정되어 있는 것이 보이는데,
이는 03_avoided_crossing_tls / 05_tls_noise_diagnosis 프로젝트에서 서술한
"게이트 전압과 무관한 계통오차 밴드"와 같은 종류의 패턴을 보여주는
재현 예시입니다. (이 그림의 특정 세 밴드가 당시 확정했던 4개 계통오차와
정확히 동일한 주파수인지는 이번 재실행에서 별도로 대조하지 않았습니다 —
"같은 현상이 실제 데이터에서 이렇게 보인다"는 것을 보여주는 예시로
봐주시면 됩니다.)

---

## 재현 방법

원본 데이터와 코드는 `01_zenodo_baseline/`, `02_fano_resonator/` 폴더의
스크립트를 참고하시면 됩니다. 이 그림들은 다음 방식으로 생성했습니다:

```python
from fano_models import autofit  # 02_fano_resonator/ 의 circle-fit 이식 코드
import numpy as np

d = np.load('resonator_N_powersweep_XXX.npz', allow_pickle=True)
z_data = d['amplitude'][idx] * np.exp(1j * d['phase'][idx])
result = autofit(d['frequency'], z_data)
# result['Ql'], result['Qc'], result['Qi'] 사용
```
