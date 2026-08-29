# 10_mathematica_fisher_pipeline

지금까지의 01~09 폴더가 전부 Python(emcee/scipy) 기반이었다면, 이
폴더는 **같은 "GW 물리학의 베이지안/피셔행렬 방법론을 큐빗 신호에
적용한다"는 아이디어를 Mathematica(Wolfram Language)로 별도
구현한 기록**입니다.

## 왜 별도 도구로 다시 했는가

Mathematica는 기호 연산(symbolic computation)과 통계 분포 다루기에
강점이 있어, 이론 중심 연구 그룹에서 여전히 널리 쓰입니다. 같은
목표(피셔행렬 기반 파라미터 불확실성 추정)를 Python과 Mathematica
양쪽에서 구현해본 것 자체가, 도구에 종속되지 않는 방법론적 이해를
보여주는 근거가 됩니다.

## 다루는 신호 모델

avoided-crossing(01~03, 07~09 폴더)과 달리, 여기서는 **감쇠
진동(damped oscillation) 모델**을 다룹니다:

```
signalModel(t, A, gamma, omega) = A * Exp(-gamma*t) * Cos(omega*t)
```

이건 큐빗의 **T2 결맞음 붕괴(decoherence decay) + 라비 진동(Rabi
oscillation)**을 나타내는 표준 모델입니다. 코드 주석에 "중력파
strain 데이터 d(t)=h(t)+n(t)"라는 표현이 그대로 남아있는데, 이는
GW 파형 피팅에 쓰던 신호+잡음 모델(d=h+n)의 틀을 큐빗 신호에
그대로 이식한 흔적입니다.

## 파일 구성 (내용 기반 순서로 정리, 원 파일명은 한글이었음)

| 파일 | 원래 파일명 | 내용 |
|---|---|---|
| `01_single_qubit_fisher_prototype.nb` | 초전도.nb | 최초 프로토타입: 단일 큐빗 감쇠진동 신호 + 피셔행렬 |
| `02_qubit_fisher_pipeline_precision.nb` | (초전도 큐빗 정밀 피셔 파이프라인) | `QubitFisherFixedSNR`, `RunQubitFisherPipeline` 등 함수화 |
| `03_mock_data_fisher_validation.nb` | (가짜 데이터 Mock Data 생성 및 피셔행렬 검증) | `NonlinearModelFit` 기반 검증, 가장 방대한 버전 |
| `04_mock_data_multiqubit_classification.nb` | (Mathematica 기반 Mock Data 생성 및 분류 전처리) | `MultiQubitFisherFixedSNR`, `RunMultiQubitPipeline` — 다중 큐빗으로 확장 |
| `05_psd_two_qubit_cross_coupling.nb` | (확장 코드: PSD, 2-큐빗 교차 결합) | 파워스펙트럼밀도(PSD) + 2큐빗 교차결합 확장 |
| `06_prior_informed_map_4d_fisher_report.nb` | (Prior가 적용된 MAP 4D 피셔 정밀 분석 리포트) | `sigmaGamma2Prior` — prior 정보를 포함한 4차원 MAP 추정 |
| `07_systematic_bias_fisher_pipeline.nb` | (초전도 큐빗 피셔 + Systematic Bias 연산 파이프라인) | 계통오차(systematic bias)까지 포함한 최종 확장판, 가장 큰 파일 |

## 열람 시 참고

`.nb` 파일은 순수 텍스트(ASCII, Wolfram 소스 형식)라 GitHub에서
diff/코드 검토는 가능하지만, **GitHub이 노트북을 렌더링해서
미리보기로 보여주지는 않습니다** (`.ipynb`와 다름). 내용을 그대로
보려면 Mathematica 또는 무료 Wolfram Player로 열어야 합니다.

## 오늘(Python) 작업과의 관계

이 Mathematica 파이프라인과 01~09의 Python 파이프라인은 **서로
다른 신호 모델(감쇠진동 vs avoided-crossing)을 다루지만, 핵심
방법론(피셔행렬, prior, 다중 파라미터 불확실성 추정)은 동일**합니다.
같은 통계적 사고방식을 두 개의 서로 다른 도구/신호모델에 일관되게
적용했다는 걸 보여주는 사례로 볼 수 있습니다.
