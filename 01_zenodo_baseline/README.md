# 01_zenodo_baseline

[English](README_en.md) | [日本語](README_ja.md) | 한국어(현재 문서)

초전도 큐빗 avoided-crossing(S21) 베이지안/피셔 분석 파이프라인의
가장 첫 프로젝트. mock 데이터로 파이프라인을 검증한 뒤, 실제 Zenodo
공개 데이터("Emergent Macroscopic Bistability Induced by a Single
Superconducting Qubit", DOI 10.5281/zenodo.10518320 - 상세 출처는
`../DATA_SOURCES.md` 참고)에 적용하기까지의 4단계 진화 과정을 담고
있습니다.

## 폴더 구조 (시간순 4단계)

```
01_zenodo_baseline/
├── v0_first_prototype/         # 1단계: 모듈화 전, 단일 스크립트
│   ├── anticrossing_full_flux_sweep.py   # 최초 배치 MCMC 피팅
│   ├── generate_noisy_mock_data.py       # 1/f잡음+드리프트+outlier mock 생성
│   ├── threshold_methods_comparison.py   # 안정추정 임계값 3방식 비교
│   └── delta_vs_g_uncertainty.py         # |Δ| vs g 오차 진단
│
├── core/                        # 2단계: 재사용 가능한 부품으로 모듈화
│   ├── models.py                # forward model (avoided-crossing S21)
│   ├── likelihood.py            # prior/likelihood (gaussian/robust)
│   ├── mcmc_pipeline.py         # emcee 실행부 (단일/배치, warm-start)
│   ├── diagnostics.py           # 잡음추정, threshold 탐지
│   ├── bayesian_toolkit.py      # credible interval, R-hat, AIC/BIC 등
│   └── fisher_matrix.py         # 피셔행렬 기반 오차추정(MCMC 대안)
│
├── mock_data_validation/        # 3단계: mock 데이터로 파이프라인 검증
│   ├── mock_data.py             # clean신호+잡음주입 생성기
│   ├── example_run.py           # core 6개 모듈을 엮은 실행 예시
│   ├── example_run_bay.py       # + bayesian_toolkit 심층분석 추가판
│   └── compare_gaussian_vs_robust.py  # 정확도(accuracy) 비교
│
└── real_data_application/       # 4단계: 실제 Zenodo 데이터에 적용
    ├── zenodo_loader.py, zenodo_load_real_data.py
    ├── zenodo_models.py, zenodo_likelihood.py, zenodo_mcmc_pipeline.py
    ├── zenodo_bayesian_toolkit.py, zenodo_diagnostics.py, zenodo_fisher_matrix.py
    ├── zenodo_fit_and_compare.py, zenodo_joint_fit.py
    ├── zenodo_multi_file_check.py, zenodo_verify_parser_equivalence.py
```

## 설계 원칙

**바뀌는 부분 (물리계마다 교체)**
- `core/models.py`: 새 물리계를 다룰 때 함수 하나 추가
- `mock_data_validation/mock_data.py`의 clean 신호 생성 로직
- `core/likelihood.py`의 `_theta_to_kwargs` (파라미터 종류가 바뀌면 매핑도 바뀜)

**거의 안 바뀌는 부분 (뼈대)**
- `core/mcmc_pipeline.py`: 샘플러 실행, warm-start, 수렴 진단 로직
- `core/diagnostics.py`: 잡음 추정, threshold 탐지 방법론
- `core/likelihood.py`의 팩토리 함수 구조 (`make_uniform_log_prior`, `make_log_probability`)

## 확장 시나리오

| 상황 | 바꿀 곳 |
|---|---|
| 다른 물리계 (예: 3-큐빗) | `core/models.py`에 함수 추가, `example_run.py`의 STEP 1~4 교체 |
| Outlier에 강건한 분석 | `example_run.py`에서 `aa_likelihood_type = 'robust'`로 변경 |
| 다른 잡음 특성 실험 | `example_run.py`의 `aa_flicker_level`, `aa_drift_amplitude`, `aa_outlier_probability` 조정 |
| 계층적 모델 (파라미터 공유) | `core/mcmc_pipeline.py`를 확장해 여러 slice를 동시에 피팅하는 새 함수 추가 필요 |
| 다른 샘플러 (nested sampling) | `core/mcmc_pipeline.py`에 `run_single_nested()` 같은 병렬 함수 추가 |

## 사용법

```bash
cd 01_zenodo_baseline
# core 모듈들을 import 경로에 잡을 수 있게, mock_data_validation
# 또는 real_data_application 안에서 실행하거나 PYTHONPATH에 core 추가
PYTHONPATH=core python mock_data_validation/example_run.py
```

`example_run.py` 맨 위 `aa_` 변수들을 조정해서 잡음 종류를 켜고 끄거나,
우도 함수(gaussian/robust)를 바꿔가며 실험할 수 있습니다.

## 검증 이력

- `core/models.py`의 `s21_anticrossing_model`: mock 데이터로 fr, g, kappa 모두
  참값과 오차범위 내 일치 확인 (1j 컨벤션 확정 - 2j로 쓰면 결합강도가
  실제의 절반으로 과소평가되는 버그가 있었음)
- `core/diagnostics.py`의 threshold 탐지 3방식 비교: crossing 방식과 inflection
  방식이 서로 다른 접근임에도 동일한 문턱값(|Δ|≈0.29 GHz)으로 수렴,
  discrete-label 방식(방식 A)은 이상치에 취약함을 확인

## 데이터 출처

`real_data_application/`에서 사용한 실측 데이터의 출처는 저장소
최상위 `../DATA_SOURCES.md`를 참고하세요.
