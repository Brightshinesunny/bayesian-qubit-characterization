"""
zenodo_diagnostics.py  (diagnostics.py를 Zenodo 실측 데이터 워크플로우 전용으로 복제한 버전)
=================================================================
분석 결과를 검증/시각화하는 범용 도구 모음. 물리 모델이 바뀌어도
거의 그대로 재사용 가능한 부품들입니다.

포함된 기능:
  - baseline 기반 잡음(noise_sigma) 추정 (고정 비율 / 적응형)
  - 이동평균 스무딩
  - "안정 추정 가능한 범위"를 찾는 세 가지 방식 (이산라벨 극값 / threshold-crossing / 변곡점 자동탐지)
"""

import numpy as np


def moving_average(x, window):
    """이동평균 스무딩. window(윈도우 크기)가 클수록 더 매끈해짐."""
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='same')


def estimate_noise_sigma(s21_1d, method='adaptive',
                            baseline_fraction=0.15,
                            amp_threshold=0.99,
                            min_points=10):
    """
    측정 잡음의 표준편차를 데이터에서 직접 추정.

    method='fixed'    : 양쪽 끝 baseline_fraction 비율을 고정적으로 사용.
                         구현이 단순하지만, 공진 딥의 폭이 달라지는 상황
                         (예: 디튜닝이 큰 slice)에서는 비효율적이거나 위험함.
    method='adaptive'  : |S21| > amp_threshold인 지점만 자동으로 baseline
                         선택. 딥 폭이 slice마다 달라져도 스스로 적응함.
    """
    if method == 'adaptive':
        mag = np.abs(s21_1d)
        baseline_mask = mag > amp_threshold
        if np.sum(baseline_mask) < min_points:
            n_pts = len(s21_1d)
            n_edge = max(int(n_pts * baseline_fraction), 5)
            baseline_region = np.concatenate([s21_1d[:n_edge], s21_1d[-n_edge:]])
        else:
            baseline_region = s21_1d[baseline_mask]
    else:
        n_pts = len(s21_1d)
        n_edge = max(int(n_pts * baseline_fraction), 5)
        baseline_region = np.concatenate([s21_1d[:n_edge], s21_1d[-n_edge:]])

    return np.mean([np.std(np.real(baseline_region)), np.std(np.imag(baseline_region))])


def find_stability_threshold_by_crossing(x_values, y_uncertainty, threshold_line):
    """
    [권장 방식 - "방식 B"] 연속값을 x 순서로 정렬해, y_uncertainty가
    처음으로 threshold_line을 넘는 지점을 찾음. 이산 라벨을 경유하지 않고
    원본 연속값에서 딱 한 번만 자르므로, 이산 라벨의 극값을 쓰는 방식보다
    이상치(outlier)에 훨씬 덜 민감함.
    """
    sort_idx = np.argsort(x_values)
    x_sorted = x_values[sort_idx]
    y_sorted = y_uncertainty[sort_idx]

    over = y_sorted > threshold_line
    if not np.any(over):
        return np.nan
    return x_sorted[np.argmax(over)]
        # np.argmax(불리언배열): True가 처음 나오는 위치를 찾는 관용적 트릭


def find_stability_threshold_by_inflection(x_values, y_uncertainty,
                                              smoothing_window=7,
                                              slope_multiplier=5):
    """
    [권장 방식 - "방식 C"] 사람이 정한 절대적 문턱값(threshold_line) 없이,
    y_uncertainty가 x에 대해 "완만 -> 급격히 증가"로 바뀌는 변곡점을
    데이터 자체의 기울기 변화로 자동 탐지.

    주의: x_values에 중복값이 있으면(예: |Δ|처럼 절댓값을 취해 서로 다른
    조건이 같은 x를 갖게 되는 경우) 기울기 계산(dx로 나누기)에서
    0-나누기 에러가 남. 이 함수는 np.unique로 중복을 평균 병합해
    이 문제를 방지함.
    """
    sort_idx = np.argsort(x_values)
    x_sorted = x_values[sort_idx]
    y_sorted = y_uncertainty[sort_idx]

    x_unique, inverse_idx = np.unique(x_sorted, return_inverse=True)
        # np.unique: 배열에서 중복을 제거한 "유일값 목록"을 반환.
        # return_inverse=True를 주면, 원래 배열의 각 원소가 유일값 목록의
        # 몇 번째 위치에 해당하는지 나타내는 "역인덱스" 배열도 함께 반환.
        # (필요한 이유: x_values에 정확히 같은 값이 여러 개 있으면, 아래
        #  np.gradient가 x간 간격(dx)으로 나눗셈을 하다가 dx=0으로
        #  "0으로 나누기" 에러가 남. 예를 들어 |Δ|처럼 절댓값을 취하는
        #  물리량은 서로 다른 두 조건(예: Φ와 -Φ)이 같은 값을 가질 수 있음.)
    y_unique = np.array([y_sorted[inverse_idx == i].mean() for i in range(len(x_unique))])
        # 리스트 컴프리헨션(list comprehension): "각 유일값 i에 대해 ~를 계산해서
        # 리스트로 만들어라"는 파이썬의 축약 문법.
        # inverse_idx == i : i번째 유일값에 해당하는 모든 원소 위치를 찾는
        # 불리언 마스크(참/거짓 배열). 그 위치들의 y값을 평균 -> 중복된 x값에
        # 대한 "대표 y값"을 만듦 (물리적으로 같은 조건에서 두 번 잰 것과
        # 마찬가지이므로 평균내는 것이 자연스러움).

    y_smooth = moving_average(y_unique, smoothing_window)
    slope = np.gradient(y_smooth, x_unique)
        # np.gradient: 배열의 국소 기울기(수치 미분)를 계산하는 함수.
        # 두 번째 인자(x_unique)를 주면, 그 x값 간격을 기준으로 dy/dx를 계산함
        # (x 간격이 일정하지 않아도 정확한 기울기를 구할 수 있음).

    n_reference = max(len(slope) // 5, 5)   # 앞쪽 20%를 "평탄 구간 기준"으로 사용
        # // : 정수 나눗셈(나머지를 버리고 몫만 취함). 배열 길이의 20%를 계산.
        # max(..., 5): 배열이 너무 짧아 20%가 5보다 작아지는 경우를 대비한 하한선.
    reference_slope = np.mean(np.abs(slope[:n_reference]))
        # slope[:n_reference]: 배열의 앞부분 n_reference개만 슬라이싱
        # (|Δ|가 작은, 즉 데이터 신뢰도가 높을 것으로 기대되는 초반 구간)

    steep_region = np.abs(slope) > slope_multiplier * reference_slope
    if not np.any(steep_region):
        # np.any: 배열 안에 True가 하나라도 있는지 확인
        return np.nan
    return x_unique[np.argmax(steep_region)]
        # np.argmax(불리언배열): True가 처음 나오는 위치를 찾는 관용적 트릭
        # (True=1, False=0으로 취급되어, 가장 큰 값=1이 처음 나오는 인덱스를 반환)


def find_stability_threshold_by_discrete_label(x_values, status_labels, ok_label='ok'):
    """
    [비권장 - "방식 A", 비교/참고용으로만 남겨둠]
    이산 라벨(status)에서 ok로 분류된 것들 중 x의 최댓값을 찾음.
    자르기가 2회(연속값->라벨, 라벨->극값) 일어나 정보 손실이 크고,
    이상치 하나에 결과가 크게 흔들릴 수 있음. find_stability_threshold_by_crossing
    또는 by_inflection을 대신 사용하는 것을 권장.
    """
    ok_mask = status_labels == ok_label
    if not np.any(ok_mask):
        return np.nan
    return np.max(x_values[ok_mask])
