"""
fano_fisher_matrix.py
=================================================================
피셔 정보 행렬 기반 파라미터 오차 추정. Fano 데이터 파이프라인
(fano_mcmc_fit.py)에서, MCMC posterior와 나란히 비교하기
위한 모듈입니다.

[fisher_matrix.py와의 차이 - 왜 그냥 복사가 아니라 새로 설계했는가]
기존 fisher_matrix.py는 파라미터가 항상 (f_r, g, kappa) 3개라고
가정하고 _theta_to_kwargs를 하드코딩했습니다. 이번 Zenodo 모델은
파라미터가 6개(f0, ke1, ke2, ki, A0, phi)이고 이름도 다르며, 게다가
tau처럼 "고정된 채로 모델에 같이 넘겨야 하는 추가 인자"도 있습니다.
그래서 이 파일은 "파라미터 이름 목록"과 "고정 인자 딕셔너리"를 함수
인자로 직접 받는 범용 설계로 만들었습니다 - 앞으로 또 다른 물리계를
다루게 되어도 이 파일을 그대로 재사용할 수 있습니다.

[원리 요약 - fisher_matrix.py와 동일]
피셔 행렬은 log-likelihood를 최적점 근방에서 2차함수(가우시안)로
근사했을 때의 "곡률"을 나타냅니다. 이 곡률의 역행렬(공분산 행렬)의
대각성분이 각 파라미터의 1-sigma 오차 추정치가 됩니다. MCMC가 직접
샘플링해서 얻는 posterior와, 피셔 행렬이 국소 근사로 얻는 오차가
서로 얼마나 일치하는지 보면 "이 지점에서 posterior가 얼마나
가우시안에 가까운지(선형 근사가 타당한지)"를 알 수 있습니다.
"""

import numpy as np


def numerical_derivative(model_func, theta, param_index, param_names,
                            f_grid, fixed_kwargs, step_fraction=1e-4):
    """
    모델 함수를 특정 파라미터에 대해 수치적으로 미분(중앙차분법).

    theta : 현재 파라미터 값들의 배열 (param_names와 순서가 대응)
    param_names : theta의 각 원소가 어떤 이름의 키워드 인자인지 알려주는
        리스트. 예: ['f0','ke1','ke2','ki','A0','phi']
    fixed_kwargs : 미분 대상이 아니라 항상 고정해서 모델에 넘길 인자들.
        예: {'tau': 62.6379} - 이런 식으로 "추정 안 하는 파라미터"를
        분리해두면, 같은 함수로 "이번엔 tau도 추정 대상에 넣고 싶다"는
        상황에도 param_names에 'tau'만 추가하면 바로 대응 가능함.
    """
    theta_plus = np.array(theta, dtype=float).copy()
    theta_minus = np.array(theta, dtype=float).copy()

    h = step_fraction * max(abs(theta[param_index]), 1e-8)
        # max(...): 파라미터 값이 우연히 0에 아주 가까울 때 h=0이
        # 되어버리는 것을 막는 안전장치. (파이썬 기본 max 사용 -
        # 이 시점에는 numpy 전체 import로 덮어써질 위험이 없는
        # 독립적인 스크립트/함수이므로 안전)

    theta_plus[param_index] += h
    theta_minus[param_index] -= h

    kwargs_plus = dict(zip(param_names, theta_plus))
        # dict(zip(이름들, 값들)): 두 리스트를 짝지어 {이름: 값} 딕셔너리로
        # 만드는 파이썬 관용구. param_names=['f0','ke1',...],
        # theta_plus=[10.47, 0.003,...] 라면
        # {'f0':10.47, 'ke1':0.003, ...} 가 만들어짐.
    kwargs_minus = dict(zip(param_names, theta_minus))

    model_plus = model_func(f_grid, **kwargs_plus, **fixed_kwargs)
    model_minus = model_func(f_grid, **kwargs_minus, **fixed_kwargs)

    return (model_plus - model_minus) / (2 * h)


def fisher_information_matrix(model_func, theta_best, param_names, f_grid,
                                 sigma, fixed_kwargs, step_fraction=1e-4):
    """
    피셔 정보 행렬 F를 계산 (fisher_matrix.py와 동일한 수식,
    파라미터를 이름 목록으로 범용화한 버전).

    F_ij = sum_k (1/sigma_k^2) * Re[dM/dtheta_i * conj(dM/dtheta_j)] * 2
    """
    ndim = len(theta_best)
    derivatives = [
        numerical_derivative(model_func, theta_best, i, param_names,
                               f_grid, fixed_kwargs, step_fraction)
        for i in range(ndim)
    ]

    F = np.zeros((ndim, ndim))
    for i in range(ndim):
        for j in range(ndim):
            real_term = np.sum(np.real(derivatives[i]) * np.real(derivatives[j]) / sigma ** 2)
            imag_term = np.sum(np.imag(derivatives[i]) * np.imag(derivatives[j]) / sigma ** 2)
            F[i, j] = real_term + imag_term

    return F


def covariance_from_fisher(F):
    """피셔 행렬의 역행렬(공분산 행렬)을 계산. fisher_matrix.py와 동일."""
    try:
        cov = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        cov = np.full_like(F, np.nan)
    return cov


def fisher_parameter_uncertainties(model_func, theta_best, param_names, f_grid,
                                      sigma, fixed_kwargs, step_fraction=1e-4):
    """
    편의 함수: 피셔 행렬 계산부터 1-sigma 오차/상관계수 추출까지 한 번에.
    """
    F = fisher_information_matrix(model_func, theta_best, param_names,
                                    f_grid, sigma, fixed_kwargs, step_fraction)
    cov = covariance_from_fisher(F)

    sigma_params = np.sqrt(np.diag(cov))
    outer_sigma = np.outer(sigma_params, sigma_params)
    correlation_matrix = cov / outer_sigma

    return sigma_params, correlation_matrix


if __name__ == "__main__":
    print("fano_fisher_matrix.py 자가진단")
    print("-" * 55)
    expected = ['numerical_derivative', 'fisher_information_matrix',
                'covariance_from_fisher', 'fisher_parameter_uncertainties']
    for name in expected:
        print(f"  {name:32s} : {'OK' if name in dir() else '누락!!'}")
