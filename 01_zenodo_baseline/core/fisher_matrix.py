"""
fisher_matrix.py
=================================================================
피셔 정보 행렬(Fisher Information Matrix)을 이용한 파라미터 오차 추정.
MCMC(베이지안) 방식과 나란히 비교하기 위한 모듈입니다.

[두 방법의 관계 - 중력파 매개변수 추정과 동일한 구조]
중력파 분석에서 흔히 쓰시던 것과 정확히 같은 원리입니다:

  - 피셔 행렬 방식: 최적 파라미터 지점에서 log-likelihood를 2차
    테일러 전개(quadratic approximation)로 근사해, 그 국소적인
    "곡률(curvature)"만으로 오차를 빠르게 추정. 계산이 매우 빠르지만
    (모델의 미분만 있으면 됨), posterior가 실제로 가우시안에 가까울
    때만 정확함.

  - MCMC(베이지안) 방식: log-likelihood 전체 모양을 직접 샘플링해서
    탐색. 느리지만, posterior가 비대칭이거나 여러 파라미터끼리
    비선형적으로 얽혀 있어도(축퇴, degeneracy) 정확히 잡아냄.

  두 방법이 잘 일치하면 -> "이 지점에서는 선형 근사가 충분히 좋다"는 뜻
  두 방법이 어긋나면   -> "posterior가 비선형/비대칭이다"는 신호이며,
                        MCMC 쪽 결과를 신뢰하는 것이 일반적으로 더 안전함.

  중력파에서 "chirp mass와 mass ratio 사이의 강한 축퇴" 때문에
  피셔 행렬이 실제 posterior 모양을 과소/과대평가하는 것과 정확히
  같은 문제가, 오늘 다룬 g-kappa 축퇴에서도 나타날 수 있습니다.
"""

import numpy as np


def numerical_derivative(model_func, theta, param_index, f_grid, model_kwargs,
                            step_fraction=1e-4):
    """
    모델 함수를 특정 파라미터에 대해 수치적으로 미분(numerical/finite
    difference derivative)합니다. 해석적으로 미분식을 손으로 유도하지
    않고도, 함수를 아주 살짝 흔들어봐서(perturb) 기울기를 근사합니다.

    중앙차분법(central difference): f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    양쪽으로 살짝씩 이동한 값의 차이를 보는 방식이 한쪽만 보는 방식
    (전진차분, forward difference)보다 오차가 더 작음 (2차 정확도).

    step_fraction : 미분에 쓸 작은 변화량(h)을 파라미터 크기의 몇 %로
                    할지. 너무 작으면 부동소수점 반올림 오차에 묻히고,
                    너무 크면 진짜 미분과 달라지는(비선형성 때문에)
                    trade-off가 있어 적절한 크기를 골라야 함.
    """
    theta_plus = np.array(theta, dtype=float).copy()
    theta_minus = np.array(theta, dtype=float).copy()

    h = step_fraction * max(abs(theta[param_index]), 1e-8)
        # max(..., 1e-8): 파라미터 값이 우연히 0에 아주 가까울 때
        # h=0이 되어버리는 것을 막는 안전장치

    theta_plus[param_index] += h
    theta_minus[param_index] -= h

    kwargs_plus = _theta_to_kwargs(theta_plus)
    kwargs_minus = _theta_to_kwargs(theta_minus)

    model_plus = model_func(f_grid, **model_kwargs, **kwargs_plus)
    model_minus = model_func(f_grid, **model_kwargs, **kwargs_minus)

    return (model_plus - model_minus) / (2 * h)


def _theta_to_kwargs(theta):
    """
    likelihood.py의 _theta_to_kwargs와 동일한 매핑 규칙(f_r, g, kappa
    순서)을 씀. 이 템플릿 안에서 파라미터 순서 규칙을 통일해두면,
    여러 모듈에서 같은 theta 배열을 헷갈림 없이 재사용할 수 있음.
    """
    return {'f_r': theta[0], 'g': theta[1], 'kappa': theta[2]}


def fisher_information_matrix(model_func, theta_best, f_grid, sigma, model_kwargs,
                                 step_fraction=1e-4):
    """
    피셔 정보 행렬 F를 계산.

    F_ij = sum_k (1/sigma_k^2) * Re[ dM/dtheta_i |_k * conj(dM/dtheta_j |_k) ] * 2
    (실수부, 허수부 각각의 기여를 더하는 것이 위 식의 "*2"에 반영됨 -
     실제로는 실수부와 허수부를 따로 계산해 각각 더함)

    이 행렬의 (i,j) 성분이 클수록, 파라미터 i와 j를 "동시에 바꿨을 때
    모델 예측이 얼마나 민감하게(강하게) 변하는지"를 나타냄. 대각선
    성분(i=j)이 클수록 그 파라미터 하나만으로도 모델이 민감하게
    변한다는 뜻이고, 비대각 성분이 크면 두 파라미터가 서로 얽혀서
    (축퇴, degeneracy) 데이터만으로는 독립적으로 구분하기 어렵다는 뜻.

    theta_best : 피셔 행렬을 계산할 기준점 (보통 MCMC의 median이나
                 최적 우도 지점을 사용 - "이 근방에서 곡률이 얼마나
                 가파른가"를 보는 것이므로 기준점이 중요함)
    """
    ndim = len(theta_best)
    derivatives = [
        numerical_derivative(model_func, theta_best, i, f_grid, model_kwargs, step_fraction)
        for i in range(ndim)
    ]
        # derivatives[i] : i번째 파라미터에 대한 모델의 미분(복소수 배열, f_grid와 같은 길이)

    F = np.zeros((ndim, ndim))
    for i in range(ndim):
        for j in range(ndim):
            # 실수부와 허수부 각각의 기여를 더함 (복소수 데이터를
            # "실수 데이터 2배"로 취급하는 것과 동일한 관례 -
            # likelihood.py의 gaussian_log_likelihood와 일관된 방식)
            real_term = np.sum(np.real(derivatives[i]) * np.real(derivatives[j]) / sigma ** 2)
            imag_term = np.sum(np.imag(derivatives[i]) * np.imag(derivatives[j]) / sigma ** 2)
            F[i, j] = real_term + imag_term

    return F


def covariance_from_fisher(F):
    """
    피셔 행렬을 역행렬(inverse)해서 공분산 행렬(covariance matrix)을 얻음.
    Cramér-Rao 하한(Cramér-Rao bound)에 의해, 이 공분산 행렬의 대각
    성분들은 "어떤 비편향 추정량으로도 이보다 더 작은 분산을 얻을 수
    없는 이론적 하한"을 나타냄 - 즉 "이 실험 설계로 도달 가능한 최선의
    정밀도"를 알려주는 지표.

    주의: F가 특이행렬(singular, 역행렬이 없음)에 가까우면 (파라미터
    간 축퇴가 매우 심하면) 역행렬 계산이 수치적으로 불안정해질 수 있음.
    """
    try:
        cov = np.linalg.inv(F)
            # np.linalg.inv: 행렬의 역행렬을 계산하는 함수.
            # (F는 대칭행렬이므로 역행렬도 대칭)
    except np.linalg.LinAlgError:
        # LinAlgError: 역행렬이 존재하지 않을 때(특이행렬) 발생하는 에러.
        # 축퇴가 너무 심해 피셔 행렬로는 오차를 못 구하는 상황을 알림.
        cov = np.full_like(F, np.nan)
    return cov


def fisher_parameter_uncertainties(model_func, theta_best, f_grid, sigma, model_kwargs,
                                      step_fraction=1e-4):
    """
    편의 함수: 피셔 행렬 계산부터 1-sigma 오차 추출까지 한 번에.

    Returns
    -------
    sigma_params : 각 파라미터의 1-sigma 오차 (공분산 행렬의 대각성분의 제곱근)
    correlation_matrix : 파라미터 간 상관계수 행렬 (-1~1 범위. |값|이
        1에 가까울수록 두 파라미터가 강하게 얽혀 있다는 뜻 - 이게 바로
        corner plot에서 봤던 g-kappa 대각선 방향 상관관계를, MCMC 없이
        피셔 행렬만으로도 미리 예측할 수 있다는 의미)
    """
    F = fisher_information_matrix(model_func, theta_best, f_grid, sigma, model_kwargs, step_fraction)
    cov = covariance_from_fisher(F)

    sigma_params = np.sqrt(np.diag(cov))
        # np.diag(행렬): 정사각행렬의 대각선 성분만 뽑아 1차원 배열로 반환.
        # 공분산 행렬의 대각성분 = 각 파라미터의 분산(variance) -> 제곱근하면 표준편차.

    # 상관계수 행렬: Corr_ij = Cov_ij / (sigma_i * sigma_j)
    outer_sigma = np.outer(sigma_params, sigma_params)
        # np.outer(a,b): 두 벡터의 외적(outer product)으로 행렬을 만듦.
        # outer_sigma[i,j] = sigma_params[i] * sigma_params[j]
    correlation_matrix = cov / outer_sigma

    return sigma_params, correlation_matrix


# =========================================================
# [확장] Prior 정보를 포함한 피셔 행렬
# =========================================================
# 지금까지의 fisher_information_matrix()는 "데이터(우도)만으로 얼마나
# 정밀하게 알 수 있는가"를 계산했습니다. 하지만 실전에서는 종종
# 사전 지식(prior)이 있고, 이 사전지식도 최종 불확실성을 줄여줍니다.
# 이 절의 함수들은 그 prior의 기여를 피셔 행렬에 정식으로 합산합니다.
#
# [핵심 원리 - 베이즈 정리에서 유도]
# posterior ∝ likelihood × prior
# log(posterior) = log(likelihood) + log(prior)
# 피셔 행렬은 log-확률의 2차 미분(곡률)에서 나오는 양이므로
# (F_ij = -E[∂²log(확률)/∂θi∂θj]), 덧셈 관계가 그대로 피셔 행렬의
# 덧셈으로 이어집니다:
#
#     F_total = F_likelihood + F_prior
#
# 즉 "데이터가 주는 정보"와 "사전에 이미 알고 있던 정보"가 각각
# 독립적인 피셔 행렬로 표현되고, 둘을 단순히 더하면 최종 정밀도가
# 나옵니다. (이는 GW 분석에서 "여러 검출기의 정보를 합산할 때 각
# 검출기의 피셔 행렬을 더한다"는 것과 완전히 같은 원리이기도 합니다 -
# 서로 다른 독립적인 정보원의 피셔 행렬은 항상 더해집니다.)


def gaussian_prior_fisher_contribution(prior_sigmas):
    """
    각 파라미터에 독립적인 가우시안 prior N(mean, sigma_p^2)이 걸려
    있을 때, 그 prior가 기여하는 피셔 행렬 성분을 계산.

    유도: 가우시안 prior의 로그값은
        log p(theta) = -0.5 * ((theta-mean)/sigma_p)^2 + const
    이를 theta로 두 번 미분하면 -1/sigma_p^2 (상수, theta에 안 붙음).
    피셔 행렬 정의(F = -E[2차미분])에 따라 부호가 뒤집혀 +1/sigma_p^2.

    각 파라미터의 prior가 서로 독립적이라고 가정하므로(교차항 없음),
    결과는 대각행렬(diagonal matrix)이 됩니다 - 파라미터끼리 서로
    엮이지 않고 자기 자신에게만 정보를 더해준다는 뜻.

    prior_sigmas : 각 파라미터의 가우시안 prior 표준편차를 담은 배열.
        prior가 없는(사실상 매우 넓은/uninformative) 파라미터는
        np.inf를 넣으면 1/inf^2 = 0이 되어 자동으로 "기여 없음"이 됨.

    사용 예 (bayesian_toolkit.py의 composite prior와 같은 맥락):
        # f_r만 사전 캘리브레이션으로 5.000±0.001 GHz를 안다고 가정
        F_prior = gaussian_prior_fisher_contribution([0.001, np.inf, np.inf])
    """
    prior_sigmas = np.array(prior_sigmas, dtype=float)
    return np.diag(1.0 / prior_sigmas ** 2)
        # np.diag(1차원배열): 반대로, 1차원 배열을 대각선 성분으로 하는
        # 정사각행렬을 만드는 함수 (위에서 쓴 np.diag(행렬)과는 반대 방향 용법 -
        # numpy는 같은 함수 이름이 입력 차원에 따라 다르게 동작함).


def uniform_prior_fisher_contribution(ndim):
    """
    균일(uniform/flat) prior의 피셔 행렬 기여.

    균일분포는 범위 내부에서 log p(theta)가 상수이므로, 미분하면
    항상 0입니다 (경계 바로 위에서는 무한대로 뛰지만, 피셔 행렬은
    "국소적인 매끄러운 곡률"을 가정하는 근사이므로 경계 효과는 이
    프레임워크에서 다루지 않음 - 즉 이 근사는 파라미터가 prior 경계에서
    충분히 떨어져 있을 때만 유효함).

    즉 균일 prior는 "데이터 안에서 어떤 값도 동등하게 가능하다"는
    뜻이라 정보를 전혀 추가하지 않고, 기여는 항상 영행렬(zero matrix).
    이 함수는 명시성을 위해 존재 - "prior가 없다"를 굳이 이렇게
    표현해두면, 나중에 다른 prior 종류들과 나란히 코드에서 다루기 쉬움.
    """
    return np.zeros((ndim, ndim))


def fisher_information_matrix_with_prior(model_func, theta_best, f_grid, sigma, model_kwargs,
                                            prior_type='gaussian', prior_params=None,
                                            step_fraction=1e-4):
    """
    우도 피셔 행렬과 prior 피셔 행렬을 합산해 "posterior 전체의" 피셔
    행렬을 계산. 이렇게 구한 공분산 행렬의 대각성분이, MCMC로 얻는
    posterior의 (근사적인) 1-sigma 오차와 직접 비교 가능한 값이 됩니다
    (STEP 8의 MCMC는 prior까지 포함한 posterior에서 샘플링하므로,
     "공정한 비교"를 하려면 피셔 쪽도 prior를 반드시 포함해야 함 -
     이전 버전(prior 없는 순수 우도 피셔)은 사실 MCMC와 완전히 같은
     대상을 비교한 게 아니었다는 뜻이기도 함).

    prior_type : 'gaussian' 이면 prior_params가 [sigma_1, sigma_2, ...]
                 (각 파라미터의 prior 표준편차),
                 'uniform' 이면 prior_params는 무시하고 기여 0으로 처리.
    """
    F_likelihood = fisher_information_matrix(
        model_func, theta_best, f_grid, sigma, model_kwargs, step_fraction
    )

    ndim = len(theta_best)
    if prior_type == 'gaussian':
        if prior_params is None:
            raise ValueError("prior_type='gaussian'이면 prior_params(표준편차 리스트)가 필요합니다.")
        F_prior = gaussian_prior_fisher_contribution(prior_params)
    elif prior_type == 'uniform':
        F_prior = uniform_prior_fisher_contribution(ndim)
    else:
        raise ValueError(f"알 수 없는 prior_type: {prior_type}")

    F_total = F_likelihood + F_prior
        # 앞서 설명한 핵심 원리: 로그 posterior = 로그 우도 + 로그 prior
        # 이므로, 각각의 곡률(피셔 행렬)도 단순히 더해짐.

    cov = covariance_from_fisher(F_total)
    sigma_params = np.sqrt(np.diag(cov))
    outer_sigma = np.outer(sigma_params, sigma_params)
    correlation_matrix = cov / outer_sigma

    return sigma_params, correlation_matrix, F_total
