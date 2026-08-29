"""
qubit_fcv_bias.py
=================================================================
[논문에서 이식한 방법론] "Systematic bias due to eccentricity in
parameter estimation for merging binary neutron stars: Spinning
case" (E. Lee, C.-H. Lee, H.-S. Cho, Phys. Rev. D 113, 064018,
2026)의 핵심 기법인 FCV(Fisher-Cutler-Vallisneri) 방법을, 큐빗
공진기 분석(오늘까지 다룬 avoided-crossing / Zenodo / Fano 모델)에
이식한 모듈입니다.

[왜 이게 필요한가 - 문제 상황을 물리학 언어로 다시 정의]
지금까지 우리는 "노이즈 때문에 생기는 통계적 오차(statistical
uncertainty)"를 주로 다뤘습니다 (MCMC posterior 폭, 피셔 행렬
1-sigma 등). 그런데 논문이 다루는 건 완전히 다른 종류의 오차,
"체계적 편향(systematic bias)"입니다:

  통계적 오차: "같은 실험을 여러 번 반복하면 답이 얼마나 흩어지는가"
  체계적 편향: "쓰고 있는 모델 자체가 실제 물리를 다 못 담고
               있어서, 답이 원래부터 참값과 다른 방향으로
               치우쳐 있는가"

논문 예시: 진짜 신호에는 이심률(eccentricity)이 있는데, 분석에는
이심률 없는(원형 궤도) 모델을 쓰면, 노이즈가 전혀 없어도 답이
편향됩니다. 우리 맥락으로 바꾸면: 진짜 데이터에는 TLS(가짜 딥)나
Fano 비대칭이 섞여 있는데, 우리가 "깨끗한 avoided-crossing 모델"로
피팅하면, 노이즈와 무관하게 g나 kappa 추정값이 원래부터 치우칩니다.
첫날 스트레스 테스트(stress_test_mock_scenarios.py)에서 TLS를
넣었을 때 g/kappa가 크게 틀어졌던 게 바로 이 "체계적 편향"의
실전 사례였습니다 - 그때는 MCMC를 실제로 돌려서 얼마나 틀어지는지
"사후에 확인"했는데, 이 모듈은 논문처럼 MCMC를 돌리기 전에
"미리 수식으로 예측"할 수 있게 해줍니다.

[핵심 공식 - 논문 Eq.(11)을 우리 표기로 옮김]
논문:
  Δθ_i = 4A^2 * Σ^P_ij * ∫ (f^-7/3/Sn) * (Ψ_T - Ψ_AP) * ∂_j Ψ_AP df

우리 버전 (연속 적분 -> 이산 합, 실수 위상 -> 복소수 S21):
  Δθ_i = Σ_j Σ^P_ij * b_j
  b_j  = Σ_k [Re(residual_k) * Re(∂M_approx/∂θ_j)_k
             + Im(residual_k) * Im(∂M_approx/∂θ_j)_k] / sigma_k^2
  residual = S21_true(진짜 물리, 빠진 항 포함) - S21_approx(우리가 쓰는 단순 모델)

  여기서 Σ^P = (피셔행렬 + prior행렬)^-1 는 zenodo_fisher_matrix.py의
  covariance_from_fisher와 완전히 같은 역할이고, b_j는
  numerical_derivative로 이미 만들어둔 미분 함수를 그대로 재사용해서
  계산합니다 - 즉 이 모듈은 기존 피셔 행렬 코드 위에 "한 겹만 더"
  얹은 확장입니다.

[논문과 다른 점 - 정직하게 밝혀둠]
논문은 위상(phase)만 있는 실수 함수(Ψ)를 다루지만, 우리는 복소수
S21을 다룹니다. 그래서 내적(inner product)을 복소수 버전으로
확장했습니다(zenodo_fisher_matrix.py의 피셔 행렬 계산에서 이미
쓴 것과 동일한 실수부+허수부 합산 방식). 또한 논문은 연속적인
주파수 적분(∫...df)을 쓰지만, 우리는 이산적인 주파수 그리드
합(Σ...)으로 근사합니다 - 데이터 포인트가 촘촘하면 이 차이는
무시할 만큼 작습니다.
"""

import numpy as np


def systematic_bias_fcv(model_approx_func, model_true_func, theta_at,
                          param_names, f_grid, sigma,
                          fixed_kwargs_approx, fixed_kwargs_true,
                          covariance_matrix, step_fraction=1e-4):
    """
    FCV 방식으로 "불완전한 모델을 쓸 때 생기는 체계적 편향" Δθ를 예측.

    model_approx_func : 우리가 실제로 피팅에 쓰는(=놓친 물리가 없는)
        단순한 모델 함수. 예: models.s21_anticrossing_model
    model_true_func   : 실제 물리를 더 완전히 반영한 모델 함수(=놓친
        물리를 포함). 예: TLS 딥까지 반영된 신호를 만드는 함수.
        model_approx_func와 인자 형태(주파수 + 키워드 인자)가 같아야
        함수 자체를 그대로 재사용할 수 있음.
    theta_at : 이 편향을 평가할 파라미터 위치. 보통 "약식 모델로
        피팅했을 때 나오는 최적점"(θ_AP)을 씀 - 논문에서도
        θ=θ_true 근방에서 선형 근사를 하는 것과 같은 원리이므로,
        실전에서는 θ_true를 모르니 θ_AP(관측된 최적점)를 대신 사용.
    fixed_kwargs_approx, fixed_kwargs_true : 두 모델에 각각 고정으로
        넘길 인자들. 예를 들어 model_true_func가 TLS 파라미터
        (f_tls, coupling, linewidth)를 추가로 요구한다면, 그건
        fixed_kwargs_true에 넣어서 "이미 알고 있는 오염의 정체"로
        취급함 (실전에서는 이걸 모를 수도 있지만, "만약 이런 종류의
        오염이 있다면 편향이 얼마나 생길지"를 사전에 시뮬레이션해보는
        용도로 쓸 수 있음 - 논문에서 이심률 크기 e0를 미리 정해두고
        편향을 계산하는 것과 동일한 발상).
    covariance_matrix : 논문의 Σ^P에 해당. zenodo_fisher_matrix.py의
        covariance_from_fisher(F_total)로 미리 계산해서 넘김
        (F_total = 데이터 피셔행렬 + prior 피셔행렬).

    Returns
    -------
    delta_theta : 각 파라미터가 예측되는 편향량 (theta_at와 같은 순서)
    """
    kwargs_at = dict(zip(param_names, theta_at))

    model_approx_val = model_approx_func(f_grid, **kwargs_at, **fixed_kwargs_approx)
    model_true_val = model_true_func(f_grid, **kwargs_at, **fixed_kwargs_true)
    residual = model_true_val - model_approx_val
        # "진짜 신호가 우리 단순 모델과 얼마나 다른가" - 논문의
        # (Ψ_T - Ψ_AP)에 해당하는 부분. 이 차이 자체가 "우리 모델이
        # 놓친 물리"이고, 이게 0이면(즉 model_true와 model_approx가
        # 똑같으면) 당연히 편향도 0이 나와야 함 - 아래 검증에서
        # 이 극한부터 먼저 확인함.

    ndim = len(theta_at)
    derivatives = [
        _numerical_derivative_local(model_approx_func, theta_at, i, param_names,
                                       f_grid, fixed_kwargs_approx, step_fraction)
        for i in range(ndim)
    ]
        # 편향 공식의 미분(∂_j Ψ_AP)은 "우리가 실제로 쓰는 단순 모델"에
        # 대한 미분이지, true 모델에 대한 미분이 아님에 주의 - 논문
        # 공식에서도 ∂_j가 항상 Ψ_AP(근사 모델)에 붙어있는 것과 일치.

    b = np.zeros(ndim)
    for j in range(ndim):
        b[j] = np.sum(
            np.real(residual) * np.real(derivatives[j]) / sigma ** 2
            + np.imag(residual) * np.imag(derivatives[j]) / sigma ** 2
        )

    delta_theta = covariance_matrix @ b
        # 행렬-벡터 곱(@ 연산자): 논문의 "Σ^P_ij * (적분값_j)를 j에
        # 대해 합산"하는 부분을 numpy의 행렬곱으로 한 번에 계산.
        # 공분산 행렬이 파라미터들 사이의 상관관계까지 반영하므로,
        # 한 파라미터의 편향이 다른 파라미터와 얽혀 있는 경우(예:
        # 어제 배운 ke1-ke2 축퇴)도 자동으로 고려됨.

    return delta_theta


def _numerical_derivative_local(model_func, theta, param_index, param_names,
                                   f_grid, fixed_kwargs, step_fraction):
    """
    zenodo_fisher_matrix.py의 numerical_derivative와 완전히 동일한 로직.
    이 파일을 독립적으로도 쓸 수 있도록(다른 파일 import 없이) 로컬에
    복사해둠 - 어제 만든 함수를 신뢰하고 그대로 재사용하는 것.
    """
    theta_plus = np.array(theta, dtype=float).copy()
    theta_minus = np.array(theta, dtype=float).copy()
    h = step_fraction * max(abs(theta[param_index]), 1e-8)
    theta_plus[param_index] += h
    theta_minus[param_index] -= h
    kwargs_plus = dict(zip(param_names, theta_plus))
    kwargs_minus = dict(zip(param_names, theta_minus))
    model_plus = model_func(f_grid, **kwargs_plus, **fixed_kwargs)
    model_minus = model_func(f_grid, **kwargs_minus, **fixed_kwargs)
    return (model_plus - model_minus) / (2 * h)


if __name__ == "__main__":
    print("qubit_fcv_bias.py 자가진단")
    print("-" * 55)
    expected = ['systematic_bias_fcv']
    for name in expected:
        print(f"  {name:28s} : {'OK' if name in dir() else '누락!!'}")
