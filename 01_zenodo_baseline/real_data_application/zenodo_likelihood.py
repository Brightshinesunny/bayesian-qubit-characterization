"""
zenodo_likelihood.py  (likelihood.py를 Zenodo 실측 데이터 워크플로우 전용으로 복제한 버전 - 이름 충돌/캐시 문제 방지용)
=================================================================
Prior(사전분포), Likelihood(우도), Posterior(사후분포) 함수들을 모아둔
모듈. "구조를 바꾸는 연습"의 핵심 확장 지점 중 하나입니다.

[템플릿 설계 원칙]
지금까지는 가우시안 우도(residual^2 기반, chi-square)만 썼습니다.
하지만 outlier가 섞인 데이터(mock_data.py의 add_outliers 참고)에는
가우시안 우도가 잘 안 맞습니다 - 큰 잔차 하나가 chi2를 지배해버려서
파라미터 추정이 그 outlier 쪽으로 끌려갈 수 있기 때문입니다.

이 모듈은 우도 함수를 "교체 가능한 부품"으로 설계해서, 같은 MCMC
파이프라인(mcmc_pipeline.py)에 gaussian_log_likelihood 대신
robust_log_likelihood를 끼워넣기만 하면 되도록 만들었습니다.
"""

import numpy as np


def make_uniform_log_prior(bounds):
    """
    균일(flat/uniform) prior를 만드는 "팩토리 함수"(다른 함수를 만들어
    반환하는 함수). bounds는 {'파라미터이름': (min, max), ...} 형태의 딕셔너리.

    이렇게 팩토리 패턴을 쓰면, 파라미터 이름과 개수가 바뀌어도
    (fr,g,kappa 3개든, fr,g,kappa,chi 4개든) 코드를 다시 안 짜도 됨.

    사용 예:
        log_prior = make_uniform_log_prior({
            'f_r': (4.8, 5.2), 'g': (0.001, 0.1), 'kappa': (0.001, 0.1)
        })
        log_prior([5.0, 0.04, 0.03])  # -> 0.0 (범위 안) 또는 -inf (범위 밖)
    """
    param_names = list(bounds.keys())

    def log_prior(theta):
        for name, val in zip(param_names, theta):
            # zip: 두 리스트(파라미터 이름들, 실제 값들)를 짝지어 하나씩 순회
            lo, hi = bounds[name]
            if not (lo < val < hi):
                return -np.inf
        return 0.0

    return log_prior


def gaussian_log_likelihood(theta, model_func, model_kwargs, f_grid, data_1d, sigma):
    """
    [기존 방식] 가우시안(정규분포) 잡음을 가정한 우도.
    -0.5 * chi^2 형태로, 잔차(residual)가 클수록 페널티가 "제곱으로" 커짐.

    문제점: outlier(정상 범위를 크게 벗어난 값) 하나가 residual^2으로
    인해 우도 전체를 지배해버릴 수 있음. 즉 outlier 하나 때문에 나머지
    수백 개의 정상적인 점이 주는 정보가 무시되고, 파라미터 추정이
    그 outlier에 맞춰 왜곡될 위험이 있음.

    model_func : 예측 신호를 계산하는 forward model 함수 (models.py의 것)
    model_kwargs : model_func에 넘길 고정 인자들(사전 캘리브레이션 상수 등)
    """
    model = model_func(f_grid, **model_kwargs, **_theta_to_kwargs(theta, model_kwargs))
        # **딕셔너리: 딕셔너리를 "키워드 인자로 풀어서" 함수에 넘기는 문법 (언패킹, unpacking).
        # 예: func(**{'a':1,'b':2})는 func(a=1, b=2)와 동일.
        # 여기서는 model_kwargs(고정 상수들)와 _theta_to_kwargs(...)(추정 대상
        # 파라미터들)를 각각 딕셔너리로 만들어 한 번에 model_func에 전달함.
        # 이렇게 하면 model_func의 인자 순서를 몰라도 이름으로 정확히 매칭됨.
    real_res = np.real(data_1d - model)
    imag_res = np.imag(data_1d - model)
    chi2 = np.sum((real_res / sigma) ** 2 + (imag_res / sigma) ** 2)
    return -0.5 * chi2


def robust_log_likelihood(theta, model_func, model_kwargs, f_grid, data_1d, sigma,
                            nu=4.0):
    """
    [개선안] Student's t-분포 기반 "강건한(robust)" 우도.

    핵심 아이디어: 가우시안 분포는 "꼬리"가 얇아서, 잔차가 큰 점(outlier)에
    극단적으로 큰 페널티(제곱으로 증가)를 매김. 반면 Student's t-분포는
    "꼬리"가 두꺼워서(heavy-tailed), 큰 잔차에 대한 페널티가 상대적으로
    완만하게 증가함 -> outlier가 전체 결과를 덜 지배하게 됨.

    nu(자유도, degrees of freedom): 값이 작을수록(예: 3~5) 꼬리가 더
    두꺼워져 outlier에 더 관대해지고, nu가 커질수록(예: 30 이상) 가우시안에
    가까워짐. nu -> 무한대의 극한이 정확히 가우시안 분포와 같아짐.

    로그우도 형태 (Student's t-분포의 로그확률밀도에서 유도):
        log_lik = -0.5*(nu+1) * sum( log(1 + residual^2 / (nu*sigma^2)) )
    (정규화 상수 항은 파라미터에 의존하지 않으므로 MCMC에서는 생략 가능)
    """
    model = model_func(f_grid, **model_kwargs, **_theta_to_kwargs(theta, model_kwargs))
    real_res = np.real(data_1d - model)
    imag_res = np.imag(data_1d - model)

    def t_term(res):
        # log(1 + x^2/nu) 형태: x가 커져도 chi2처럼 폭발적으로 커지지 않고
        # log 함수 특성상 완만하게 증가함 -> "강건함(robustness)"의 수학적 근거
        return np.log1p((res / sigma) ** 2 / nu)
            # np.log1p(x) = log(1+x) 를 수치적으로 더 안정적으로 계산하는 함수
            # (x가 0에 가까울 때 log(1+x)를 직접 계산하면 정밀도 손실이 생길 수 있어서
            #  전용 함수를 쓰는 것이 관례)

    log_lik = -0.5 * (nu + 1) * np.sum(t_term(real_res) + t_term(imag_res))
    return log_lik


def _theta_to_kwargs(theta, model_kwargs):
    """
    [내부 헬퍼 함수] 이름 앞의 밑줄(_)은 파이썬의 관례로, "이 모듈
    바깥에서는 직접 부르지 말고, 내부적으로만 쓰는 보조 함수"라는 뜻을
    나타냄 (강제되는 규칙은 아니고, 관례적 신호).

    theta(추정 대상 파라미터를 담은 배열, MCMC가 탐색하는 값들)를
    model_func가 이해하는 키워드 인자 형태(딕셔너리)로 바꿔주는 역할.
    이 템플릿에서는 theta의 순서를 (f_r, g, kappa)로 고정해 사용합니다.
    다른 물리 모델(파라미터 종류/개수가 다른 경우)을 쓸 때는 이 매핑
    함수를 프로젝트에 맞게 바꿔주면 됩니다.

    예: theta = [5.0, 0.04, 0.03] 이면
        반환값 = {'f_r': 5.0, 'g': 0.04, 'kappa': 0.03}
    """
    return {'f_r': theta[0], 'g': theta[1], 'kappa': theta[2]}


def make_log_probability(model_func, model_kwargs, log_prior_func,
                           likelihood_type='gaussian', **likelihood_kwargs):
    """
    log_prior + log_likelihood를 합쳐 최종 log_probability 함수를 만드는
    팩토리 함수. likelihood_type을 바꾸는 것만으로 가우시안<->robust를
    전환할 수 있음 (이게 "구조를 바꾸는" 지점을 최소한의 코드 변경으로
    가능하게 만드는 템플릿의 핵심 설계).

    사용 예:
        log_prob = make_log_probability(
            s21_anticrossing_model, {'fq_max':5.15,'EC':0.25,'tau':0.12,'flux_val':0.0},
            log_prior, likelihood_type='robust', nu=4.0
        )
    """
    if likelihood_type == 'gaussian':
        likelihood_func = gaussian_log_likelihood
    elif likelihood_type == 'robust':
        likelihood_func = robust_log_likelihood
    else:
        raise ValueError(f"알 수 없는 likelihood_type: {likelihood_type}")
            # raise: 파이썬에서 에러를 일부러 발생시키는 명령어.
            # ValueError: "값이 잘못됐다"는 뜻의 표준 예외(에러) 종류.
            # 오타나 지원 안 하는 값이 들어왔을 때, 조용히 넘어가거나 엉뚱한
            # 결과를 내는 대신 즉시 명확한 에러 메시지로 알려주기 위한 안전장치.

    def log_probability(theta, f_grid, data_1d, sigma):
        """
        [클로저(closure)] 이 안쪽 함수는 바깥 make_log_probability 함수의
        지역변수(model_func, model_kwargs, log_prior_func, likelihood_func 등)를
        기억한 채로 반환됨. 이런 구조를 "클로저"라고 부르며, 파이썬에서
        "설정이 이미 반영된 맞춤 함수"를 만들어내는 표준적인 방법.
        MCMC 샘플러(emcee)는 이 log_probability(theta, ...)만 반복 호출하면
        되므로, 내부적으로 어떤 모델/우도를 쓰는지는 몰라도 됨.
        """
        lp = log_prior_func(theta)
        if not np.isfinite(lp):
            # np.isfinite: 값이 무한대(inf)나 정의되지 않은 값(NaN)이 아닌지 확인.
            # prior 범위를 벗어나 -inf가 나왔다면, 굳이 비싼 우도 계산을
            # 하지 않고 바로 -inf를 반환해 계산을 절약함 (조기 종료, short-circuit).
            return -np.inf
        ll = likelihood_func(theta, model_func, model_kwargs, f_grid, data_1d, sigma,
                               **likelihood_kwargs)
            # **likelihood_kwargs: robust 방식일 때 nu=4.0 같은 추가 인자를
            # 이 자리에서 그대로 전달하기 위한 언패킹.
        return lp + ll
            # 베이즈 정리: posterior ∝ prior × likelihood.
            # 로그를 취하면 곱셈이 덧셈이 되므로 log(posterior) = log(prior)+log(likelihood)

    return log_probability
