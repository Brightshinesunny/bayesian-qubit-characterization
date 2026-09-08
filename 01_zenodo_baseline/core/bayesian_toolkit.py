"""
bayesian_toolkit.py
=================================================================
특정 물리 모델에 종속되지 않는, 범용 베이지안 통계 도구 모음입니다.
likelihood.py가 "이 S21 모델 전용" 우도 함수들을 담았다면, 이 파일은
"어떤 물리 모델을 쓰든 재사용 가능한" 베이지안 분석의 표준 부품들을
담습니다.

포함된 기능:
  1. Prior(사전분포) 종류 모음 - uniform 외에 gaussian, log-uniform 등
  2. Posterior(사후분포) 요약 통계 - credible interval, HPD interval
  3. MCMC 수렴 진단 - Gelman-Rubin R-hat, 유효 샘플 크기
  4. 모델 비교 - AIC/BIC (evidence의 저렴한 근사치)
  5. Prior predictive check - "이 prior가 말이 되는 데이터를 만드는가" 검증

[베이지안 통계의 기본 뼈대 - 처음 보는 사람을 위한 요약]
베이즈 정리:  posterior(파라미터|데이터) ∝ likelihood(데이터|파라미터) × prior(파라미터)
  - prior: 데이터를 보기 전에 이미 갖고 있는 믿음/가정
  - likelihood: 특정 파라미터가 맞다고 가정했을 때 이 데이터가 나올 확률
  - posterior: 데이터를 본 뒤 갱신된, 파라미터에 대한 최종 믿음
MCMC(Markov Chain Monte Carlo)는 이 posterior 분포에서 직접 샘플을
뽑아내는 알고리즘 계열입니다 (분포의 수식을 몰라도, 샘플들의 히스토그램이
곧 분포 모양을 근사하게 됨).
"""

import numpy as np


# =========================================================
# 1. Prior(사전분포) 종류 모음
# =========================================================

def uniform_log_prior_term(val, lo, hi):
    """
    균일(uniform/flat) prior 한 항목의 로그값.
    "이 범위 안에서는 어느 값이나 똑같이 그럴듯하다"는 가장 단순한 가정.
    범위 밖이면 -inf(확률 0), 범위 안이면 상수(정규화 상수는 무시해도
    MCMC 결과에는 영향 없음 - posterior의 "모양"만 중요하고 전체적인
    크기 스케일은 상관없기 때문).
    """
    if lo < val < hi:
        return 0.0
    return -np.inf


def gaussian_log_prior_term(val, mean, sigma):
    """
    가우시안(정규분포) prior 한 항목의 로그값.
    "이 파라미터는 대략 mean 근처일 것이고, sigma만큼의 불확실성이
    있다"는 사전 지식이 있을 때 사용. 예: 별도의 독립적인 캘리브레이션
    실험에서 이미 fr ≈ 5.000 ± 0.001 GHz라는 걸 알고 있다면, 균일 prior
    대신 이 가우시안 prior를 걸어서 그 사전 정보를 반영할 수 있음.

    로그 가우시안 확률밀도 (정규화 상수 제외, MCMC엔 불필요):
        log p(val) = -0.5 * ((val - mean) / sigma)^2
    """
    return -0.5 * ((val - mean) / sigma) ** 2


def log_uniform_log_prior_term(val, lo, hi):
    """
    로그균일(log-uniform, 일명 "Jeffreys prior"의 근사) prior.
    val이 여러 자릿수(order of magnitude)에 걸쳐 있을 수 있는 파라미터
    (예: 결합강도가 1 kHz~1 GHz까지 어느 스케일일지 전혀 모를 때)에
    적합함. 일반 uniform prior는 "큰 값 쪽"에 훨씬 더 많은 확률을
    암묵적으로 부여하게 되는 반면, log-uniform은 "자릿수" 단위로 고르게
    분포시킴 (10~100 사이에 있을 확률과 100~1000 사이에 있을 확률이
    같다고 가정).

    주의: val이 반드시 0보다 커야 함 (log를 취하므로).
    """
    if not (lo < val < hi) or val <= 0:
        return -np.inf
    # log(val)에 대해 uniform prior를 건 것과 동일 -> val 자체의 밀도로
    # 환산하면 1/val이 곱해짐 (변수변환의 야코비안, Jacobian)
    return -np.log(val)


def make_composite_log_prior(prior_specs):
    """
    여러 종류의 prior를 한 파라미터 벡터에 섞어서 쓸 수 있게 만드는
    팩토리 함수. likelihood.py의 make_uniform_log_prior보다 유연한 버전.

    prior_specs: 리스트. 각 원소는 (종류, 파라미터들) 튜플.
        ('uniform', lo, hi)
        ('gaussian', mean, sigma)
        ('log_uniform', lo, hi)

    사용 예:
        log_prior = make_composite_log_prior([
            ('gaussian', 5.000, 0.001),   # f_r: 사전 캘리브레이션 있음
            ('uniform', 0.001, 0.1),       # g: 모르니 균일
            ('log_uniform', 1e-4, 1.0),    # kappa: 스케일을 모를 때
        ])
    """
    def log_prior(theta):
        total = 0.0
        for val, spec in zip(theta, prior_specs):
            # zip: theta의 각 값과 prior_specs의 각 규칙을 짝지어 순회
            kind = spec[0]
            if kind == 'uniform':
                total += uniform_log_prior_term(val, spec[1], spec[2])
            elif kind == 'gaussian':
                total += gaussian_log_prior_term(val, spec[1], spec[2])
            elif kind == 'log_uniform':
                total += log_uniform_log_prior_term(val, spec[1], spec[2])
            else:
                raise ValueError(f"알 수 없는 prior 종류: {kind}")
            if not np.isfinite(total):
                # 이미 -inf가 됐으면 나머지 파라미터는 계산할 필요 없이
                # 즉시 반환 (조기 종료로 계산 절약)
                return -np.inf
        return total
    return log_prior


# =========================================================
# 2. Posterior(사후분포) 요약 통계
# =========================================================

def credible_interval(samples, level=0.68):
    """
    등꼬리(equal-tailed) 신뢰구간(credible interval)을 계산.
    "posterior 확률질량의 level(예:68%)이 이 구간 안에 있다"는 의미.
    (참고: 빈도주의 통계의 "신뢰구간(confidence interval)"과 이름은
    비슷하지만 해석이 다름 - credible interval은 "파라미터가 이 구간
    안에 있을 확률이 68%"라고 직접 말할 수 있는 베이지안적 해석이 가능함.)

    level=0.68이면 16th~84th 백분위수 (가우시안의 ±1시그마와 유사),
    level=0.95면 2.5th~97.5th 백분위수를 사용.

    Returns: (하한, 중앙값, 상한) 튜플
    """
    tail = (1 - level) / 2 * 100   # 양쪽 꼬리에 남길 확률을 백분율로 변환
    lo, med, hi = np.percentile(samples, [tail, 50, 100 - tail])
    return lo, med, hi


def highest_posterior_density_interval(samples, level=0.68, n_bins=1000):
    """
    HPD(Highest Posterior Density) 구간을 계산.
    등꼬리 구간과 다른 점: posterior가 비대칭이거나 다봉(multi-modal,
    봉우리가 여러 개)일 때, "확률밀도가 가장 높은 지점들을 우선적으로
    포함하는" 구간을 찾음. 즉 폭이 최소가 되는 구간.

    구현 방법: 샘플을 정렬한 뒤, level 비율만큼의 샘플을 포함하는
    모든 "연속된 구간 후보"들 중 폭이 가장 좁은 것을 선택.
    (히스토그램 기반의 근사적 방법 - 완전히 엄밀한 해석적 해는 아니지만
    실전에서 널리 쓰이는 실용적 근사)
    """
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    n_included = int(np.ceil(level * n))
        # np.ceil: 올림(예: 3.2 -> 4). 최소한 level 비율만큼은 포함되도록
        # 넉넉하게 올림 처리.

    # 가능한 모든 "연속 구간"의 폭을 계산해서 가장 좁은 것을 찾음
    interval_widths = sorted_samples[n_included:] - sorted_samples[:n - n_included]
        # 예: n_included=680, n=1000이면
        # sorted_samples[680:] - sorted_samples[:320]
        # 이는 "320번째부터 시작해 680개를 포함하는 구간의 폭"들을
        # 한꺼번에 벡터 연산으로 계산하는 것 (반복문보다 훨씬 빠름)
    min_idx = np.argmin(interval_widths)
        # 가장 좁은 폭을 갖는 구간의 시작 위치

    return sorted_samples[min_idx], sorted_samples[min_idx + n_included]


# =========================================================
# 3. MCMC 수렴 진단
# =========================================================

def gelman_rubin_rhat(chains):
    """
    Gelman-Rubin R-hat(R̂) 통계량을 계산. 여러 독립적인 체인(예: emcee의
    여러 walker, 또는 서로 다른 초기값에서 시작한 여러 번의 MCMC 실행)이
    "같은 posterior 분포로 수렴했는지"를 판단하는 표준적 진단 지표.

    원리: 체인 "안"에서의 분산(within-chain variance)과 체인들 "사이"의
    분산(between-chain variance)을 비교. 만약 체인들이 서로 다른 곳을
    맴돌고 있다면(수렴 안 됨) between-chain 분산이 크고, R-hat이 1보다
    많이 커짐. 잘 수렴했다면 R-hat이 1에 매우 가까워짐 (관례적으로
    R-hat < 1.01 또는 1.05를 "수렴했다"는 기준으로 흔히 사용).

    chains : (n_chains, n_steps) 형태의 2D 배열. 하나의 파라미터에 대해
             여러 체인의 샘플들을 담고 있어야 함.
    """
    n_chains, n_steps = chains.shape

    chain_means = np.mean(chains, axis=1)   # 각 체인의 평균 (n_chains,)
    overall_mean = np.mean(chain_means)      # 전체 평균

    # Between-chain variance: 체인 평균들이 서로 얼마나 흩어져 있는지
    B = n_steps / (n_chains - 1) * np.sum((chain_means - overall_mean) ** 2)

    # Within-chain variance: 각 체인 내부의 분산을 평균낸 것
    chain_vars = np.var(chains, axis=1, ddof=1)
        # ddof=1: 표본분산(sample variance) 계산 시 n-1로 나누는 보정
        # (ddof: delta degrees of freedom, 자유도 보정값)
    W = np.mean(chain_vars)

    # 전체 분산의 추정치 (두 분산을 적절히 결합)
    var_hat = (n_steps - 1) / n_steps * W + B / n_steps

    rhat = np.sqrt(var_hat / W)
    return rhat


def effective_sample_size_approx(autocorr_time, n_samples):
    """
    유효 샘플 크기(Effective Sample Size, ESS)의 간단한 근사치.
    MCMC 샘플들은 연속된 것끼리 서로 닮아있어(자기상관), "진짜로
    독립적인 정보의 양"은 전체 샘플 개수보다 훨씬 적음. ESS는
    "이 샘플들이 실질적으로 몇 개의 독립적인 샘플과 맞먹는가"를 나타냄.

    근사식: ESS ≈ n_samples / autocorr_time
    (emcee의 get_autocorr_time()이 반환하는 자기상관 시간을 그대로 활용)

    실전 기준: ESS가 최소 수백 이상이어야 posterior 요약 통계(중앙값,
    신뢰구간 등)가 안정적으로 신뢰할 만하다고 봄.
    """
    return n_samples / autocorr_time


# =========================================================
# 4. 모델 비교 (evidence의 저렴한 근사치)
# =========================================================

def aic(log_likelihood_max, n_params):
    """
    AIC(Akaike Information Criterion, 아카이케 정보량 기준).
    여러 모델 중 어느 것이 데이터를 더 잘 설명하는지 비교할 때 씀.
    "우도가 높을수록 좋지만, 파라미터가 많을수록 페널티를 준다"는
    원리 (파라미터가 많으면 오버피팅으로 우도가 인위적으로 좋아지는
    경향을 보정하기 위함).

    AIC = 2*n_params - 2*log_likelihood_max
    값이 작을수록 더 좋은 모델로 평가됨.

    주의: AIC/BIC는 완전한 베이지안 모델 비교(evidence, 베이즈 인자)의
    "저렴하고 근사적인 대체재"입니다. 엄밀한 모델 비교를 하려면 nested
    sampling(dynesty 등)으로 evidence 자체를 계산하는 것이 정석이지만,
    계산 비용이 훨씬 크므로 AIC/BIC로 먼저 대략적인 감을 잡는 것도
    실전에서 흔히 쓰는 방법입니다.
    """
    return 2 * n_params - 2 * log_likelihood_max


def bic(log_likelihood_max, n_params, n_data):
    """
    BIC(Bayesian Information Criterion, 베이지안 정보량 기준).
    AIC와 비슷하지만, 데이터 개수(n_data)가 많을수록 파라미터 개수에
    대한 페널티를 더 강하게 매김 (즉 더 단순한 모델을 선호하는 경향이 강함).

    BIC = n_params * log(n_data) - 2*log_likelihood_max
    값이 작을수록 더 좋은 모델. 두 모델의 BIC 차이가 클수록(예: >10)
    한쪽 모델이 뚜렷하게 더 낫다는 증거로 흔히 해석됨 (경험적 기준).
    """
    return n_params * np.log(n_data) - 2 * log_likelihood_max


# =========================================================
# 5. Prior predictive check (사전예측 점검)
# =========================================================

def prior_predictive_check(prior_sampler, model_func, model_kwargs, f_grid, n_draws=50):
    """
    "내가 정한 prior가 물리적으로 말이 되는 데이터를 만들어내는가"를
    확인하는 진단. 실제 데이터를 보기 전에, prior에서 파라미터를 여러 번
    무작위로 뽑아 모델에 넣어보고, 그 결과 신호들이 "그럴듯한 범위" 안에
    있는지 눈으로 확인하는 절차.

    예: prior로 g의 범위를 (0.001, 100) GHz로 너무 넓게 잡으면, 이
    prior에서 뽑은 신호들 중 상당수가 물리적으로 말이 안 되는(예: 봉우리가
    스캔 범위를 완전히 벗어나는) 모양이 나올 수 있음 - 이러면 prior
    범위를 좁혀야 한다는 신호.

    prior_sampler : ()를 인자로 받아 파라미터 하나(theta)를 무작위로
        뽑아 반환하는 함수 (분석자가 prior 분포에 맞게 직접 구현)
    n_draws : 몇 번 뽑아서 확인해볼지

    Returns: (n_draws, len(f_grid)) 형태의 복소수 배열들의 리스트
    """
    draws = []
    for _ in range(n_draws):
        theta = prior_sampler()
        signal = model_func(f_grid, **model_kwargs,
                              f_r=theta[0], g=theta[1], kappa=theta[2])
        draws.append(signal)
    return draws
