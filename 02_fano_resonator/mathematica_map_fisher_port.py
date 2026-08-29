"""
mathematica_map_fisher_port.py
=================================================================
사용자가 Mathematica로 직접 만든 "Prior가 적용된 MAP 4D 피셔 정밀
분석" 코드를 파이썬으로 그대로 포팅한 버전입니다.

[포팅 목적]
Mathematica(기호 미분 + 수치 적분)와 파이썬(수치 미분 + 이산합)이라는
완전히 다른 두 계산 방식이, 같은 물리 문제에 대해 같은 결론(특히
"진짜 완전 해제 축퇴가 아니라, prior가 약해서 축퇴처럼 보이는
것뿐이다")에 도달하는지 교차검증합니다. 이건 오늘 계속 해온
"독립적인 방법으로 같은 결론에 도달하면 신뢰도가 올라간다"는
원칙을, 이번엔 "언어/소프트웨어가 다른 두 구현" 사이에 적용한 것.

[원본 Mathematica 코드와의 대응 관계]
  tildeS21[f,w1,w2,g12,gamma2]  ->  tilde_s21(f, w1, w2, g12, gamma2)
  QubitPSD[f]                    ->  qubit_psd(f)
  QubitFreqIP[a,b,PSD,fmin,fmax] ->  freq_inner_product(...)
  MultiQubitFisherFixedSNR[...]  ->  fisher_matrix_fixed_snr(...)
  RunMAPPipeline[...]            ->  run_map_pipeline(...)

[물리적 배경]
tildeS21 모델은 오늘 다룬 avoided-crossing(①~④번 식)과 수학적으로
매우 비슷한 구조입니다 - 두 개의 로렌츠 항의 합으로 두 혼성 모드
(wPlus, wMinus)를 표현합니다. 다만 오늘 모델은 "1 - 로렌츠"(딥) 형태였고,
여기서는 로렌츠 항을 그대로 더한(피크형) 형태이며, gamma2 하나로
두 모드의 선폭이 같다고 가정한 단순화된 버전입니다.
"""

import numpy as np
from scipy import integrate


# =========================================================
# STEP 0. 분석자 설정값 (aa_ 접두사) - Mathematica의 true*, sigma*Prior와 대응
# =========================================================
aa_true_w1 = 5.0        # GHz, 첫 번째 모드 주파수 참값
aa_true_w2 = 5.1        # GHz, 두 번째 모드 주파수 참값
aa_true_g12 = 0.05      # GHz, 결합 세기 참값 (=50 MHz)
aa_true_gamma2 = 0.0001 # GHz, 선폭(감쇠율) 참값 (=1/T2*)

aa_f_min, aa_f_max = 4.8, 5.3   # GHz, 적분/스캔 주파수 범위
aa_target_snr = 100.0            # 목표 신호대잡음비 (피셔 행렬 스케일링에 사용)

# Prior 표준편차 (작을수록 "강한/좁은" 사전지식, 클수록 "느슨한" prior)
aa_sigma_w1_prior = 0.001      # GHz = 1 MHz
aa_sigma_w2_prior = 0.001      # GHz = 1 MHz
aa_sigma_g12_prior = 0.005     # GHz = 5 MHz
aa_sigma_gamma2_prior = 0.00001  # GHz

aa_step_fraction = 1e-6
    # 수치 미분에 쓸 상대적 스텝 크기. Mathematica는 D[...]로 "기호
    # 미분"(정확한 해석적 도함수)을 쓰지만, 파이썬에서는 수치 미분
    # (중앙차분법)을 씁니다. 두 방법이 얼마나 일치하는지 보는 것 자체가
    # 이번 교차검증의 핵심 포인트 중 하나.

print("=" * 60)
print("분석자 설정값(aa_*) 요약:")
for k, v in list(globals().items()):
    if k.startswith("aa_"):
        print(f"  {k:22s} = {v}")
print("=" * 60)


# =========================================================
# STEP 1. 물리 모델 (Mathematica tildeS21, QubitPSD 포팅)
# =========================================================
def tilde_s21(f, w1, w2, g12, gamma2):
    """
    Mathematica 원본:
      tildeS21[f_,w1_,w2_,g12_,gamma2_] :=
        wPlus = 0.5(w1+w2+Sqrt[(w1-w2)^2+4g12^2]);
        wMinus = 0.5(w1+w2-Sqrt[(w1-w2)^2+4g12^2]);
        (1/(gamma2+I*2*Pi*(f-wPlus))) + (1/(gamma2+I*2*Pi*(f-wMinus)))

    물리적으로 오늘 다룬 ②번 식(혼성 모드 wPlus/wMinus)과 동일한
    구조지만, 여기서는 진동수를 각주파수(2*pi*f)로 다루고, 두 로렌츠
    항을 "1 - ..." 없이 그대로 더한 피크형이라는 점이 다름.
    """
    w_plus = 0.5 * (w1 + w2 + np.sqrt((w1 - w2) ** 2 + 4 * g12 ** 2))
    w_minus = 0.5 * (w1 + w2 - np.sqrt((w1 - w2) ** 2 + 4 * g12 ** 2))
    return (1.0 / (gamma2 + 1j * 2 * np.pi * (f - w_plus))) + \
           (1.0 / (gamma2 + 1j * 2 * np.pi * (f - w_minus)))


def qubit_psd(f):
    """
    Mathematica 원본: QubitPSD[f_] := 1.0*^-7 + (1.0*^-4/Max[f,0.1])
    PSD(Power Spectral Density, 파워 스펙트럼 밀도): 주파수마다 잡음의
    "세기"가 다를 수 있다는 걸 표현하는 함수. 값이 클수록 그 주파수에서
    잡음이 강해서(신뢰도가 낮아서) 피셔 정보에 기여하는 가중치가 작아짐
    (아래 내적 정의에서 1/PSD로 나뉘는 것과 관련).
    """
    return 1.0e-7 + (1.0e-4 / np.maximum(f, 0.1))
        # np.maximum(f, 0.1): Mathematica의 Max[f,0.1]과 동일하게,
        # f가 0.1보다 작아지는 걸 방지 (0 근처에서 1/f가 발산하는 것을 막는 안전장치)


# =========================================================
# STEP 2. 수치 안정화 내적 + 피셔 행렬 (Mathematica QubitFreqIP,
#          MultiQubitFisherFixedSNR 포팅)
# =========================================================
def freq_inner_product(a_func, b_func, psd_func, fmin, fmax, resonance_points=None):
    """
    Mathematica 원본:
      QubitFreqIP[a_,b_,PSD_,fmin_,fmax_] :=
        Chop[Re[4*NIntegrate[(a*Conjugate[b])/PSD[f], {f,fmin,fmax}, ...]]]

    "내적(inner product)"이라는 이름의 의미: 두 신호(또는 두 신호의
    미분)가 얼마나 "겹치는지"를 잡음으로 가중해서 적분한 값. 이 값이
    피셔 행렬의 각 성분이 됩니다. 중력파 매칭필터에서 쓰는
    (h1|h2) 내적과 완전히 같은 개념/공식 구조입니다 - 신호 자체가
    아니라 "신호의 각 파라미터에 대한 미분끼리의 겹침"을 재는 것.

    resonance_points : [w1, w2]처럼, 적분 구간 안에서 피적분함수가
        뾰족하게 솟는(공진) 지점들의 리스트. scipy.integrate.quad는
        기본적으로 "이 근처에 중요한 특징이 있다"는 정보가 없으면,
        gamma2처럼 매우 좁은 선폭(폭 0.0001 GHz)의 봉우리를 놓치고
        지나갈 수 있음 (실제로 이 함수를 처음 만들었을 때
        IntegrationWarning이 발생하며 부정확한 값이 나왔던 원인).
        Mathematica의 NIntegrate는 적응형 알고리즘이 이런 상황에
        기본적으로 더 강건하지만, scipy는 points 인자로 힌트를 줘야
        비슷한 수준의 정확도가 나옴.
    """
    def integrand_real(f):
        val = a_func(f) * np.conj(b_func(f)) / psd_func(f)
        return np.real(val)

    integral, _ = integrate.quad(
        integrand_real, fmin, fmax, limit=400,
        points=resonance_points,
            # points: 적분 구간을 이 지점들 근처에서 미리 쪼개서 적분하도록
            # scipy에 알려주는 인자. 봉우리 위치를 정확히 알려주면,
            # 적응형 알고리즘이 그 근처를 훨씬 촘촘하게 재계산함.
        epsabs=1e-14, epsrel=1e-12,
            # 절대/상대 허용오차를 기본값보다 훨씬 타이트하게 지정 -
            # 극도로 좁은 봉우리를 다루려면 기본 정밀도로는 부족함.
    )
    return 4 * integral


def numerical_partial_derivative(model_func, param_names, theta, param_index,
                                    step_fraction):
    """
    Mathematica는 D[...] (기호 미분, 해석적으로 정확한 도함수)를 쓰지만,
    파이썬에서는 중앙차분법(수치 미분)으로 근사합니다. 두 방법의 결과가
    충분히 가까운지가, 이번 포팅이 "제대로 원본을 재현했는지"의 1차
    검증 포인트입니다.
    """
    theta_plus = list(theta)
    theta_minus = list(theta)
    h = step_fraction * max(abs(theta[param_index]), 1e-8)
    theta_plus[param_index] += h
    theta_minus[param_index] -= h

    def make_func(t):
        kwargs = dict(zip(param_names, t))
        return lambda f: model_func(f, **kwargs)

    f_plus = make_func(theta_plus)
    f_minus = make_func(theta_minus)
    return lambda f: (f_plus(f) - f_minus(f)) / (2 * h)
        # 함수 자체를 반환(클로저): freq_inner_product가 "주파수를 받는
        # 함수"를 기대하므로, 미분 결과도 배열이 아니라 함수 형태로
        # 만들어서 넘겨줌.


def fisher_matrix_fixed_snr(derivs, psd_func, fmin, fmax, h_func, target_snr,
                               resonance_points=None):
    """
    Mathematica 원본 MultiQubitFisherFixedSNR을 그대로 포팅.

    scalefac = (targetSNR / 현재 SNR)^2 : "지금 이 신호가 원래 갖는
    SNR"과 "우리가 원하는 목표 SNR" 사이의 비율 제곱을 피셔 행렬 전체에
    곱함. 피셔 정보는 SNR의 제곱에 비례하므로(신호가 세질수록 정보가
    빠르게 늘어남), 이렇게 하면 "이 실험이 SNR=100이었다면 오차가
    얼마였을까"를 재현 가능. (오늘 다룬 noise_sigma를 직접 정하는 대신,
    "목표 SNR"이라는 더 실험적으로 익숙한 단위로 잡음 수준을 정하는
    방식 - 중력파 매칭필터 SNR과 동일한 발상.)
    """
    current_snr = np.sqrt(freq_inner_product(h_func, h_func, psd_func, fmin, fmax,
                                                resonance_points=resonance_points))
    scalefac = (target_snr / current_snr) ** 2

    n = len(derivs)
    F = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            F[i, j] = freq_inner_product(derivs[i], derivs[j], psd_func, fmin, fmax,
                                            resonance_points=resonance_points)
            F[j, i] = F[i, j]   # 피셔 행렬은 항상 대칭이므로 위쪽만 계산하고 복사
    return scalefac * F


# =========================================================
# STEP 3. Prior가 적용된 MAP 파이프라인 (Mathematica RunMAPPipeline 포팅)
# =========================================================
def run_map_pipeline(fmin, fmax, w1, w2, g12, gamma2, target_snr,
                        sigma_priors):
    """
    Mathematica RunMAPPipeline을 그대로 포팅.
    sigma_priors : [sigma_w1, sigma_w2, sigma_g12, sigma_gamma2] 리스트.
    """
    param_names = ['w1', 'w2', 'g12', 'gamma2']
    theta = [w1, w2, g12, gamma2]

    h_func = lambda f: tilde_s21(f, w1, w2, g12, gamma2)
    resonance_points = [w1, w2]
        # 적분기에 "이 두 지점 근처에 뾰족한 공진 봉우리가 있다"고
        # 알려주는 힌트. gamma2가 매우 좁을 때 이 힌트가 없으면
        # scipy.integrate.quad가 봉우리를 놓쳐 부정확한 값을 냄.

    derivs = [
        numerical_partial_derivative(tilde_s21, param_names, theta, i, aa_step_fraction)
        for i in range(4)
    ]

    F_data = fisher_matrix_fixed_snr(derivs, qubit_psd, fmin, fmax, h_func, target_snr,
                                        resonance_points=resonance_points)

    F_prior = np.diag([1.0 / s ** 2 for s in sigma_priors])
        # Mathematica의 DiagonalMatrix[{1/sigma^2, ...}]와 동일.
        # 어제 avoided-crossing 피셔 행렬 모듈에서 만든
        # gaussian_prior_fisher_contribution과 완전히 같은 원리.

    F_total = F_data + F_prior
        # 로그 posterior = 로그 우도 + 로그 prior 이므로, 곡률(피셔
        # 행렬)도 단순히 더해짐 - 오늘 여러 번 확인한 원리.

    cov_total = np.linalg.inv(F_total)
    errs_total = np.sqrt(np.abs(np.diag(cov_total)))
        # Mathematica의 Sqrt[Abs[CovMatTotal[[i,i]]]]와 동일.
        # Abs를 씌우는 이유: 수치오차로 아주 작은 음수가 나오는 걸 방지.

    outer = np.outer(errs_total, errs_total)
    corr_total = cov_total / outer

    t2star_err = (errs_total[3] / gamma2 ** 2) * 1e-3
        # T2* = 1/gamma2 이므로, 오차 전파 공식(delta(1/x) = delta(x)/x^2)
        # 을 적용해 gamma2의 오차를 T2*의 오차로 변환.
        # 1e-3: GHz^-1 단위를 us(마이크로초) 단위로 바꾸는 스케일 조정.

    print("\n" + "=" * 55)
    print("   [파이썬 포팅판: Prior가 적용된 MAP 4D 피셔 정밀 분석]")
    print("=" * 55)
    print(f"• Target SNR         : {target_snr}")
    print(f"• w1 참값 & 오차(MAP)  : {w1} GHz  ± {errs_total[0]*1e6:.3f} kHz")
    print(f"• w2 참값 & 오차      : {w2} GHz  ± {errs_total[1]*1e6:.3f} kHz")
    print(f"• g12 결합 세기 오차   : {g12} GHz  ± {errs_total[2]*1e6:.3f} kHz")
    print(f"• T2* 디페이징 오차    : {(1/gamma2)*1e-3:.3f} us  ± {t2star_err:.6f} us")
    print("-" * 55)
    print("• MAP Parameter Correlation Matrix (상관계수 행렬, 소수점 10자리):")
    for row in corr_total:
        print("  " + "  ".join(f"{v: .10f}" for v in row))
    print("=" * 55)

    return {
        'errs': errs_total, 'corr': corr_total,
        'F_data': F_data, 'F_prior': F_prior,
    }


# =========================================================
# STEP 4. 파이프라인 실행 및 Mathematica 결과와 직접 대조
# =========================================================
sigma_priors = [aa_sigma_w1_prior, aa_sigma_w2_prior,
                aa_sigma_g12_prior, aa_sigma_gamma2_prior]

result = run_map_pipeline(
    aa_f_min, aa_f_max, aa_true_w1, aa_true_w2, aa_true_g12, aa_true_gamma2,
    aa_target_snr, sigma_priors
)

# Mathematica가 실제로 낸 결과값 (사용자가 공유한 원본 출력, 비교 기준)
mathematica_errs_kHz = [700.142, 700.142, 700.142, None]   # w1,w2,g12 (kHz), T2*는 별도 단위
mathematica_corr_01 = -0.9999997933   # w1-w2 상관계수
mathematica_corr_02 = 0.9999998429    # w1-g12 상관계수

print("\n[Mathematica 원본과 직접 대조]")
print(f"  w1 오차: 파이썬={result['errs'][0]*1e6:.3f} kHz vs Mathematica={mathematica_errs_kHz[0]} kHz")
print(f"  w2 오차: 파이썬={result['errs'][1]*1e6:.3f} kHz vs Mathematica={mathematica_errs_kHz[1]} kHz")
print(f"  g12 오차: 파이썬={result['errs'][2]*1e6:.3f} kHz vs Mathematica={mathematica_errs_kHz[2]} kHz")
print(f"  w1-w2 상관계수: 파이썬={result['corr'][0,1]:.7f} vs Mathematica={mathematica_corr_01}")
print(f"  w1-g12 상관계수: 파이썬={result['corr'][0,2]:.7f} vs Mathematica={mathematica_corr_02}")
