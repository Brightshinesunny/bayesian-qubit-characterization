"""
fano_models.py
=================================================================
원저자 circuit.py (Rieger & Günzler et al., KIT / Sebastian Probst의
resonator_tools 기반)의 대수적 circle fit 을 이식한 모듈.

핵심: 복소 S21 데이터는 복소평면에서 원을 그린다. 그 원의 기하학
(중심, 반지름, 회전각)만으로 물리 파라미터(fr, Ql, Qc, Qi, Fano
위상 phi)를 반복 탐색 없이 대수적으로 구할 수 있다.

MCMC 와의 차이:
  MCMC    - 파라미터 공간을 확률적으로 탐색. 축퇴가 있으면 수렴이 어렵다.
  circle  - "데이터가 원을 그린다"는 기하학적 사실을 이용해 행렬 연산으로
            직접 푼다. 대신 그 전제가 성립하는 경우에만 쓸 수 있다.
  (원 중심의 특성방정식 근, delay 미세보정 등 국소적인 짧은 반복은
   여전히 들어간다. "전체 파라미터를 반복 탐색하지 않는다"는 뜻.)

-----------------------------------------------------------------
v1 -> v2 변경 이력
-----------------------------------------------------------------
[FIX-1] R_err 의 max(...,0) 안전장치가 위험하게 작동하던 문제.
    sin^2(phi) > b^2 이면 R_err=0 이 되어 R_min=R_max=R_mid,
    결국 Qi_min == Qi_max 로 "불확실성 0" 을 보고했다.
    isolation 가정이 깨지는 바로 그 순간에 가장 자신 있는 답을
    내놓는 셈이다. 이제 fano_assumption_violated 플래그를 세우고
    Qi 범위를 NaN 으로 반환한다.

[FIX-2] isolation 이 측정값이 아니라 가정임을 명시.
    Qi_min/Qi_max 는 전적으로 이 값에 의존한다. 실제 배선의
    격리도를 모르면 qi_range_vs_isolation() 으로 민감도를 보일 것.

[FIX-3] spopt.newton 수렴 실패 시 전체가 죽던 문제.
    공진기를 루프로 돌릴 때 하나가 실패하면 나머지도 못 본다.
    RuntimeError 를 잡아 CircleFitError 로 바꿔 던진다.

[FIX-4] autofit docstring 의 STEP 라벨이 B -> D 로 건너뛰던 문제.
    normalize 는 별도 단계가 아니라 r0_relative = r0/a 로 흡수됨.

[FIX-5] 쓰이지 않던 calc_errors 인자 제거.

[FIX-6] 중복 주석 정리. 같은 설명이 인라인과 용어표에 두 번 있던 것을
    한쪽으로 통일.
"""

import warnings
import numpy as np
import scipy.optimize as spopt


class CircleFitError(RuntimeError):
    """circle fit 이 실패했을 때. 루프에서 개별 공진기를 건너뛰는 용도."""
    pass


# =========================================================
# 1. 물리 모델
# =========================================================
def Sij(f, fr, Ql, Qc, phi=0., a=1., alpha=0., delay=0., n_ports=2.):
    """
    공진기 산란 파라미터.  n_ports=1 -> 반사(S11), 2 -> notch 투과(S21).

        complexQc = Qc·cos(phi)·exp(-i·phi)
        S(f) = a·exp(i(alpha - 2*pi*f*delay))
               · [1 - 2Ql / (complexQc · n_ports · (1 + 2i·Ql·(f/fr - 1)))]

    complexQc 가 이 모델의 핵심이다. Qc 를 복소수로 두면 Fano 비대칭이
    위상 phi 로 들어가고, |S| <= 1 과 베이스라인 1 이 자연스럽게 나온다.
    (Khalil et al. 2012, Probst et al. 2015 의 표준형)

    Parameters
    ----------
    fr    : 공진 주파수
    Ql    : loaded Q. 총 손실. Ql = fr / kappa_total 이므로
            kappa 가 작을수록 Ql 은 커진다.
    Qc    : coupling Q. 외부(케이블)로 나가는 손실만.
    phi   : Fano 위상 [rad]. 0 이면 대칭 로렌츠, 0 이 아니면 비대칭.
            기하학적으로는 "복소평면의 원이 phi 만큼 회전한 것".
    a     : 진폭 스케일 (장비 배경)
    alpha : 위상 오프셋 (장비 배경).
            NOTE: 이 alpha 는 케이블 위상 오프셋이고, 위의 phi 는
            Fano 비대칭 각도다. 다른 스크립트에서 케이블 오프셋을
            phi 로 부른 적이 있어 혼동 주의.
    delay : 케이블 전기 지연 [s]. 장비가 알려주면 고정값으로 쓴다.
    """
    complexQc = Qc * np.cos(phi) * np.exp(-1j * phi)
    return a * np.exp(1j * (alpha - 2 * np.pi * f * delay)) * (
        1. - 2. * Ql / (complexQc * n_ports * (1. + 2j * Ql * (f / fr - 1.)))
    )


# =========================================================
# 2. 대수적 circle fit
# =========================================================
def fit_circle_algebraic(z_data):
    """
    복소 데이터에 가장 잘 맞는 원의 중심(xc, yc)과 반지름(r0)을
    반복 없이 대수적으로 계산.

    원리: (x-xc)^2 + (y-yc)^2 = r0^2 는 미지수가 2차로 얽혀 있어
    선형 최소제곱을 바로 못 쓴다. 그런데 펼쳐서 정리하면

        (x^2+y^2) = 2·xc·x + 2·yc·y + (r0^2 - xc^2 - yc^2)

    이고, z = x^2+y^2 를 보조 변수로 도입하면 z 가 (x, y, 1) 의 선형
    결합이 된다. 2차 문제가 선형 문제로 바뀐다. (arXiv:1410.3365)

    절차: 정규화 -> 모멘트 행렬 M -> 특성방정식 근으로 제약 보정 -> SVD

    Raises
    ------
    CircleFitError : 특성방정식의 근을 못 찾거나 반지름이 비물리적일 때.
    """
    # --- A. 정규화 ---
    # S21 은 보통 0.01 수준의 작은 수라 그대로 행렬 연산하면 반올림
    # 오차가 커진다. 중심을 원점 근처로, 크기를 1 근처로 맞춘다.
    x_norm = 0.5 * (np.max(z_data.real) + np.min(z_data.real))
    y_norm = 0.5 * (np.max(z_data.imag) + np.min(z_data.imag))
    z_data = z_data[:] - (x_norm + 1j * y_norm)
    amp_norm = np.max(np.abs(z_data))
    if amp_norm == 0 or not np.isfinite(amp_norm):
        raise CircleFitError("데이터의 진폭이 0 이거나 유한하지 않습니다.")
    z_data = z_data / amp_norm

    # --- B. 모멘트 행렬 ---
    # 공분산 행렬을 (x, y, z, 1) 4차원으로 확장한 것. 최소제곱에
    # 필요한 정보가 이 행렬 하나에 압축된다.
    xi = z_data.real
    yi = z_data.imag
    xi_sqr, yi_sqr = xi * xi, yi * yi
    zi = xi_sqr + yi_sqr          # 선형화를 가능하게 하는 보조 변수
    Nd = float(len(xi))

    xi_sum, yi_sum, zi_sum = xi.sum(), yi.sum(), zi.sum()
    xiyi_sum = (xi * yi).sum()
    xizi_sum = (xi * zi).sum()
    yizi_sum = (yi * zi).sum()

    M = np.array([
        [(zi * zi).sum(), xizi_sum,      yizi_sum,      zi_sum],
        [xizi_sum,        xi_sqr.sum(),  xiyi_sum,      xi_sum],
        [yizi_sum,        xiyi_sum,      yi_sqr.sum(),  yi_sum],
        [zi_sum,          xi_sum,        yi_sum,        Nd],
    ])

    # --- C. 특성방정식의 근(eta)으로 "원이 되기 위한 제약" 보정 ---
    # 아래 a0~a4 는 논문 부록의 유도 결과를 그대로 옮긴 것이다.
    # 항마다 이해할 필요는 없고, "M 에서 계수를 만들어 근을 찾는다"는
    # 흐름만 알면 된다.
    a0 = ((M[2][0]*M[3][2]-M[2][2]*M[3][0])*M[1][1]-M[1][2]*M[2][0]*M[3][1]-M[1][0]*M[2][1]*M[3][2]+M[1][0]*M[2][2]*M[3][1]+M[1][2]*M[2][1]*M[3][0])*M[0][3]+(M[0][2]*M[2][3]*M[3][0]-M[0][2]*M[2][0]*M[3][3]+M[0][0]*M[2][2]*M[3][3]-M[0][0]*M[2][3]*M[3][2])*M[1][1]+(M[0][1]*M[1][3]*M[3][0]-M[0][1]*M[1][0]*M[3][3]-M[0][0]*M[1][3]*M[3][1])*M[2][2]+(-M[0][1]*M[1][2]*M[2][3]-M[0][2]*M[1][3]*M[2][1])*M[3][0]+((M[2][3]*M[3][1]-M[2][1]*M[3][3])*M[1][2]+M[2][1]*M[3][2]*M[1][3])*M[0][0]+(M[1][0]*M[2][3]*M[3][2]+M[2][0]*(M[1][2]*M[3][3]-M[1][3]*M[3][2]))*M[0][1]+((M[2][1]*M[3][3]-M[2][3]*M[3][1])*M[1][0]+M[1][3]*M[2][0]*M[3][1])*M[0][2]
    a1 = (((M[3][0]-2.*M[2][2])*M[1][1]-M[1][0]*M[3][1]+M[2][2]*M[3][0]+2.*M[1][2]*M[2][1]-M[2][0]*M[3][2])*M[0][3]+(2.*M[2][0]*M[3][2]-M[0][0]*M[3][3]-2.*M[2][2]*M[3][0]+2.*M[0][2]*M[2][3])*M[1][1]+(-M[0][0]*M[3][3]+2.*M[0][1]*M[1][3]+2.*M[1][0]*M[3][1])*M[2][2]+(-M[0][1]*M[1][3]+2.*M[1][2]*M[2][1]-M[0][2]*M[2][3])*M[3][0]+(M[1][3]*M[3][1]+M[2][3]*M[3][2])*M[0][0]+(M[1][0]*M[3][3]-2.*M[1][2]*M[2][3])*M[0][1]+(M[2][0]*M[3][3]-2.*M[1][3]*M[2][1])*M[0][2]-2.*M[1][2]*M[2][0]*M[3][1]-2.*M[1][0]*M[2][1]*M[3][2])
    a2 = ((2.*M[1][1]-M[3][0]+2.*M[2][2])*M[0][3]+(2.*M[3][0]-4.*M[2][2])*M[1][1]-2.*M[2][0]*M[3][2]+2.*M[2][2]*M[3][0]+M[0][0]*M[3][3]+4.*M[1][2]*M[2][1]-2.*M[0][1]*M[1][3]-2.*M[1][0]*M[3][1]-2.*M[0][2]*M[2][3])
    a3 = (-2.*M[3][0]+4.*M[1][1]+4.*M[2][2]-2.*M[0][3])
    a4 = -4.

    def char_pol(x):
        return a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4

    def d_char_pol(x):
        return a1 + 2*a2*x + 3*a3*x**2 + 4*a4*x**3

    # [FIX-3] 수렴 실패를 잡는다. 공진기를 루프로 돌 때 하나가 실패해도
    # 나머지는 계속 볼 수 있어야 한다.
    try:
        eta = spopt.newton(char_pol, 0., fprime=d_char_pol, maxiter=100)
    except (RuntimeError, OverflowError, ValueError) as exc:
        raise CircleFitError(f"특성방정식의 근을 찾지 못했습니다: {exc}") from exc
    if not np.isfinite(eta):
        raise CircleFitError("특성방정식의 근이 유한하지 않습니다.")

    M[3][0] += 2*eta
    M[0][3] += 2*eta
    M[1][1] -= eta
    M[2][2] -= eta

    # --- D. SVD 로 원 파라미터 추출 ---
    # 가장 작은 특이값에 대응하는 벡터가 "M 으로 표현된 제약을 가장
    # 잘 만족하는 해" 즉 원의 방정식 계수다.
    U, s, Vt = np.linalg.svd(M)
    A_vec = Vt[np.argmin(s), :]
    if A_vec[0] == 0:
        raise CircleFitError("원의 이차항 계수가 0 입니다 (직선에 가까운 데이터).")

    xc = -A_vec[1] / (2. * A_vec[0])
    yc = -A_vec[2] / (2. * A_vec[0])
    # sqrt 안은 수치오차로 아주 살짝 음수가 될 수 있다(제약이 완전히
    # 만족되지 않는 경우). 원저자도 같은 보정을 둔다.
    disc = A_vec[1]**2 + A_vec[2]**2 - 4.*A_vec[0]*A_vec[3]
    r0 = 1. / (2. * np.absolute(A_vec[0])) * np.sqrt(max(disc, 0.0))
    if r0 <= 0 or not np.isfinite(r0):
        raise CircleFitError(f"반지름이 비물리적입니다: r0={r0}")

    return xc*amp_norm + x_norm, yc*amp_norm + y_norm, r0*amp_norm


# =========================================================
# 3. 위상 응답 (delay, fr, Ql 초기 추정)
# =========================================================
def phase_centered(f, fr, Ql, theta, delay=0.):
    """
    원점에 중심이 맞춰진 공진기의 이론 위상 응답.

    2·arctan(2Ql(1 - f/fr)) 항이 핵심. f=fr 를 지나며 arctan 이
    -pi/2 -> +pi/2 로 급변하는데, 이것이 공진 근처에서 위상이 급격히
    튀는 현상의 수학적 근원이다.

    theta : offset phase. 공진에서 먼 지점의 위상 기준점.
    """
    return theta - 2*np.pi*delay*(f-fr) + 2.*np.arctan(2.*Ql*(1. - f/fr))


def periodic_boundary(angle):
    """각도를 [-pi, pi) 로 접는다. 3.1 과 -3.1 이 큰 차이로 잘못
    계산되는 것을 막기 위해 비교 전에 항상 정리한다."""
    return (angle + np.pi) % (2*np.pi) - np.pi


def phase_distance(angle):
    """두 각도 사이의 원 위 거리를 [0, pi] 로 반환. 잔차 계산에서
    위상이 감기는 지점의 가짜 큰 오차를 막는다."""
    return np.pi - np.abs(np.pi - np.abs(angle))


def _smooth_edge_safe(x, kernel_size=11):
    """
    이동평균. 경계를 edge 값으로 패딩한다.

    np.convolve(mode='same') 은 배열 밖을 0 으로 채운다. phase 값이
    0 근처가 아니면 양 끝에서 가짜로 큰 미분값이 생기고, argmax 가
    그 가짜 봉우리를 집어 fr_guess 가 스캔 경계로 잡힌다.
    (실제로 이 버그로 Ql 이 -1e17 수준의 비물리적 값이 나온 적이 있음)
    """
    pad = kernel_size // 2
    xp = np.pad(x, pad, mode='edge')
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(xp, kernel, mode='same')[pad:-pad]


def fit_phase(f_data, z_data, guesses=None):
    """
    원점 근처로 옮겨진 데이터의 위상에 phase_centered 를 피팅해
    (fr, Ql, theta, delay) 초기 추정치를 구한다.

    4개를 한꺼번에 피팅하면 국소최적점에 빠지기 쉬워, 적은 파라미터부터
    순차적으로 정교화한다: Ql -> (fr, theta) -> delay -> (fr, Ql) -> 전체.
    """
    phase = np.unwrap(np.angle(z_data))

    # roll_off: 위상이 전체적으로 얼마나 회전했는지. 원이 완전히
    # 닫혀 있으면 2pi, 아니면 실제 회전량을 쓴다.
    if np.max(phase) - np.min(phase) <= 0.8*2*np.pi:
        roll_off = np.max(phase) - np.min(phase)
    else:
        roll_off = 2*np.pi

    if guesses is None:
        phase_smooth = _smooth_edge_safe(phase)
        phase_derivative = np.gradient(phase_smooth)
        # 위상이 가장 빠르게 변하는 지점 = fr 초기 추정
        fr_guess = f_data[np.argmax(np.abs(phase_derivative))]
        Ql_guess = 2 * fr_guess / (f_data[-1] - f_data[0])
        slope = phase[-1] - phase[0] + roll_off
        delay_guess = -slope / (2*np.pi*(f_data[-1]-f_data[0]))
    else:
        fr_guess, Ql_guess, delay_guess = guesses

    # 스캔 양 끝(공진에서 가장 먼 지점) 위상의 평균
    theta_guess = 0.5 * (np.mean(phase[:5]) + np.mean(phase[-5:]))

    def residuals_full(params):
        return phase_distance(phase - phase_centered(f_data, *params))

    def residuals_Ql(p):
        return residuals_full((fr_guess, p[0], theta_guess, delay_guess))

    def residuals_fr_theta(p):
        return residuals_full((p[0], Ql_guess, p[1], delay_guess))

    def residuals_delay(p):
        return residuals_full((fr_guess, Ql_guess, theta_guess, p[0]))

    def residuals_fr_Ql(p):
        return residuals_full((p[0], p[1], theta_guess, delay_guess))

    Ql_guess, = spopt.leastsq(residuals_Ql, [Ql_guess])[0]
    fr_guess, theta_guess = spopt.leastsq(
        residuals_fr_theta, [fr_guess, theta_guess])[0]
    delay_guess, = spopt.leastsq(residuals_delay, [delay_guess])[0]
    fr_guess, Ql_guess = spopt.leastsq(
        residuals_fr_Ql, [fr_guess, Ql_guess])[0]

    return spopt.leastsq(
        residuals_full, [fr_guess, Ql_guess, theta_guess, delay_guess])[0]


# =========================================================
# 4. 전체 파이프라인
# =========================================================
def autofit(f_data, z_data_raw, n_ports=2., fixed_delay=None, isolation=15.):
    """
    측정 데이터에서 공진기 파라미터를 자동 추출.

    처리 순서
      A. delay 결정 (fixed_delay 가 있으면 그것, 없으면 데이터에서 추정)
      B. 원의 기하학으로 fr, Ql, phi, a, alpha 계산
         (정규화는 별도 단계가 아니라 r0_relative = r0/a 로 흡수됨)
      C. 반지름으로부터 Qc, Qi 계산
      D. Fano 간섭에 의한 Qi 불확실성 범위

    Parameters
    ----------
    isolation : float [dB]
        배경(직접 누설) 경로가 신호 대비 얼마나 억제되어 있는지.

        [주의] 이것은 측정값이 아니라 **가정**이다. Qi_min/Qi_max 가
        전적으로 이 값에 의존한다. 실제 배선의 격리도를 모른다면
        qi_range_vs_isolation() 으로 민감도를 함께 보고할 것.

    Returns
    -------
    dict. 주요 키:
        fr, Ql, Qc, Qi, phi, a, alpha, delay
        Qi_min, Qi_max, Qc_min, Qc_max   (isolation 가정 하의 범위)
        fano_assumption_violated : bool
            True 면 sin^2(phi) > b^2 이라 이 isolation 으로는 관측된
            비대칭을 설명할 수 없다. 이때 Qi 범위는 NaN 이며,
            **인용하면 안 된다.**
    """
    fitresults = {}

    # --- A. delay ---
    if fixed_delay is not None:
        delay = fixed_delay
    else:
        # 원 중심을 원점으로 옮겨가며 위상 기울기를 보정. 5회 고정.
        # 감쇠 계수(0.05, 0.1)는 발산 방지를 위해 보정치를 조금씩만
        # 반영하는 장치다.
        xc, yc, r0 = fit_circle_algebraic(z_data_raw)
        z_centered = z_data_raw - complex(xc, yc)
        fr, Ql, theta, delay = fit_phase(f_data, z_centered)
        delay *= 0.05
        for _ in range(5):
            z_data = z_data_raw * np.exp(2j*np.pi*delay*f_data)
            xc, yc, r0 = fit_circle_algebraic(z_data)
            z_data = z_data - complex(xc, yc)
            fr, Ql, theta, delay_corr = fit_phase(
                f_data, z_data, guesses=(fr, Ql, 5e-11))
            delay += 0.1 * delay_corr
    fitresults['delay'] = delay

    # --- B. 원의 기하학 -> 물리 파라미터 ---
    z_data = z_data_raw * np.exp(2j*np.pi*delay*f_data)
    xc, yc, r0 = fit_circle_algebraic(z_data)
    zc = complex(xc, yc)
    z_centered = z_data - zc

    fr, Ql, theta, delay_remaining = fit_phase(f_data, z_centered)
    theta = periodic_boundary(theta)

    # beta: 원 위에서 공진점의 정반대편. 여기가 off-resonant point 다.
    beta = periodic_boundary(theta - np.pi)
    offrespoint = zc + r0*np.cos(beta) + 1j*r0*np.sin(beta)
    a = np.absolute(offrespoint)
    alpha = np.angle(offrespoint)

    # phi: off-resonant point 의 각도와 원 위 대응점 각도의 차이.
    # 이것이 Fano 위상이며, 곧 "원이 얼마나 회전했는가" 다.
    phi = periodic_boundary(beta - alpha)

    if a == 0 or not np.isfinite(a):
        raise CircleFitError("off-resonant point 의 진폭이 0 입니다.")
    r0_relative = r0 / a

    fitresults.update({
        'fr': fr, 'Ql': Ql, 'theta': theta, 'phi': phi,
        'a': a, 'alpha': alpha, 'delay_remaining': delay_remaining,
        'r0_relative': r0_relative,
    })

    # --- C. Qc, Qi ---
    # diameter correction: 정규화된 원의 지름이 Ql/|Qc| 와 같다.
    #   2·r0_relative = Ql/absQc  ->  absQc = Ql/(n_ports·r0_relative)
    absQc = Ql / (n_ports * r0_relative)
    Qc = absQc / np.cos(phi)          # Fano 위상 보정
    Qi = 1. / (1./Ql - 1./Qc)

    # 1/Ql - 1/Qc 가 0 에 가까우면 Qi 가 발산한다. 임계결합 부근에서
    # circle fit 이든 MCMC 든 공통으로 겪는 문제다.
    fitresults.update({'Qc': Qc, 'Qc_no_dia_corr': absQc, 'Qi': Qi})

    # --- D. Fano 간섭에 의한 Qi 범위 ---
    b = 10**(-isolation/20.)          # dB -> 진폭비 (전력비면 /10)
    b = b / (1 - b)

    R_mid = r0_relative * np.cos(phi)
    disc = b**2 - np.sin(phi)**2

    # [FIX-1] 가정 위반을 조용히 삼키지 않는다.
    # v1 은 sqrt 안을 max(...,0) 으로 잘라 R_err=0 을 만들었고,
    # 그 결과 Qi_min == Qi_max 가 되어 "불확실성 0" 을 보고했다.
    # 불확실성을 구하는 함수가 가정이 깨질 때 가장 자신 있는 답을
    # 내놓는 셈이라, 방향이 정반대다.
    violated = bool(disc < 0)
    if violated:
        warnings.warn(
            f"Fano 가정 위반: sin^2(phi)={np.sin(phi)**2:.4g} > "
            f"b^2={b**2:.4g} (isolation={isolation} dB, phi={phi:.4g} rad). "
            f"이 isolation 으로는 관측된 비대칭을 설명할 수 없습니다. "
            f"Qi 범위를 NaN 으로 반환합니다.",
            RuntimeWarning, stacklevel=2)
        fitresults.update({
            'fano_assumption_violated': True, 'fano_b': b,
            'Qc_min': np.nan, 'Qc_max': np.nan,
            'Qi_min': np.nan, 'Qi_max': np.nan,
        })
        return fitresults

    R_err = r0_relative * np.sqrt(disc)
    R_min, R_max = R_mid - R_err, R_mid + R_err

    def _safe_div(num, den):
        return num / den if den > 0 else np.inf

    fitresults.update({
        'fano_assumption_violated': False,
        'fano_b': b,
        'Qc_min': _safe_div(Ql, n_ports * R_max),
        'Qc_max': _safe_div(Ql, n_ports * R_min),
        'Qi_min': _safe_div(Ql, 1 - n_ports * R_min),
        'Qi_max': _safe_div(Ql, 1 - n_ports * R_max),
    })
    return fitresults


# =========================================================
# 5. isolation 민감도 [FIX-2]
# =========================================================
def qi_range_vs_isolation(f_data, z_data_raw, isolations=(10., 15., 20.),
                          n_ports=2., fixed_delay=None, verbose=True):
    """
    isolation 가정을 바꿔가며 Qi 범위가 얼마나 달라지는지 본다.

    Qi_min/Qi_max 는 측정에서 직접 나오는 값이 아니라 isolation 가정의
    함수다. 실제 배선의 격리도를 모른다면, 하나의 범위를 단정하기보다
    이 표를 함께 제시하는 편이 정직하다.

    Returns
    -------
    list of dict. 각 isolation 에 대한 {isolation, Qi, Qi_min, Qi_max,
    violated}.
    """
    rows = []
    for iso in isolations:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            r = autofit(f_data, z_data_raw, n_ports=n_ports,
                        fixed_delay=fixed_delay, isolation=iso)
        rows.append({
            'isolation': iso, 'Qi': r['Qi'],
            'Qi_min': r['Qi_min'], 'Qi_max': r['Qi_max'],
            'violated': r['fano_assumption_violated'],
        })

    if verbose:
        print(f"  {'iso[dB]':>8s} {'Qi':>12s} {'Qi_min':>12s} "
              f"{'Qi_max':>12s}  {'가정':>6s}")
        print("  " + "-" * 56)
        for r in rows:
            flag = 'VIOLATED' if r['violated'] else 'ok'
            print(f"  {r['isolation']:8.1f} {r['Qi']:12.4g} "
                  f"{r['Qi_min']:12.4g} {r['Qi_max']:12.4g}  {flag:>6s}")
        print("\n  Qi 자체는 isolation 과 무관하지만, 범위는 전적으로")
        print("  이 가정에 의존한다. 배선 격리도를 실험 쪽에서 확인할 것.")

    return rows


# =========================================================
# 자가진단
# =========================================================
if __name__ == "__main__":
    print("fano_models.py v2 자가진단")
    print("-" * 60)
    expected = ['Sij', 'fit_circle_algebraic', 'phase_centered',
                'periodic_boundary', 'phase_distance', 'fit_phase',
                'autofit', 'qi_range_vs_isolation', 'CircleFitError']
    ok_all = True
    for name in expected:
        ok = name in dir()
        print(f"  {name:26s} : {'OK' if ok else '누락!!'}")
        ok_all = ok_all and ok
    print("-" * 60)
    print("함수 상태:", "정상 (v2)" if ok_all else "일부 누락")

    # --- 합성 데이터 왕복 검증 ---
    print("\n합성 데이터로 참값 복원 검증")
    print("-" * 60)
    FR, QL, QC, PHI = 5.0e9, 8000., 12000., 0.25
    f = np.linspace(FR - 3e6, FR + 3e6, 1001)
    z = Sij(f, FR, QL, QC, phi=PHI, a=0.9, alpha=0.3, delay=0.0)
    rng = np.random.default_rng(0)
    z_noisy = z + (rng.normal(0, 2e-3, f.size)
                   + 1j*rng.normal(0, 2e-3, f.size))

    res = autofit(f, z_noisy, n_ports=2., fixed_delay=0.0, isolation=15.)
    for k, true in [('fr', FR), ('Ql', QL), ('Qc', QC), ('phi', PHI)]:
        est = res[k]
        err = abs(est - true) / abs(true) * 100
        print(f"  {k:5s} 추정={est:14.6g}  참값={true:14.6g}  "
              f"오차={err:7.2f}%")
    print(f"\n  Fano 가정 위반: {res['fano_assumption_violated']}")
    print(f"  Qi = {res['Qi']:.4g}  범위 = "
          f"[{res['Qi_min']:.4g}, {res['Qi_max']:.4g}]")

    print("\nisolation 민감도")
    print("-" * 60)
    qi_range_vs_isolation(f, z_noisy, isolations=(10., 15., 20., 25.),
                          fixed_delay=0.0)
