"""
fano_models.py  (정리본)
=================================================================
원저자 circuit.py (Rieger & Guenzler et al., KIT / Sebastian Probst의
resonator_tools 기반)의 "대수적 원 피팅(algebraic circle fit)" 방법을
함수형으로 이식한 모듈.

-----------------------------------------------------------------
이 파일이 하는 일 (한 문단 요약)
-----------------------------------------------------------------
주파수마다 측정된 복소수 S21 값을 복소평면에 점으로 찍으면 "원"을
그린다. 이 원의 기하학적 성질(중심, 반지름, 기울어진 각도)만 알면
공진기의 물리 파라미터 — 공진주파수 fr, 품질계수 Ql/Qc/Qi, Fano
위상 phi — 를 반복 탐색 없이 대수 공식으로 한 번에 계산할 수 있다.

-----------------------------------------------------------------
MCMC 방식과의 차이
-----------------------------------------------------------------
MCMC : 파라미터 후보를 계속 바꿔가며 "이 후보가 데이터를 얼마나 잘
       설명하는가"를 반복 평가해 점점 접근(수렴)한다. 시간이 걸리고,
       파라미터끼리 축퇴(degeneracy)가 있으면 수렴이 어렵다.
이 파일 : 데이터가 원을 그린다는 기하학적 사실을 이용해 행렬 연산
       한 번으로 원의 방정식을 직접 푼다. 항상 정해진 시간 안에
       끝난다. 단, 데이터가 실제로 원을 그리는 상황에서만 쓸 수 있다.

주의: "반복이 전혀 없다"는 것은 정확한 표현이 아니다. 전체 파라미터를
한꺼번에 탐색하지 않을 뿐, 원 중심을 구하는 특성방정식의 근(뉴턴법)이나
delay 보정(최소제곱) 같은 국소적인 짧은 반복은 여전히 쓰인다.

-----------------------------------------------------------------
정리본에서 바뀐 점
-----------------------------------------------------------------
[FIX-1] Fano 불확실성 계산의 max(..., 0) 처리에 플래그 추가.
    기존에는 sin(phi)^2 > b^2 (= isolation 가정이 깨진 경우)일 때
    R_err 가 0 이 되어 Qi_min == Qi_max, 즉 "불확실성이 0" 으로
    보고되었다. 가정이 무너지는 바로 그 순간에 가장 자신 있는 답을
    내놓는 셈이라, 이 함수의 목적과 정반대다.
    이제 fano_assumption_violated 플래그를 결과에 넣고, 위반 시
    Qi 범위를 NaN 으로 반환한다.

[FIX-2] isolation 이 "측정값이 아니라 가정" 임을 명시.
    Qi_min/Qi_max 는 전적으로 이 값에 달려 있다. 결과 딕셔너리에
    isolation_dB 를 함께 담아, 나중에 어떤 가정으로 얻은 범위인지
    추적할 수 있게 했다.

[FIX-3] 뉴턴법 수렴 실패 시 np.roots 폴백 + 예외 처리.
    공진기를 루프로 돌리다 하나가 실패하면 전체가 죽던 문제.

[FIX-4] 쓰이지 않던 calc_errors 인자 제거.

[FIX-5] autofit 의 STEP 표기를 실제 코드 흐름에 맞춤 (A~E).
    기존 docstring 은 STEP C(normalize)를 적어놓고 코드에는 없었다.
"""

import warnings

import numpy as np
import scipy.optimize as spopt
# scipy.optimize : 최적화(어떤 값을 최소/최대로 만드는 파라미터 찾기)
#   함수 모음. 이 파일에서 쓰는 것은 두 개다.
#     newton  : 뉴턴-랩슨법. 방정식 f(x)=0 의 근을 도함수(기울기)
#               정보를 써서 빠르게 찾는 반복 알고리즘.
#     leastsq : Levenberg-Marquardt 알고리즘 기반 비선형 최소제곱법.
#               데이터와 모델 예측값의 차이(잔차)의 제곱합을 최소로
#               만드는 파라미터를 찾는다.


# =========================================================
# 1. 물리 모델
# =========================================================
def Sij(f, fr, Ql, Qc, phi=0.0, a=1.0, alpha=0.0, delay=0.0, n_ports=2.0):
    """
    공진기의 산란 파라미터(scattering parameter) 모델식.

    산란 파라미터 : 포트에 신호를 넣었을 때 다른 포트로 나오는(또는
        되돌아오는) 신호의 복소수 비율. S21 은 포트1→포트2 투과,
        S11 은 포트1 반사를 뜻한다.

    수식
    ----
        complexQc = Qc · cos(phi) · exp(-i·phi)
        S(f) = a · exp(i·(alpha - 2π·f·delay))
               · [ 1 - 2Ql / ( complexQc · n_ports · (1 + 2i·Ql·(f/fr - 1)) ) ]

    이 형태는 Khalil et al. (2012), Probst et al. (2015) 의 표준형이다.
    결합 품질계수 Qc 를 복소수로 두어 Fano 비대칭을 위상 phi 로
    넣는 것이 핵심 — 곱셈으로 진폭을 건드리지 않으므로 공진에서
    멀어지면 |S| → a 로 자동 수렴한다.

    Parameters
    ----------
    f : array. 주파수 (Hz 또는 GHz. delay 와 단위만 맞으면 됨)
    fr : float. 공진 주파수(resonance frequency). 공진기가 가장 강하게
        반응하는 주파수.
    Ql : float. loaded Q (로드된 품질계수). 무차원. "에너지를 잃기 전에
        몇 번이나 진동하는가". 내부 손실과 외부 결합 손실을 모두 합친
        총 손실에 대응한다. Ql = fr / kappa_total 이므로, 손실 kappa 가
        작을수록 Ql 은 커진다 (방향이 반대인 점에 주의).
    Qc : float. coupling Q (결합 품질계수). 공진기가 외부 케이블로
        에너지를 내보내는 정도만 뽑은 값.
    phi : float. Fano 위상 (라디안). 0 이면 대칭 로렌츠 공진,
        0 이 아니면 배경 신호와의 간섭으로 딥/피크가 비대칭으로
        기울어진다. 기하학적으로는 "복소평면의 원이 phi 만큼
        회전한 것" 과 같은 효과다.
    a : float. 전체 신호의 진폭 스케일 (장비 배경 보정용).
    alpha : float. 전체 위상 오프셋 (장비 배경 보정용).
        주의 — 여기서 phi 와 alpha 는 다른 물리량이다.
        phi 는 Fano 비대칭 각도, alpha 는 케이블/장비 위상 오프셋.
    delay : float. 케이블 전기 지연(electrical delay). 신호가 배선을
        지나며 생기는 선형 위상 회전의 원인. 장비(VNA)가 값을
        알려주는 경우가 많다.
    n_ports : float. 1 이면 단일 포트 반사(S11), 2 이면 2포트
        투과/notch(S21).

    Returns
    -------
    복소수 배열. f 와 같은 길이.
    """
    complex_Qc = Qc * np.cos(phi) * np.exp(-1j * phi)
    background = a * np.exp(1j * (alpha - 2.0 * np.pi * f * delay))
    lorentzian = 2.0 * Ql / (
        complex_Qc * n_ports * (1.0 + 2j * Ql * (f / fr - 1.0))
    )
    return background * (1.0 - lorentzian)


# =========================================================
# 2. 대수적 원 피팅
# =========================================================
def fit_circle_algebraic(z_data):
    """
    복소수 데이터에 가장 잘 맞는 원의 중심과 반지름을, 반복 탐색 없이
    대수적으로 계산한다.

    왜 대수 문제가 되는가
    --------------------
    원의 방정식 (x-xc)² + (y-yc)² = r0² 는 미지수가 제곱으로 얽혀
    있어 선형 최소제곱법을 바로 쓸 수 없다. 그런데 펼쳐서 정리하면

        (x² + y²) = 2·xc·x + 2·yc·y + (r0² - xc² - yc²)

    가 되고, 여기서 z = x² + y² 라는 보조 변수를 도입하면 z 가
    x, y, 1 의 선형 결합으로 표현된다. 겉보기 2차 문제가 선형
    최소제곱 문제로 바뀐다. (Probst et al., arXiv:1410.3365)

    절차
    ----
    A. 정규화 — 숫자 크기를 1 근처로 맞춰 반올림 오차를 줄인다
    B. 모멘트 행렬 M 구성
    C. 특성방정식의 근(eta)으로 제약 조건 보정
    D. SVD 로 원의 파라미터 추출

    Parameters
    ----------
    z_data : complex array. 실수부가 x 좌표, 허수부가 y 좌표.

    Returns
    -------
    xc, yc, r0 : float. 원의 중심 좌표와 반지름 (원래 스케일).
    """
    z_data = np.asarray(z_data, dtype=complex)
    if z_data.size < 4:
        raise ValueError(f"원 피팅에는 최소 4개 점이 필요합니다 "
                         f"(받은 개수: {z_data.size})")

    # --- STEP A. 정규화 ---------------------------------------------
    # S21 값은 보통 0.01 근처의 작은 수다. 이대로 4x4 행렬 연산을 하면
    # 부동소수점 오차가 커진다. 중심을 원점 근처로 옮기고 크기를 1
    # 근처로 맞춘 뒤 계산하고, 마지막에 되돌린다.
    x_norm = 0.5 * (np.max(z_data.real) + np.min(z_data.real))
    y_norm = 0.5 * (np.max(z_data.imag) + np.min(z_data.imag))
    z = z_data - (x_norm + 1j * y_norm)

    amp_norm = np.max(np.abs(z))
    if amp_norm == 0:
        raise ValueError("모든 데이터 점이 동일합니다. 원을 정의할 수 없습니다.")
    z = z / amp_norm

    # --- STEP B. 모멘트 행렬 구성 -----------------------------------
    # 모멘트(moment) : 통계에서 데이터 분포의 퍼짐/치우침을 나타내는 양.
    #   여기서는 x, y, z 값들을 짝지어 곱하고 합한 값들로 4x4 행렬을
    #   만든다. 공분산 행렬을 (x, y, z, 1) 4차원으로 확장한 것과 비슷한
    #   역할로, 최소제곱에 필요한 모든 정보가 이 행렬 하나에 압축된다.
    xi = z.real
    yi = z.imag
    zi = xi * xi + yi * yi          # 선형화를 가능하게 하는 보조 변수
    Nd = float(len(xi))             # 데이터 점 개수

    xi_sum, yi_sum, zi_sum = xi.sum(), yi.sum(), zi.sum()
    xiyi_sum = (xi * yi).sum()
    xizi_sum = (xi * zi).sum()
    yizi_sum = (yi * zi).sum()

    M = np.array([
        [(zi * zi).sum(), xizi_sum,      yizi_sum,      zi_sum],
        [xizi_sum,        (xi * xi).sum(), xiyi_sum,    xi_sum],
        [yizi_sum,        xiyi_sum,      (yi * yi).sum(), yi_sum],
        [zi_sum,          xi_sum,        yi_sum,        Nd],
    ])

    # --- STEP C. 특성방정식의 근으로 제약 조건 보정 -----------------
    # 특성방정식(characteristic polynomial) : 어떤 조건을 만족시키는
    #   값을 찾기 위해 세우는 다항식. 그 근이 우리가 찾는 해다.
    #
    # 이차곡선(원, 타원, 쌍곡선...) 중에서 "정확히 원" 이 되려면
    # 만족해야 하는 제약이 있다. 라그랑주 승수법과 비슷한 방식으로
    # 보정값 eta 를 구해 M 을 수정한다.
    #
    # 아래 a0~a4 는 M 의 원소를 조합한 긴 대수식이다. 손으로 유도된
    # 공식을 그대로 옮긴 것이라 항마다 이해할 필요는 없다. 흐름은
    # "M 에서 계수를 뽑아 4차 방정식을 세우고 그 근을 찾는다" 이다.
    a0 = (
        ((M[2][0] * M[3][2] - M[2][2] * M[3][0]) * M[1][1]
         - M[1][2] * M[2][0] * M[3][1] - M[1][0] * M[2][1] * M[3][2]
         + M[1][0] * M[2][2] * M[3][1] + M[1][2] * M[2][1] * M[3][0]) * M[0][3]
        + (M[0][2] * M[2][3] * M[3][0] - M[0][2] * M[2][0] * M[3][3]
           + M[0][0] * M[2][2] * M[3][3] - M[0][0] * M[2][3] * M[3][2]) * M[1][1]
        + (M[0][1] * M[1][3] * M[3][0] - M[0][1] * M[1][0] * M[3][3]
           - M[0][0] * M[1][3] * M[3][1]) * M[2][2]
        + (-M[0][1] * M[1][2] * M[2][3] - M[0][2] * M[1][3] * M[2][1]) * M[3][0]
        + ((M[2][3] * M[3][1] - M[2][1] * M[3][3]) * M[1][2]
           + M[2][1] * M[3][2] * M[1][3]) * M[0][0]
        + (M[1][0] * M[2][3] * M[3][2]
           + M[2][0] * (M[1][2] * M[3][3] - M[1][3] * M[3][2])) * M[0][1]
        + ((M[2][1] * M[3][3] - M[2][3] * M[3][1]) * M[1][0]
           + M[1][3] * M[2][0] * M[3][1]) * M[0][2]
    )
    a1 = (
        ((M[3][0] - 2.0 * M[2][2]) * M[1][1] - M[1][0] * M[3][1]
         + M[2][2] * M[3][0] + 2.0 * M[1][2] * M[2][1]
         - M[2][0] * M[3][2]) * M[0][3]
        + (2.0 * M[2][0] * M[3][2] - M[0][0] * M[3][3]
           - 2.0 * M[2][2] * M[3][0] + 2.0 * M[0][2] * M[2][3]) * M[1][1]
        + (-M[0][0] * M[3][3] + 2.0 * M[0][1] * M[1][3]
           + 2.0 * M[1][0] * M[3][1]) * M[2][2]
        + (-M[0][1] * M[1][3] + 2.0 * M[1][2] * M[2][1]
           - M[0][2] * M[2][3]) * M[3][0]
        + (M[1][3] * M[3][1] + M[2][3] * M[3][2]) * M[0][0]
        + (M[1][0] * M[3][3] - 2.0 * M[1][2] * M[2][3]) * M[0][1]
        + (M[2][0] * M[3][3] - 2.0 * M[1][3] * M[2][1]) * M[0][2]
        - 2.0 * M[1][2] * M[2][0] * M[3][1]
        - 2.0 * M[1][0] * M[2][1] * M[3][2]
    )
    a2 = (
        (2.0 * M[1][1] - M[3][0] + 2.0 * M[2][2]) * M[0][3]
        + (2.0 * M[3][0] - 4.0 * M[2][2]) * M[1][1]
        - 2.0 * M[2][0] * M[3][2] + 2.0 * M[2][2] * M[3][0]
        + M[0][0] * M[3][3] + 4.0 * M[1][2] * M[2][1]
        - 2.0 * M[0][1] * M[1][3] - 2.0 * M[1][0] * M[3][1]
        - 2.0 * M[0][2] * M[2][3]
    )
    a3 = -2.0 * M[3][0] + 4.0 * M[1][1] + 4.0 * M[2][2] - 2.0 * M[0][3]
    a4 = -4.0

    def char_pol(x):
        return a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3 + a4 * x ** 4

    def d_char_pol(x):
        # 도함수. 뉴턴법이 "현재 위치의 기울기를 보고 근이 있을 법한
        # 방향으로 이동" 하는 방식이므로 필요하다.
        return a1 + 2 * a2 * x + 3 * a3 * x ** 2 + 4 * a4 * x ** 3

    # [FIX-3] 뉴턴법이 실패해도 전체가 죽지 않도록 폴백을 둔다.
    try:
        eta = spopt.newton(char_pol, 0.0, fprime=d_char_pol, maxiter=100)
    except (RuntimeError, OverflowError, ValueError):
        # 폴백: 4차 방정식의 모든 근을 직접 구하고, 실근 중 0 에 가장
        # 가까운 것을 고른다. 뉴턴법이 0 에서 출발하는 것과 같은 의도다.
        roots = np.roots([a4, a3, a2, a1, a0])
        real_roots = roots[np.abs(roots.imag) < 1e-9].real
        if real_roots.size == 0:
            raise RuntimeError(
                "특성방정식의 실근을 찾지 못했습니다. "
                "데이터가 원 형태가 아닐 수 있습니다.")
        eta = float(real_roots[np.argmin(np.abs(real_roots))])
        warnings.warn("뉴턴법이 수렴하지 않아 np.roots 폴백을 사용했습니다.",
                      RuntimeWarning)

    M[3][0] += 2 * eta
    M[0][3] += 2 * eta
    M[1][1] -= eta
    M[2][2] -= eta

    # --- STEP D. SVD 로 원의 파라미터 추출 --------------------------
    # SVD(특이값분해, Singular Value Decomposition) : 임의의 행렬을
    #   U · diag(s) · Vᵀ 세 부분으로 분해하는 선형대수의 표준 도구.
    #   여기서는 "M 과 곱했을 때 결과가 0 에 가장 가까워지는 방향"을
    #   찾는 데 쓴다. 그 방향이 곧 제약 조건을 가장 잘 만족하는 해다.
    _, s, Vt = np.linalg.svd(M)
    A_vec = Vt[np.argmin(s), :]     # 가장 작은 특이값에 대응하는 벡터

    if A_vec[0] == 0:
        raise RuntimeError("원의 방정식 계수가 퇴화했습니다 (A_vec[0]=0).")

    xc = -A_vec[1] / (2.0 * A_vec[0])
    yc = -A_vec[2] / (2.0 * A_vec[0])

    # sqrt 안쪽은 수치오차로 아주 살짝 음수가 될 수 있다(제약이 완벽히
    # 만족되지 않는 경우). 원저자 코드도 이 점을 명시한다.
    disc = A_vec[1] ** 2 + A_vec[2] ** 2 - 4.0 * A_vec[0] * A_vec[3]
    r0 = np.sqrt(max(disc, 0.0)) / (2.0 * np.abs(A_vec[0]))

    # 정규화를 되돌려 원래 스케일로 복원
    return (xc * amp_norm + x_norm,
            yc * amp_norm + y_norm,
            r0 * amp_norm)


# =========================================================
# 3. 위상 응답 피팅
# =========================================================
def phase_centered(f, fr, Ql, theta, delay=0.0):
    """
    원점에 중심이 맞춰진 공진기의 이론적 위상 응답.

        phase(f) = theta - 2π·delay·(f - fr) + 2·arctan(2·Ql·(1 - f/fr))

    핵심은 arctan 항이다. f 가 fr 을 지나가면서 arctan 이 -π/2 에서
    +π/2 로 급격히 변하는데, 이것이 "공진 근처에서 위상이 급격히
    튀어오르는" 현상의 수학적 근원이다.

    theta : offset phase(오프셋 위상). 공진에서 아주 멀리 떨어진
        지점의 위상 값. 배경 위상의 기준점 역할.
    """
    return theta - 2 * np.pi * delay * (f - fr) + 2.0 * np.arctan(
        2.0 * Ql * (1.0 - f / fr))


def periodic_boundary(angle):
    """
    임의의 각도를 [-π, π) 구간으로 접어넣는다.

    위상은 원형(circular) 값이라 2π 를 더하거나 빼도 물리적으로 같다.
    항상 대표 구간 안으로 정리해두지 않으면, 예를 들어 3.1 과 -3.1 이
    실제로는 거의 같은 값인데 큰 차이로 잘못 계산된다.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def phase_distance(angle):
    """
    두 각도 사이의 "원 위에서의 거리" 를 [0, π] 로 반환.

    각도 차이가 350° 든 -10° 든 실제로는 10° 떨어진 것이다.
    잔차 계산에 이 거리를 써야 위상이 감기는 지점에서 엉뚱하게 큰
    오차가 잡히는 것을 막을 수 있다.
    """
    return np.pi - np.abs(np.pi - np.abs(angle))


def _smooth_edge_safe(x, kernel_size=11):
    """
    가장자리 안전 이동평균.

    np.convolve(mode='same') 은 배열 밖을 0 으로 채운다(zero-padding).
    데이터 값이 0 근처가 아니면 양 끝에서 실제 신호가 아닌 인위적인
    급변이 생기고, 그 가짜 봉우리를 argmax 가 집어 fr 초기값이 스캔
    경계로 잘못 뽑힌다. (합성 데이터 검증 중 Ql 이 -10^17 수준의
    비물리적 값으로 나와 역추적해 발견한 문제)

    np.pad(mode='edge') 로 양 끝을 가장자리 값으로 늘린 뒤 합성곱하면
    이 문제가 사라진다.
    """
    pad = kernel_size // 2
    x_padded = np.pad(x, pad, mode='edge')
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(x_padded, kernel, mode='same')
    return smoothed[pad:-pad]


def fit_phase(f_data, z_data, guesses=None):
    """
    (이미 원점 근처로 옮겨진) 데이터의 위상 응답에 phase_centered
    모델을 피팅해 (fr, Ql, theta, delay) 를 구한다.

    단계적 피팅 전략
    ---------------
    4개를 처음부터 동시에 피팅하면 국소최적점(local minimum)에 빠지기
    쉽다. 그래서 적은 파라미터부터 순차적으로 피팅해 초기값을 다듬은 뒤
    마지막에 전체를 함께 피팅한다.

        1. Ql 만
        2. fr, theta
        3. delay 만
        4. fr, Ql
        5. 전체 4개

    Returns
    -------
    (fr, Ql, theta, delay)
    """
    phase = np.unwrap(np.angle(z_data))
    # np.angle  : 복소수의 위상(각도)
    # np.unwrap : ±π 경계를 넘을 때 생기는 인위적 점프를 제거해
    #             물리적으로 연속인 곡선으로 펼친다.
    #             예: 3.0, 3.1, -3.1, -3.0 → 3.0, 3.1, 3.2, 3.3

    # roll_off : 위상이 전체적으로 얼마나 회전했는가(2π = 한 바퀴).
    #   원점에 중심이 완전히 맞으면 정확히 한 바퀴를 돌아야 한다.
    #   실제 데이터가 그에 못 미치면(원이 안 닫혀 있으면) 실제 값을 쓴다.
    phase_span = np.max(phase) - np.min(phase)
    roll_off = phase_span if phase_span <= 0.8 * 2 * np.pi else 2 * np.pi

    if guesses is None:
        phase_smooth = _smooth_edge_safe(phase)
        phase_derivative = np.gradient(phase_smooth)
        # 위상이 가장 빠르게 변하는 지점이 공진 주파수다.
        # phase_centered 식에서 f = fr 근처에서 arctan 의 기울기가
        # 최대가 되기 때문이다.
        fr_guess = f_data[np.argmax(np.abs(phase_derivative))]
        Ql_guess = 2 * fr_guess / (f_data[-1] - f_data[0])
        slope = phase[-1] - phase[0] + roll_off
        delay_guess = -slope / (2 * np.pi * (f_data[-1] - f_data[0]))
    else:
        fr_guess, Ql_guess, delay_guess = guesses

    # 스캔 양 끝(공진에서 가장 먼 지점)의 위상 평균을 오프셋 초기값으로
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
def _fit_delay_iterative(f_data, z_data_raw, n_iter=5):
    """
    데이터 자체에서 케이블 지연을 반복 추정한다.
    (circuit.py 의 _fit_delay 를 단순화한 버전)

    원의 중심을 원점으로 옮겨가며 위상 기울기를 보정하는 절차를
    정해진 횟수만 반복한다. 매번 보정치를 조금씩만 반영하는데,
    이는 한 번에 다 반영하면 발산할 수 있기 때문이다
    ("과잉반응하지 않기", do not overreact).
    """
    xc, yc, _ = fit_circle_algebraic(z_data_raw)
    z_centered = z_data_raw - complex(xc, yc)
    fr, Ql, _theta, delay = fit_phase(f_data, z_centered)

    delay *= 0.05                       # 첫 추정치는 5% 만 반영
    for _ in range(n_iter):
        z = z_data_raw * np.exp(2j * np.pi * delay * f_data)
        xc, yc, _ = fit_circle_algebraic(z)
        z = z - complex(xc, yc)
        fr, Ql, _theta, delay_corr = fit_phase(
            f_data, z, guesses=(fr, Ql, 5e-11))
        delay += 0.1 * delay_corr       # 이후로는 10% 씩
    return delay


def autofit(f_data, z_data_raw, n_ports=2.0, fixed_delay=None,
            isolation_dB=15.0):
    """
    측정 데이터로부터 공진기의 물리 파라미터를 자동 추출한다.

    처리 순서
    ---------
    A. delay 결정      — 주어지면 그대로, 아니면 데이터에서 추정
    B. calibrate       — 원 + 위상 피팅으로 fr, Ql, theta, phi, a, alpha
    C. normalize       — 진폭 a 로 반지름을 정규화 (r0_relative)
    D. extract Qs      — 반지름에서 Qc, Qi 계산
    E. Fano 범위       — Fano 간섭에 의한 Qi 불확실성 범위

    Parameters
    ----------
    f_data : array. 주파수
    z_data_raw : complex array. 측정된 S21 (보정 전)
    n_ports : float. 1(반사) 또는 2(투과/notch)
    fixed_delay : float or None. VNA 가 알려준 electrical delay 가
        있으면 넣는다. None 이면 데이터에서 추정한다.
    isolation_dB : float.
        [중요] 이것은 측정값이 아니라 **가정**이다.
        배경(간섭) 경로가 원래 신호 대비 얼마나 억제되어 있는지를
        나타내는 dB 값으로, STEP E 의 Qi_min/Qi_max 가 전적으로 이
        값에 달려 있다. 실제 배선의 격리도를 모른다면, 여러 값
        (예: 10 / 15 / 20 dB)으로 돌려 범위가 얼마나 달라지는지
        함께 보고하는 편이 정직하다.

    Returns
    -------
    dict. 주요 키:
        fr, Ql, Qc, Qi, phi, a, alpha, delay
        Qc_no_dia_corr           : 지름 보정 전 |Qc|
        Qi_min, Qi_max           : Fano 불확실성 범위
        Qc_min, Qc_max
        isolation_dB             : 어떤 가정으로 얻은 범위인지 추적용
        fano_assumption_violated : True 면 Qi 범위를 신뢰하지 말 것
    """
    f_data = np.asarray(f_data, dtype=float)
    z_data_raw = np.asarray(z_data_raw, dtype=complex)
    if f_data.shape != z_data_raw.shape:
        raise ValueError(f"f_data 와 z_data_raw 의 길이가 다릅니다: "
                         f"{f_data.shape} vs {z_data_raw.shape}")

    results = {}

    # --- STEP A. delay 결정 -----------------------------------------
    if fixed_delay is not None:
        delay = float(fixed_delay)
    else:
        delay = _fit_delay_iterative(f_data, z_data_raw)
    results['delay'] = delay

    # --- STEP B. calibrate ------------------------------------------
    # delay 를 되돌린(de-embed) 뒤 원을 맞춘다.
    z_data = z_data_raw * np.exp(2j * np.pi * delay * f_data)
    xc, yc, r0 = fit_circle_algebraic(z_data)
    zc = complex(xc, yc)
    z_centered = z_data - zc

    fr, Ql, theta, delay_remaining = fit_phase(f_data, z_centered)
    theta = periodic_boundary(theta)

    # beta : off-resonant point 의 각도. 원 위에서 공진점의 정반대편.
    #   off-resonant point 란 "공진에서 무한히 멀어졌을 때 신호가
    #   수렴하는 지점" 으로, 배경 신호의 기준점이 된다.
    beta = periodic_boundary(theta - np.pi)
    offres_point = zc + r0 * np.cos(beta) + 1j * r0 * np.sin(beta)

    a = np.abs(offres_point)            # 배경 진폭
    alpha = np.angle(offres_point)      # 배경 위상
    phi = periodic_boundary(beta - alpha)   # Fano 위상

    results.update({
        'fr': fr, 'Ql': Ql, 'theta': theta, 'phi': phi,
        'a': a, 'alpha': alpha, 'delay_remaining': delay_remaining,
        'circle_center': zc, 'circle_radius': r0,
    })

    # --- STEP C. normalize ------------------------------------------
    # 배경 진폭 a 로 나누어, 장비 이득과 무관한 상대 반지름을 만든다.
    if a == 0:
        raise RuntimeError("배경 진폭 a 가 0 입니다. 데이터를 확인하세요.")
    r0_relative = r0 / a

    # --- STEP D. Qc, Qi 추출 ----------------------------------------
    # diameter correction method : 원의 지름이 Qc 와 직접 연결된다는
    #   기하학적 사실을 이용한다. 원이 클수록(반지름이 클수록) 결합이
    #   강하고 Qc 는 작다 — 그래서 r0_relative 가 분모에 온다.
    abs_Qc = Ql / (n_ports * r0_relative)
    Qc = abs_Qc / np.cos(phi)           # Fano 위상 반영한 "진짜" Qc

    # 1/Ql = 1/Qi + 1/Qc  →  Qi = 1/(1/Ql - 1/Qc)
    #
    # [주의] 이 식이 지원서에서 말한 "임계결합 부근에서 분모가 0 이
    # 되어 발산하는" 바로 그 구조다. Ql 과 Qc 가 가까워지면
    # (1/Ql - 1/Qc) → 0 이 되어 Qi 가 폭발하고, Ql·Qc 의 작은 오차가
    # Qi 로 크게 증폭되어 전파된다.
    denom = 1.0 / Ql - 1.0 / Qc
    Qi = np.inf if denom == 0 else 1.0 / denom

    results.update({'Qc': Qc, 'Qc_no_dia_corr': abs_Qc, 'Qi': Qi,
                    'r0_relative': r0_relative})

    # --- STEP E. Fano 간섭에 의한 Qi 불확실성 범위 -------------------
    # isolation(dB) 을 선형 진폭비로 환산.
    #   진폭비 = 10^(-dB/20).  (전력비라면 /10 이지만 여기는 진폭이다)
    b_raw = 10 ** (-isolation_dB / 20.0)
    b = b_raw / (1.0 - b_raw)

    R_mid = r0_relative * np.cos(phi)
    disc = b ** 2 - np.sin(phi) ** 2

    # [FIX-1] disc < 0 이면 이 isolation 가정으로는 설명할 수 없을 만큼
    # phi 가 크다는 뜻이다. 예전처럼 0 으로 눌러버리면 R_err = 0 이 되어
    # Qi_min == Qi_max, 즉 "불확실성이 0" 이라는 정반대의 결론이 나온다.
    # 플래그를 세우고 범위는 NaN 으로 반환한다.
    violated = disc < 0
    results['fano_assumption_violated'] = bool(violated)
    results['isolation_dB'] = float(isolation_dB)
    results['fano_b'] = b

    if violated:
        warnings.warn(
            f"Fano 가정 위반: sin(phi)^2 = {np.sin(phi)**2:.4g} > "
            f"b^2 = {b**2:.4g}.\n"
            f"  isolation_dB={isolation_dB} 로는 설명할 수 없는 크기의 "
            f"phi={phi:.4f} rad 입니다.\n"
            f"  Qi_min/Qi_max 를 NaN 으로 반환합니다. "
            f"isolation 가정을 재검토하세요.",
            RuntimeWarning)
        results.update({'Qc_min': np.nan, 'Qc_max': np.nan,
                        'Qi_min': np.nan, 'Qi_max': np.nan})
        return results

    R_err = r0_relative * np.sqrt(disc)
    R_min, R_max = R_mid - R_err, R_mid + R_err

    def _safe_div(num, den):
        return num / den if den > 0 else np.inf

    results.update({
        'Qc_min': _safe_div(Ql, n_ports * R_max),
        'Qc_max': _safe_div(Ql, n_ports * R_min),
        'Qi_min': _safe_div(Ql, 1.0 - n_ports * R_min),
        'Qi_max': _safe_div(Ql, 1.0 - n_ports * R_max),
    })
    return results


# =========================================================
# 자가진단
# =========================================================
if __name__ == "__main__":
    print("fano_models.py 자가진단")
    print("-" * 60)
    expected = ['Sij', 'fit_circle_algebraic', 'phase_centered',
                'periodic_boundary', 'phase_distance', '_smooth_edge_safe',
                'fit_phase', '_fit_delay_iterative', 'autofit']
    all_ok = True
    for name in expected:
        ok = name in dir()
        print(f"  {name:24s} : {'OK' if ok else '누락!!'}")
        all_ok = all_ok and ok
    print("-" * 60)
    print("함수 상태:", "정상" if all_ok else "일부 누락")

    # ---- 왕복 시험: Sij 로 만든 신호를 autofit 이 복원하는가 ----
    print("\n왕복 시험 (round-trip): 참값 복원 정확도")
    print("-" * 60)
    rng = np.random.default_rng(0)
    for phi_true in [0.0, 0.2, 0.5]:
        fr_t, Ql_t, Qc_t = 6.0e9, 8000.0, 12000.0
        f = np.linspace(fr_t - 3e6, fr_t + 3e6, 601)
        z = Sij(f, fr_t, Ql_t, Qc_t, phi=phi_true, a=0.8,
                alpha=0.3, delay=3e-9, n_ports=2.0)
        z = z + (rng.normal(0, 1e-4, z.size)
                 + 1j * rng.normal(0, 1e-4, z.size))

        try:
            r = autofit(f, z, n_ports=2.0, fixed_delay=3e-9)
            e_fr = abs(r['fr'] - fr_t) / fr_t * 100
            e_Ql = abs(r['Ql'] - Ql_t) / Ql_t * 100
            e_Qc = abs(r['Qc'] - Qc_t) / Qc_t * 100
            e_phi = abs(r['phi'] - phi_true)
            flag = " [가정위반]" if r['fano_assumption_violated'] else ""
            print(f"  phi_true={phi_true:.2f} | "
                  f"fr {e_fr:6.3f}%  Ql {e_Ql:6.2f}%  "
                  f"Qc {e_Qc:6.2f}%  phi 오차 {e_phi:.4f} rad{flag}")
        except Exception as exc:
            print(f"  phi_true={phi_true:.2f} | 실패: "
                  f"{type(exc).__name__}: {exc}")


# =========================================================================
# 용어 참고표 (Glossary)
# 본문 주석에서 처음 나올 때 설명한 용어를 한곳에 모은 색인.
# =========================================================================
#
# --- 물리 ---
# S21 / S11        포트1→포트2 투과 / 포트1 반사의 복소수 비율.
# fr               공진 주파수.
# Ql (loaded Q)    총 손실 기준 품질계수. Ql = fr / kappa_total.
# Qc (coupling Q)  외부 케이블로의 결합 손실만 나타내는 품질계수.
# Qi (internal Q)  공진기 자체의 내부 손실만. 1/Ql = 1/Qi + 1/Qc.
# phi              Fano 위상. 배경 신호와의 간섭에 의한 비대칭 정도.
# a, alpha         전체 진폭 스케일 / 위상 오프셋 (장비 배경 보정용).
# delay            케이블 전기 지연. 선형 위상 회전의 원인.
# isolation        배경 경로가 신호 대비 얼마나 억제되어 있는지(dB).
#                  측정값이 아니라 가정임에 주의.
# off-resonant pt  공진에서 무한히 멀 때 신호가 수렴하는 지점.
# over/undercoupled  Qc < Qi 면 overcoupled, Qc > Qi 면 undercoupled,
#                  같으면 critical coupling.
#
# --- 수학 / 통계 ---
# algebraic fit    반복 없이 방정식을 한 번 풀어 답을 구하는 방식.
# iterative fit    후보를 여러 번 시도하며 접근하는 방식(MCMC, leastsq).
# 특성방정식        어떤 제약을 만족시키는 값을 찾기 위해 세우는 다항식.
# 뉴턴법            도함수를 이용해 방정식의 근을 빠르게 찾는 반복법.
# SVD              행렬을 U·diag(s)·Vᵀ 로 분해하는 선형대수 도구.
# 최소제곱법        잔차의 제곱합을 최소로 만드는 파라미터를 찾는 방법.
# Levenberg-Marquardt  비선형 최소제곱의 표준 알고리즘. leastsq 내부.
# unwrap           위상의 ±π 불연속을 제거해 연속 곡선으로 펴는 처리.
# 모멘트            데이터 분포의 퍼짐/치우침을 나타내는 통계량.
# 축퇴(degeneracy)  두 파라미터가 데이터상 구별되지 않는 상태.
#
# --- 코드 ---
# np.linalg.svd         특이값분해.
# np.roots              다항식의 모든 근.
# np.argmin / argmax    최소/최대값의 인덱스.
# np.unwrap             위상 펼치기.
# np.gradient           수치 미분.
# np.convolve           합성곱. 여기서는 이동평균에 사용.
# np.pad(mode='edge')   가장자리 값으로 배열을 늘림.
# spopt.newton          뉴턴법.
# spopt.leastsq         Levenberg-Marquardt 최소제곱.
# =========================================================================
