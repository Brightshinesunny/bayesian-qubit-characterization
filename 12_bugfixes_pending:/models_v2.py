"""
models.py  (수정본 v2)
=================================================================
"Forward model"(정방향 모델) 모듈: 파라미터를 넣으면 예측 신호를 돌려주는
물리 방정식을 모아두는 곳.

[함수 목록 - 새로 업로드할 때 아래 목록으로 버전 확인]
  1. s21_anticrossing_model()        - avoided-crossing 기본 물리 모델 (①~④번 식)
  2. qubit_frequency_vs_flux()       - ①번 식만 뽑아둔 헬퍼
  3. anticrossing_flux()             - [NEW] ①번 식의 역함수. 반교차 flux 계산
  4. describe_operating_point()      - [NEW] 특정 flux의 식별 가능성 진단
  5. fano_lineshape_correction()     - [수정] Fano 비대칭. 아래 v1->v2 참조
  6. add_spurious_tls_dip()          - TLS로 인한 가짜 추가 딥

=================================================================
v1 -> v2 변경 이력 (중요)
=================================================================
[FANO-1] `fano_factor /= np.max(fano_factor)` 제거.
    (q+x)^2/(1+x^2) 은 x->±∞ 에서 이미 1 로 수렴한다. 여기에 최대값
    (x=1/q 에서 (1+q^2)... 정확히는 1+q^2)으로 다시 나누면 신호 전체가
    상수배로 줄어든다. q=2 일 때 최대값은 5.0 이므로 베이스라인이
    1 -> 0.2 로 눌렸다.
    이 상태에서 표준 모델(베이스라인 1)로 피팅하면, (f_r, g, kappa)
    어떤 조합으로도 5배 스케일 차이를 맞출 수 없다. 즉 "모델 오설정에
    의한 파라미터 편향"이 아니라 "애초에 맞출 수 없는 신호"가 된다.

[FANO-2] x 의 정의를 그리드 의존 -> 선폭(kappa) 정규화로 변경.
    v1: x = 2*(f-fc)/(f[-1]-f[0]) * len(f)/20
        -> n_freq 나 스캔 범위를 바꾸면 Fano 구조의 물리적 폭이 변한다.
           401 -> 801 로 바꾸면 폭이 절반이 되어 재현이 안 된다.
    v2: x = 2*(f-fc)/gamma   (gamma = 공진 선폭)

[FANO-3] 기본 구현을 곱셈 보정 -> 복소 위상(complex Qc) 방식으로 변경.
    곱셈 방식은 x=1/q 근처에서 factor 가 1+q^2 까지 올라가 |S21| > 1 이
    된다. 수동 소자에서 물리적으로 불가능하다.
    공진기 문헌의 표준(Khalil et al. 2012, Probst et al. 2015)은
    결합 Q 를 복소수로 두어 비대칭을 위상 phi 로 넣는다. 이러면
    |S21| <= 1 과 베이스라인 1 이 자동으로 보장된다.
    v1 방식도 `method='multiplicative'` 로 남겨두었다(비교용).

[FLUX] anticrossing_flux() 추가.
    fq_max=5.150, EC=0.250, f_r=5.000 이면 Φ=0 에서
      fq(0) = 5.150,  delta = 0.150 = 3.75g  -> dispersive 영역
    이다. 반교차가 아니다. 실제 반교차는 Φ ≈ ±0.1055.
    Φ=0 에서는 두 혼성 모드의 가중치가 0.941 / 0.059 로 갈려
    딥이 사실상 하나만 보이고, g 가 거의 식별되지 않는다.
    (g 는 splitting √(δ²+4g²) 과 mixing 을 통해서만 들어오는데
     δ≫g 이면 둘 다 δ 가 지배한다 -> g 는 데이터가 아니라 prior 가 정한다)
"""

import numpy as np
from scipy.optimize import brentq


# =========================================================
# 1. 기본 물리 모델
# =========================================================
def s21_anticrossing_model(f_grid, flux_val, f_r, g, kappa,
                           fq_max, EC, tau):
    """
    Avoided-crossing (큐빗-공진기 강결합) S21 모델. ①~④번 식.

    ① 자속에 따른 큐빗 주파수 변화 (SQUID 유효 조셉슨 에너지)
    ② 결합강도 g 에 의한 두 혼성 모드 (Jaynes-Cummings)
    ③ 믹싱 각도 (Hopfield 계수)
    ④ 두 로렌츠 곡선의 합

    Parameters
    ----------
    f_grid : array. 주파수 (GHz)
    flux_val : float. 자속 (Φ/Φ0)
    f_r, g, kappa : float. 추정 대상 파라미터 (GHz)
    fq_max, EC, tau : float. 사전 캘리브레이션 상수

    Returns
    -------
    복소수 배열. 예측 S21(f).
    """
    fq = (fq_max + EC) * np.sqrt(np.abs(np.cos(np.pi * flux_val))) - EC
    delta = fq - f_r

    splitting = np.sqrt(delta**2 + 4 * g**2)
    hybrid_plus = 0.5 * (f_r + fq + splitting)
    hybrid_minus = 0.5 * (f_r + fq - splitting)

    sin2_theta = 0.5 * (1.0 - delta / splitting)
    cos2_theta = 1.0 - sin2_theta

    # 1j 컨벤션 유지. 2j 로 바꾸면 kappa 가 HWHM 의 절반을 가리키게 됨.
    s21_mode1 = cos2_theta / (1.0 + 1j * (f_grid - hybrid_minus) / (kappa / 2.0))
    s21_mode2 = sin2_theta / (1.0 + 1j * (f_grid - hybrid_plus) / (kappa / 2.0))

    cable_delay = np.exp(-1j * 2 * np.pi * f_grid * tau)
    return (1.0 - (s21_mode1 + s21_mode2)) * cable_delay


def qubit_frequency_vs_flux(flux_val, fq_max, EC):
    """①번 식만 따로 뽑은 헬퍼."""
    return (fq_max + EC) * np.sqrt(np.abs(np.cos(np.pi * flux_val))) - EC


# =========================================================
# [NEW] 동작점 진단
# =========================================================
def anticrossing_flux(f_r, fq_max, EC):
    """
    fq(Φ) = f_r 을 만족하는 flux (반교차 지점)를 해석적으로 계산.

        (fq_max + EC)·√|cos(πΦ)| - EC = f_r
        √|cos(πΦ)| = (f_r + EC) / (fq_max + EC)
        Φ = ± arccos( ((f_r+EC)/(fq_max+EC))² ) / π

    Returns
    -------
    float 또는 None. |Φ| 값 (±로 대칭). 도달 불가면 None.

    Examples
    --------
    >>> anticrossing_flux(5.000, 5.150, 0.250)   # doctest: +SKIP
    0.1055...
    """
    ratio = (f_r + EC) / (fq_max + EC)
    if not (0.0 <= ratio <= 1.0):
        return None                      # 큐빗이 f_r 에 도달하지 못함
    cos_val = ratio ** 2
    return float(np.arccos(np.clip(cos_val, -1.0, 1.0)) / np.pi)


def describe_operating_point(flux_val, f_r, g, kappa, fq_max, EC, verbose=True):
    """
    이 flux 에서 (f_r, g, kappa) 를 실제로 식별할 수 있는지 진단.

    반교차에서 멀면 g 는 데이터에 거의 안 들어온다. 그 상태로 MCMC 를
    돌리면 g 추정값은 prior 가 정하게 되고, 그 결과의 "오차 %" 는
    모델 오설정의 크기로 해석할 수 없다.
    """
    fq = qubit_frequency_vs_flux(flux_val, fq_max, EC)
    delta = fq - f_r
    splitting = np.sqrt(delta**2 + 4 * g**2)
    hp = 0.5 * (f_r + fq + splitting)
    hm = 0.5 * (f_r + fq - splitting)
    sin2 = 0.5 * (1.0 - delta / splitting)
    cos2 = 1.0 - sin2

    # g 에 대한 splitting 의 감도. d(splitting)/dg = 4g/splitting 이고
    # 반교차(delta=0)에서 splitting=2g 이므로 최대값은 2. 0~1 로 정규화한다.
    dsplit_dg = 4.0 * g / splitting
    sensitivity = dsplit_dg / 2.0
    # 두 딥이 선폭으로 분해되는가
    resolved = splitting > kappa
    # 약한 쪽 모드가 실제로 보이는가. 이것이 식별 가능성의 1차 기준이다.
    weak_weight = min(cos2, sin2)

    info = {
        'flux': flux_val, 'fq': fq, 'delta': delta, 'delta_over_g': delta / g,
        'splitting': splitting, 'hybrid_minus': hm, 'hybrid_plus': hp,
        'weight_minus': cos2, 'weight_plus': sin2,
        'dsplitting_dg': dsplit_dg, 'g_sensitivity': sensitivity,
        'resolved': resolved, 'weak_weight': weak_weight,
    }

    if verbose:
        print(f"  [동작점 진단] flux = {flux_val:+.4f}")
        print(f"    fq = {fq:.4f} GHz,  delta = fq - f_r = {delta:+.4f} "
              f"= {delta/g:+.2f} g")
        print(f"    splitting = {splitting:.4f} GHz  (2g = {2*g:.4f}), "
              f"kappa = {kappa:.4f}")
        print(f"    hybrid_minus = {hm:.4f} (weight {cos2:.3f}),  "
              f"hybrid_plus = {hp:.4f} (weight {sin2:.3f})")
        print(f"    g 감도 = {sensitivity:.3f}  "
              f"(1.0 = 반교차 최대, 0 = g 가 splitting 에 안 들어옴)")

        warned = False
        if not resolved:
            print("    [경고] splitting < kappa. 두 딥이 선폭에 묻혀 "
                  "분해되지 않습니다.")
            warned = True
        if weak_weight < 0.15:
            print(f"    [경고] 약한 쪽 모드의 가중치가 {weak_weight:.3f} 입니다.")
            print("           딥이 사실상 하나만 보입니다. 그러면 관측량은")
            print("           '딥 위치 1개' 뿐인데 미지수는 (f_r, g) 2개이므로")
            print("           둘이 축퇴(degenerate)합니다. g 는 데이터가 아니라")
            print("           prior 가 정하게 되고, 그 오차%는 모델 오설정의")
            print("           크기로 해석할 수 없습니다.")
            print("           '딥 2개 찾기' 초기값 로직도 성립하지 않습니다.")
            warned = True
        if sensitivity < 0.5:
            print(f"    [경고] g 감도가 {sensitivity:.3f} 로 낮습니다.")
            warned = True
        if warned:
            fa = anticrossing_flux(f_r, fq_max, EC)
            if fa is not None:
                print(f"    -> 반교차 지점 Φ = ±{fa:.4f} 사용을 권장합니다.")
        else:
            print("    [OK] 두 딥이 분해되고 g 감도도 충분합니다.")

    return info


# =========================================================
# 실전 확장: 교과서 모델과의 괴리
# =========================================================
def fano_lineshape_correction(s21, f_grid, f_center, fano_q,
                              gamma=None, method='phase', tau=0.0):
    """
    Fano 비대칭을 반영.

    물리적 배경: 이상적 로렌츠 공진은 "공진기로 들어가는 경로가 하나"라는
    가정 위에 있다. 실제 배선에서는 공진기를 거치지 않고 새어나가는
    직접 경로(케이블 반사, 임피던스 부정합)가 섞이고, 두 경로가 간섭하면
    공진 모양이 좌우 비대칭으로 일그러진다.

    Parameters
    ----------
    s21 : complex array. 원래 신호
    f_grid : array. 주파수
    f_center : float. 공진 중심
    fano_q : float. 비대칭 계수. 클수록 대칭에 가까움
    gamma : float or None. 정규화에 쓸 선폭(GHz).
        None 이면 스캔 범위의 1/20 을 임시로 쓰지만, 재현성을 위해
        kappa 를 명시적으로 넘기는 것을 강력히 권장한다.
    tau : float. 케이블 지연(ns 단위, 모델과 동일 값).
        method='phase' 에서 필수. s21 에 이미 들어 있는 지연을 벗겨내야
        공진 성분만 회전시킬 수 있다. 넘기지 않으면 베이스라인이 1 이
        되지 않는다.
    method : {'phase', 'multiplicative'}
        'phase' (기본, 권장)
            결합 Q 를 복소수로 두는 표준 방식.
            s21 = 1 - L·exp(i·phi) 로, 공진 성분 L 만 위상 회전한다.
            공진에서 멀면 L -> 0 이므로 베이스라인 1 이 자동 보장된다.
            (날개 부분에서 |S21| 가 1 을 약간 넘을 수 있는데, 이는
             이 표준형이 원래 갖는 성질이며 v1 의 5배 스케일 붕괴와는
             성격이 다르다.)
            phi = -arctan(1/q).  q -> ∞ 면 phi -> 0 (대칭).
            gamma 는 이 방식에서 쓰이지 않는다.
        'multiplicative'
            v1 과 같은 곱셈 방식(단, /np.max 는 제거).
            문제점 두 가지가 남아 있으므로 비교용으로만 사용할 것:
              - x=1/q 근처에서 factor 가 1+q^2 까지 올라가 |S21| > 1 이 된다.
              - x->±∞ 에서 1 로 수렴하지만 수렴이 느리다. q=2, 스캔 끝에서
                |x|=20 이어도 factor 는 0.81 / 1.21 로 ±20% 어긋난다.
                즉 베이스라인이 완전히 평평해지지 않는다.

    Notes
    -----
    v1 의 두 버그:
      (a) `fano_factor /= np.max(fano_factor)` 로 신호 전체가
          1/(1+q^2) 배로 눌렸다 (q=2 -> 1/5).
      (b) x 가 len(f_grid) 와 스캔 범위에 의존해, 그리드를 바꾸면
          물리적 폭이 변했다.
    """
    if method == 'phase':
        # s21 = (1 - L)·D,  L = 공진 로렌츠 성분,  D = exp(-2πi f τ)
        #
        # [주의] 케이블 지연 D 를 먼저 벗겨내야 한다. 그냥 (1 - s21) 을
        # 쓰면 날개에서 L->0 이어도 (1 - D) 가 남아 진동하고,
        # 베이스라인이 1 이 되지 않는다(실측 1.36~1.39).
        #
        #   L        = 1 - s21/D
        #   s21_fano = (1 - L·exp(i·phi))·D
        phi = -np.arctan2(1.0, fano_q)
        D = np.exp(-1j * 2 * np.pi * f_grid * tau)
        L = 1.0 - s21 / D
        return (1.0 - L * np.exp(1j * phi)) * D

    elif method == 'multiplicative':
        if gamma is None:
            gamma = (f_grid[-1] - f_grid[0]) / 20.0
        x = 2.0 * (f_grid - f_center) / gamma
        fano_factor = (fano_q + x) ** 2 / (1.0 + x ** 2)
        # /np.max 하지 않는다. x->±∞ 에서 이미 1 로 수렴한다.
        return s21 * fano_factor

    else:
        raise ValueError(f"method must be 'phase' or 'multiplicative', "
                         f"got {method!r}")


def add_spurious_tls_dip(s21, f_grid, f_tls, coupling_strength, linewidth):
    """
    TLS(이준위계) 결합에 의한 가짜 추가 딥.

    초전도 칩 표면·유전체의 미세 결함이 작은 큐빗처럼 행동하여
    고정 주파수에서 약하게 결합한다. 이론 모델(①~④)에 없는
    좁은 추가 딥이 생기고, 이를 진짜 avoided-crossing 봉우리로
    착각하면 피팅이 엉뚱한 값으로 수렴한다.

    coupling_strength : 딥의 깊이 (0~1)
        NOTE: 0.4 는 베이스라인의 40% 를 먹는 매우 강한 TLS 다.
              실제 소자에서 흔한 값은 0.02~0.15 수준이므로,
              "현실적 지저분함" 을 보려면 그 범위도 함께 시험할 것.
    linewidth : 딥의 폭 (GHz). 보통 진짜 결합보다 훨씬 좁다.
    """
    if not (0.0 <= coupling_strength <= 1.0):
        raise ValueError(f"coupling_strength 는 0~1 이어야 합니다: "
                         f"{coupling_strength}")
    tls_dip = 1.0 - coupling_strength / (
        1.0 + 1j * (f_grid - f_tls) / (linewidth / 2.0))
    return s21 * tls_dip


# =========================================================
# 자가진단
# =========================================================
if __name__ == "__main__":
    expected_functions = [
        's21_anticrossing_model',
        'qubit_frequency_vs_flux',
        'anticrossing_flux',
        'describe_operating_point',
        'fano_lineshape_correction',
        'add_spurious_tls_dip',
    ]
    print("models.py v2 자가진단")
    print("-" * 60)
    all_ok = True
    for name in expected_functions:
        exists = name in dir()
        print(f"  {name:30s} : {'OK' if exists else '누락!!'}")
        all_ok = all_ok and exists
    print("-" * 60)
    print("함수 상태:", "정상 (v2)" if all_ok else "일부 누락")

    # ---- 동작점 점검 ----
    print()
    F_R, G, KAPPA = 5.000, 0.040, 0.030
    FQ_MAX, EC = 5.150, 0.250

    fa = anticrossing_flux(F_R, FQ_MAX, EC)
    print(f"반교차 flux: Φ = ±{fa:.4f}" if fa else "반교차 도달 불가")
    print()
    describe_operating_point(0.0, F_R, G, KAPPA, FQ_MAX, EC)
    print()
    if fa is not None:
        describe_operating_point(fa, F_R, G, KAPPA, FQ_MAX, EC)

    # ---- Fano 정규화 점검 ----
    print()
    print("Fano 베이스라인 점검 (v1 버그 재현 확인)")
    print("-" * 60)
    f = np.linspace(4.8, 5.2, 401)
    x_v1 = 2 * (f - F_R) / (f[-1] - f[0]) * len(f) / 20
    fac_v1 = (2.0 + x_v1) ** 2 / (1 + x_v1 ** 2)
    fac_v1_norm = fac_v1 / np.max(fac_v1)
    print(f"  v1 (/np.max 포함): 양끝 factor = "
          f"{fac_v1_norm[0]:.3f}, {fac_v1_norm[-1]:.3f}   <- 1 이어야 정상")
    print(f"  v1 최소값 = {fac_v1_norm.min():.5f} "
          f"at f = {f[np.argmin(fac_v1_norm)]:.4f} GHz  <- Fano 영점")
    fac_v2 = (2.0 + x_v1) ** 2 / (1 + x_v1 ** 2)
    print(f"  v2 (/np.max 제거): 양끝 factor = "
          f"{fac_v2[0]:.3f}, {fac_v2[-1]:.3f}")
