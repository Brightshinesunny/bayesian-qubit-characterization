"""
models.py
=================================================================
"Forward model"(정방향 모델) 모듈: 파라미터를 넣으면 예측 신호를 돌려주는
물리 방정식을 모아두는 곳입니다.

[템플릿 설계 원칙]
새로운 물리계(예: 다른 큐빗 구조, 다른 결합 방식)를 다루게 되면,
이 파일에 함수를 하나 더 추가하기만 하면 됩니다. 다른 모듈
(likelihood.py, mcmc_pipeline.py)은 "모델 함수를 인자로 받는" 형태로
설계되어 있어서, 이 파일을 바꿔도 나머지는 그대로 재사용됩니다.

인터페이스 규칙(새 모델을 추가할 때 지켜야 하는 약속):
  입력: (주파수 배열, 그 외 물리 파라미터들...)
  출력: 같은 길이의 복소수 배열 (예측되는 S21 값)
"""

import numpy as np


def s21_anticrossing_model(f_grid, flux_val, f_r, g, kappa,
                            fq_max, EC, tau):
    """
    Avoided-crossing (큐빗-공진기 강결합) S21 모델. ①~④번 식을 그대로 구현.

    물리적 배경:
      ① 자속(flux)에 따라 큐빗 주파수(fq)가 변함 (SQUID 구조의 유효
         조셉슨 에너지가 자속에 따라 cos 형태로 변하기 때문)
      ② 큐빗과 공진기가 결합강도 g로 상호작용하면, 원래 두 개의 독립된
         주파수(fq, fr) 대신 "혼성 모드"라 불리는 새로운 두 고유주파수
         (hybrid_plus, hybrid_minus)가 생김 (Jaynes-Cummings 모델의 결과)
      ③ 각 혼성 모드가 원래 큐빗/공진기 성분을 얼마나 섞고 있는지를
         나타내는 "믹싱 각도"(sin^2 theta, cos^2 theta)
      ④ 최종 측정되는 S21은 이 두 혼성 모드에 대응하는 두 로렌츠(Lorentzian)
         곡선의 합으로 나타남 (로렌츠 함수: 공진 현상의 표준적인 종모양 곡선)

    Parameters
    ----------
    f_grid : array. 주파수 스캔 값들 (GHz)
    flux_val : float. 이 slice의 자속 값 (Φ/Φ0)
    f_r : float. 공진기(bare resonator) 고유 주파수 (GHz) — 추정 대상 파라미터
    g : float. 큐빗-공진기 결합강도 (GHz) — 추정 대상 파라미터
    kappa : float. 공진기 감쇠율/선폭 (GHz) — 추정 대상 파라미터
    fq_max : float. 자속=0일 때 큐빗 최대 주파수 (GHz) — 사전 캘리브레이션 상수
    EC : float. 충전 에너지 (GHz) — 사전 캘리브레이션 상수
    tau : float. 케이블 지연 — 사전 캘리브레이션 상수

    Returns
    -------
    복소수 배열 (f_grid와 같은 길이). 예측되는 S21(f) 값.
    """
    # ①번 식: 자속에 따른 큐빗 주파수
    fq = (fq_max + EC) * np.sqrt(np.abs(np.cos(np.pi * flux_val))) - EC

    # 디튜닝: 큐빗과 공진기 주파수의 차이
    delta = fq - f_r

    # ②번 식: 결합에 의해 갈라지는 두 혼성 모드
    hybrid_plus = 0.5 * (f_r + fq + np.sqrt(delta**2 + 4 * g**2))
    hybrid_minus = 0.5 * (f_r + fq - np.sqrt(delta**2 + 4 * g**2))

    # ③번 식: 믹싱 각도(Hopfield 계수)
    sin2_theta = 0.5 * (1.0 - delta / np.sqrt(delta**2 + 4 * g**2))
    cos2_theta = 1.0 - sin2_theta

    # ④번 식: 두 로렌츠 함수의 합.
    # 1j (허수단위 컨벤션): 표준 정의를 따름 - 1 + i*(f-omega)/(kappa/2)
    # (주의: "2j"를 쓰면 kappa가 실제 물리적 HWHM의 절반을 가리키게 되어
    #  참값과 어긋나는 버그가 생김. 반드시 1j를 유지할 것.)
    s21_mode1 = cos2_theta / (1.0 + 1j * (f_grid - hybrid_minus) / (kappa / 2.0))
    s21_mode2 = sin2_theta / (1.0 + 1j * (f_grid - hybrid_plus) / (kappa / 2.0))

    # 케이블/회로에 의한 위상 지연 (크기에는 영향 없고 위상만 회전)
    cable_delay = np.exp(-1j * 2 * np.pi * f_grid * tau)

    return (1.0 - (s21_mode1 + s21_mode2)) * cable_delay


def qubit_frequency_vs_flux(flux_val, fq_max, EC):
    """
    ①번 식만 따로 뽑아둔 헬퍼(보조) 함수. 진단 스크립트 등에서
    "이 flux에서 큐빗 주파수가 얼마인가"만 알고 싶을 때 재사용.
    """
    return (fq_max + EC) * np.sqrt(np.abs(np.cos(np.pi * flux_val))) - EC


# =========================================================
# [신규 - TLS swap spectroscopy 전용 모델]
# 아래는 원래 s21_anticrossing_model(주파수 스캔 방식)과 물리는 같지만,
# 측정 방식이 다른 TLS 데이터(오늘 다룬 quadspec 파일)에 맞춘 버전입니다.
# =========================================================

def tls_avoided_crossing_population_model(f_qubit, f_TLS, g, gamma_decay=None):
    """
    [용어부터 정리]
      f_qubit : 큐빗의 순간 주파수(GHz). 오늘 데이터에서는 sweep2(플럭스
                진폭)를 qset.calibdat로 변환해서 얻은 값 - "빠른 축".
      f_TLS   : TLS(이준위계)의 공명 주파수(GHz). 오늘 데이터에서는
                게이트 전압(sweep1)에 따라 Stark 편이로 이동함 - "느린 축".
      g       : 큐빗-TLS 결합강도(GHz). avoided crossing에서 "최소
                간격(2g)"을 결정하는 핵심 파라미터 - 오늘 하루 종일
                MHz/V로 구하려 했던 결합세기와는 다른 개념입니다.
                결합세기(MHz/V)="전압을 바꿨을 때 TLS 주파수가 얼마나
                움직이는가"이고, 여기 g(GHz)="큐빗과 TLS가 얼마나
                강하게 에너지를 주고받는가"입니다. 서로 다른 물리량!
      gamma_decay : (선택) 혼성 모드의 감쇠율. 지금은 순수하게 위치만
                볼 거라 생략 가능.

    [물리 원리 - 왜 이런 식이 나오는가]
    큐빗과 TLS가 둘 다 "두 준위 시스템"이라, 서로 에너지를 주고받을 수
    있는 조건(공명, f_qubit≈f_TLS)에 가까워지면, 마치 두 개의 진자가
    실로 연결된 것처럼 "서로 밀어내며" 새로운 두 개의 고유 주파수
    (혼성모드, hybrid mode)를 만듭니다. 이게 avoided-crossing(회피교차)
    입니다 - 두 에너지 준위가 "만나지 않고 비켜간다"는 뜻.

    수식(Jaynes-Cummings 모델과 동일한 형태):
      delta = f_qubit - f_TLS   (디튜닝, detuning: 두 주파수의 차이)
      hybrid_plus  = (f_qubit+f_TLS)/2 + sqrt(delta^2 + 4g^2)/2
      hybrid_minus = (f_qubit+f_TLS)/2 - sqrt(delta^2 + 4g^2)/2

    delta=0(정확히 공명)일 때 두 혼성모드의 간격이 2g로 "최소"가 되고,
    delta가 커질수록(공명에서 멀어질수록) hybrid_plus->f_qubit(큰쪽),
    hybrid_minus->f_TLS(작은쪽)로 각각 원래 값에 점근합니다 - "쌍곡선"
    모양이 나오는 이유가 바로 이 sqrt(delta^2+4g^2) 항 때문입니다.

    Parameters
    ----------
    f_qubit : array or float. 큐빗 순간주파수(GHz) - 오늘 데이터의 sweep2 축
    f_TLS   : array or float. TLS 공명주파수(GHz) - 오늘 데이터의 sweep1(전압) 축에서
              Stark 편이로 결정됨. TLS 자체가 별도 파라미터(f_TLS0, gamma_stark)로
              전압의 함수임: f_TLS(V) = f_TLS0 + gamma_stark*V
    g : float. 큐빗-TLS 결합강도(GHz) - 오늘 최종적으로 추정하고 싶은 핵심 파라미터

    Returns
    -------
    hybrid_plus, hybrid_minus : 두 혼성모드의 주파수(GHz)
    """
    delta = f_qubit - f_TLS
    splitting = np.sqrt(delta**2 + 4*g**2)
    hybrid_plus = 0.5*(f_qubit + f_TLS) + 0.5*splitting
    hybrid_minus = 0.5*(f_qubit + f_TLS) - 0.5*splitting
    return hybrid_plus, hybrid_minus


def tls_stark_shifted_frequency(V, f_TLS0, gamma_stark, V0=0.0):
    """
    [용어] Stark 편이(Stark shift): 전기장(여기서는 게이트 전압)에 의해
    에너지 준위(공명 주파수)가 이동하는 현상. TLS는 전기 쌍극자 모멘트를
    가지고 있어서, 국소 전기장(전압으로 조절)에 민감하게 반응함.

    [V0 파라미터 - 왜 필요한가, 실측 데이터로 확인된 문제]
    f_TLS0을 "V=0에서의 TLS 주파수"로 정의하면, 실제 측정 전압
    범위(예: -57~-56V)가 V=0에서 아주 멀리 떨어져 있을 때 심각한
    문제가 생김: f_TLS0과 gamma_stark가 서로 극도로 얽혀버림(합성
    데이터 검증 결과 상관계수=1.0). 이유는 f_TLS(V)=f_TLS0+gamma*V를
    V=-56.5 근처에서 평가하면, gamma의 작은 변화가 f_TLS0 쪽으로
    "56.5배" 증폭되어 반영되기 때문 - 마치 다항식 회귀를 원점에서
    먼 구간에 적용할 때 생기는 수치적 불안정과 같은 원리.

    해결책: V0(측정 구간 중심, 예: -56.5)을 기준점으로 재정의:
        f_TLS(V) = f_TLS0 + gamma_stark * (V - V0)
    이러면 f_TLS0은 "V0에서의 TLS 주파수"가 되어 실제 측정 구간
    한가운데 위치하게 되고, gamma_stark와의 상관계수가 0에 가깝게
    떨어짐(검증 완료) - 두 파라미터가 서로 독립적으로 잘 추정됨.

    V           : 게이트 전압(V) - 오늘 데이터의 sweep1 축
    f_TLS0      : V=V0에서의 TLS 공명주파수(GHz) - 추정 대상
    gamma_stark : 결합세기(GHz/V) - 추정 대상
    V0          : 기준 전압(V). 측정 구간의 중앙값을 넣는 것을 권장
                  (예: 오늘 seg12는 V0=-56.5).

    Returns
    -------
    f_TLS(V) : 전압에 따른 TLS 공명주파수(GHz)
    """
    return f_TLS0 + gamma_stark * (V - V0)


def tls_swap_spectroscopy_2d_model(V_grid, f_qubit_grid, f_TLS0, gamma_stark, g,
                                     baseline, contrast, V0=0.0):
    """
    [최종 조립] 위 두 함수를 합쳐서, 오늘 데이터 형태(21 x 401 격자,
    전압 x 큐빗주파수)에 맞는 2차원 예측값(dispamp에 대응하는 신호)을
    만드는 함수.

    [핵심 아이디어 - 신호가 왜 이런 모양인가]
    측정 신호(dispamp)는 avoided-crossing 근처(delta≈0)에서 배경과
    다른 값(딥 또는 봉우리)을 보이고, 멀어질수록 배경(baseline)으로
    돌아갑니다. 이걸 "혼성모드 간격이 좁을수록(=공명에 가까울수록)
    신호가 강하게 변한다"는 형태로 모델링합니다 - 표준적인 로렌츠형
    접근(간단화를 위해 정확한 population dynamics 대신, 간격에 반비례
    하는 종모양 함수 사용).

    Parameters
    ----------
    V_grid, f_qubit_grid : 2D meshgrid (전압, 큐빗주파수) - 오늘 데이터의
                            (sweep1, sweep2)에 대응
    f_TLS0, gamma_stark   : TLS의 전압-주파수 관계 파라미터 (V0 기준)
    g                     : 큐빗-TLS 결합강도(GHz)
    baseline              : 공명에서 먼 곳의 배경 신호 레벨
    contrast              : 공명 지점에서 신호가 얼마나 변하는지(딥/피크 깊이)
    V0                    : 재중심화 기준 전압(V). 측정 구간 중앙값 권장

    Returns
    -------
    2D 배열 (V_grid, f_qubit_grid과 같은 shape) - 예측되는 dispamp 지도
    """
    f_TLS = tls_stark_shifted_frequency(V_grid, f_TLS0, gamma_stark, V0=V0)
    delta = f_qubit_grid - f_TLS
    splitting = np.sqrt(delta**2 + 4*g**2)
        # splitting이 최솟값(=2g)이 되는 지점이 바로 avoided-crossing의
        # "정점" - 이게 오늘 하루 종일 픽셀로 쫓아다닌 그 딥의 진짜 정체.
    signal = baseline - contrast * (2*g) / splitting
        # splitting이 2g에 가까울수록(공명 지점) 신호가 baseline에서
        # 가장 크게 벗어남. splitting이 커질수록(공명에서 멀수록)
        # 2g/splitting -> 0이 되어 signal -> baseline으로 수렴 - 물리적으로
        # 타당한 극한 거동.
    return signal


# =========================================================
# [용어 정리 - 물리 + 통계, 매번 첨부하기로 한 약속]
# =========================================================
"""
--- 물리 용어 ---
avoided-crossing(회피교차) : 두 시스템(여기선 큐빗과 TLS)의 에너지
    준위가 서로 가까워질 때, 직접 만나지 않고 밀어내며 갈라지는 현상.
    "닫힌 시스템은 같은 에너지를 가질 수 없다"는 양자역학적 원리의
    결과(준위 반발, level repulsion).
디튜닝(detuning, delta) : 두 주파수의 차이(f_qubit - f_TLS). 0이면
    "정확히 공명", 클수록 "멀리 벗어난" 상태.
결합강도 g(GHz 단위)     : 큐빗과 TLS가 에너지를 주고받는 속도. 공명
    지점에서 두 혼성모드 사이 간격이 정확히 2g가 됨 - 이게 avoided-
    crossing 실험에서 g를 직접 측정하는 표준적인 방법.
혼성모드(hybrid mode)    : 큐빗과 TLS가 결합해서 만드는 새로운 고유
    상태. 원래 두 시스템의 성질이 섞여 있음(믹싱).
Stark 편이               : 전기장에 의해 에너지 준위(주파수)가 이동
    하는 현상. 오늘 데이터에서는 게이트 전압이 TLS 주파수를 이동시킴.
결합세기 gamma_stark(MHz/V 단위) : 전압 1V당 TLS 주파수가 얼마나
    움직이는지. g(GHz, 큐빗-TLS 결합)와는 완전히 다른 물리량이니
    혼동 주의 - 이름이 비슷해서 헷갈리기 쉬움.
swap spectroscopy        : 큐빗 주파수를 스캔하며 TLS와의 avoided-
    crossing을 찾는 표준 실험 기법.

--- 통계/모델링 용어 ---
forward model(정방향 모델) : 파라미터를 넣으면 예측 신호를 내놓는
    함수. 우리가 최종적으로 실측 데이터와 비교할 대상.
극한 거동(limiting behavior) 검증 : 파라미터를 극단적인 값(예: 디튜닝
    아주 큼)으로 뒀을 때, 모델이 물리적으로 말이 되는 값에 수렴하는지
    미리 확인하는 절차. 오늘도 이 검증을 먼저 통과시킨 뒤에야 실측
    데이터에 적용하기로 함.
2D 격자 피팅              : 1D 궤적(오늘 하루 종일 픽셀로 추적한 것)
    대신, 전압 x 주파수 2차원 지도 전체를 한 번에 모델과 비교하는 방식.
    픽셀 하나씩 추적할 필요가 없어 계통오차/교차점 문제에서 자유로움.
"""


