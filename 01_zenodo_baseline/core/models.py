"""
models.py
=================================================================
"Forward model"(정방향 모델) 모듈: 파라미터를 넣으면 예측 신호를 돌려주는
물리 방정식을 모아두는 곳입니다.

[이 파일에 포함된 함수 목록 - 새로 업로드할 때 아래 목록으로 버전 확인]
  1. s21_anticrossing_model()        - avoided-crossing 기본 물리 모델 (①~④번 식)
  2. qubit_frequency_vs_flux()       - ①번 식만 뽑아둔 헬퍼(보조) 함수
  3. fano_lineshape_correction()     - Fano 비대칭 lineshape 왜곡 추가
  4. add_spurious_tls_dip()          - TLS로 인한 가짜 추가 딥 왜곡 추가
  (Colab에서 `import models; print(dir(models))`로 이 4개 함수가 모두
   보이는지 확인하면, 구버전 파일을 잘못 쓰고 있는 실수를 예방할 수 있음)

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
# 실전 확장: 실제 실험에서 흔히 마주치는 세 가지 "교과서 모델과의 괴리"
# =========================================================
# 지금까지의 s21_anticrossing_model은 "완벽하게 이상적인" 이론값입니다.
# 실제 측정에서는 아래 세 가지 현상이 추가로 섞여 들어오는 경우가 매우
# 흔합니다. mock_data.py의 잡음(백색/1f/드리프트)이 "측정 과정에서
# 생기는 무작위성"이라면, 아래 함수들은 "측정 대상 자체가 교과서 모델과
# 다르게 행동하는 것"이라는 점에서 성격이 다릅니다 (물리적 현상 vs 잡음).


def fano_lineshape_correction(s21, f_grid, f_center, fano_q):
    """
    Fano 비대칭(Fano asymmetry)을 기존 S21 신호에 곱해서 반영.

    물리적 배경: 이상적인 로렌츠 공진(대칭적인 종모양 딥)은 "공진기로
    들어가는 경로가 단 하나"라는 가정 위에 있습니다. 그런데 실제 배선에서는
    신호가 공진기를 거치지 않고 직접 새어나가는 경로(케이블 반사, 임피던스
    부정합 등)가 항상 조금씩 섞입니다. 이 "직접 경로"와 "공진기를 거친
    경로" 두 개가 간섭하면, 공진 모양이 좌우 비대칭으로 일그러지는데
    이를 Fano 공명(Fano resonance)이라 부릅니다 (원래는 원자물리학에서
    이산 상태와 연속 상태의 간섭을 설명하기 위해 도입된 개념).

    Fano 비대칭 계수(Fano factor, q):
        q -> 무한대: 원래의 대칭적 로렌츠 모양 (직접 경로가 거의 없음)
        q가 유한한 값(예: 1~5): 한쪽으로 기울어진 비대칭 딥/피크
        q -> 0: 완전히 뒤집힌 모양 (딥이 피크처럼 보임)

    표준적인 Fano 공식(정규화된 형태):
        F(x) = (q + x)^2 / (1 + x^2),  x = 2(f - f_center)/gamma
    이 함수는 이 비대칭 인자를 원래 신호에 곱하는 방식으로 근사 적용합니다
    (엄밀한 유도는 각 실험 배선의 산란행렬 이론에 따라 달라지지만, 여기서는
    "비대칭이 생긴다"는 정성적 효과를 재현하는 데 목적을 둠).
    """
    # 공진 중심에서 얼마나 떨어져 있는지를 무차원화(정규화)한 변수.
    # (f_grid - f_center)가 클수록 공진에서 먼 지점이라는 뜻.
    x = 2 * (f_grid - f_center) / (f_grid[-1] - f_grid[0]) * len(f_grid) / 20
        # 스캔 범위 대비 상대적인 스케일로 x를 정의 (구체적인 감쇠폭 대신
        # 스캔 그리드 전체 폭의 일부를 기준 삼아 근사)
    fano_factor = (fano_q + x) ** 2 / (1 + x ** 2)
    fano_factor = fano_factor / np.max(fano_factor)   # 최대값 1로 정규화 (진폭 스케일 보존)
    return s21 * fano_factor


def add_spurious_tls_dip(s21, f_grid, f_tls, coupling_strength, linewidth):
    """
    준입자/TLS(Two-Level System, 이준위계) 결합으로 인한 "가짜" 추가 딥을
    기존 신호에 곱해서 반영.

    물리적 배경: 초전도 칩 표면이나 유전체 층에는 원치 않는 미세한
    결함(defect)들이 존재하는데, 이들이 마치 작은 큐빗처럼 행동하며
    (Two-Level System, TLS) 특정 고정 주파수에서 원래 공진기/큐빗과
    약하게 결합할 수 있습니다. 이 결합이 생기면, avoided-crossing
    스펙트럼에 원래 이론(①~④번 식)에는 없는 "예상 못 한 좁은 추가 딥"이
    특정 flux 근처에서만 나타납니다.

    실전에서 중요한 이유: 이런 TLS 딥을 모델이 예측하는 avoided-crossing
    봉우리로 착각하면, 피팅이 완전히 엉뚱한 파라미터로 수렴할 수 있습니다.
    실제 데이터 분석에서 "이 이상한 딥이 물리적으로 의미 있는 신호인지,
    아니면 TLS 같은 결함 신호인지"를 구분하는 것이 중요한 실무 판단입니다.

    f_tls : TLS의 (거의 고정된) 공진 주파수
    coupling_strength : TLS와의 결합 세기 (딥의 깊이를 결정)
    linewidth : TLS 딥의 폭 (보통 진짜 큐빗-공진기 결합보다 훨씬 좁음 -
                TLS는 결맞음 시간이 짧아 선폭이 넓을 수도, 반대로 매우
                좁고 날카로운 경우도 있어 실험마다 다름)
    """
    # 단순 로렌츠 딥 하나를 추가로 곱함 (기존 avoided-crossing 모델과
    # 독립적으로, "국소적으로만 영향을 주는" 좁은 흡수를 표현)
    tls_dip = 1.0 - coupling_strength / (1.0 + 1j * (f_grid - f_tls) / (linewidth / 2.0))
    return s21 * tls_dip


# =========================================================
# [템플릿 확장 지점] 새로운 물리계를 추가하려면 여기에 함수를 더 만드세요.
# 예시:
#
# def s21_dispersive_readout_model(f, f0, kappa, chi):
#     """분산 영역(dispersive regime) 판독 공진기 모델 - 단일 로렌츠 + chi shift"""
#     denominator = (kappa / 2.0) + 1j * (2.0 * np.pi * (f - (f0 + chi)))
#     return 1.0 - (kappa / denominator)
#
# def s21_three_qubit_model(f, ...):
#     """3-큐빗 시스템으로 확장할 때 추가할 모델"""
#     ...
# =========================================================


if __name__ == "__main__":
    # 이 파일을 직접 실행했을 때만 동작하는 자가진단 코드
    # (다른 파일에서 import models로 불러올 때는 실행되지 않음 -
    #  파이썬에서 "이 파일이 메인으로 실행됐는지, 남이 불러다 쓴 것인지"
    #  구분하는 표준적인 관용구)
    #
    # 사용법: Colab에서 `!python models.py` 또는 이 블록만 복사해서 실행하면,
    # 지금 로드된 models.py에 어떤 함수들이 들어있는지 즉시 확인 가능.
    # "구버전 파일을 잘못 쓰고 있는" 실수를 빠르게 잡아내기 위한 장치.
    expected_functions = [
        's21_anticrossing_model',
        'qubit_frequency_vs_flux',
        'fano_lineshape_correction',
        'add_spurious_tls_dip',
    ]
    print("models.py 자가진단: 기대되는 함수들이 모두 있는지 확인")
    print("-" * 55)
    all_ok = True
    for name in expected_functions:
        exists = name in dir()
        status = "OK" if exists else "누락!! -> 구버전 파일일 수 있음"
        print(f"  {name:28s} : {status}")
        all_ok = all_ok and exists
    print("-" * 55)
    print("전체 상태:", "정상 (최신 버전)" if all_ok else "일부 함수 누락 - 파일을 새로 업로드하세요")
