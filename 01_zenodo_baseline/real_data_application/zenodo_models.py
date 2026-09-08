"""
zenodo_models.py
=================================================================
실제 Zenodo 데이터(10.5281/zenodo.10518320, Sett et al. 2024,
"Emergent Macroscopic Bistability Induced by a Single Superconducting
Qubit")를 분석하기 위한 물리 모델 모듈입니다.

[왜 models.py와 별도 파일로 분리했는가]
models.py는 avoided-crossing(①~④번 식) 물리계를 다루는 "mock 데이터
샌드박스" 용도로 충분히 검증되어 안정된 상태입니다. 반면 이 파일은
"실제 측정 데이터를 새로 분석하기 시작한" 별도의 단계이고, 앞으로
이 데이터셋을 더 파고들면서 관련 함수(예: 비선형 Duffing 공진기 모델,
쌍안정성 모델 등)가 계속 늘어날 가능성이 있습니다. 두 물리계는 측정
방식 자체가 다르므로(flux 스캔 vs 전력 스캔, reflection 계열 vs
2-포트 transmission), 한 파일에 계속 몰아넣으면 "검증이 끝난 안정된
템플릿"과 "이제 막 시작한 실험적 분석"이 뒤섞여 헷갈리기 쉽습니다.

[이 파일에 포함된 함수 목록]
  1. s21_single_resonance_transmission() - 단일 공진기 투과 S21 모델
  2. power21_single_resonance()          - 위 모델의 |S21|^2 버전
"""

import numpy as np


# =========================================================
# 단일 공진기 투과(transmission) 모델
# (Zenodo 10.5281/zenodo.10518320, Sett et al. 데이터셋 대응)
# =========================================================
# 이 모델은 models.py의 avoided-crossing(①~④번 식)과는 다른
# 물리계입니다. 큐빗-공진기 결합이 아니라, 공진기 하나를 "두 개의
# 외부 포트로 신호가 드나들고, 약간의 내부 손실이 있는" 표준적인
# 2-포트 투과 공진 회로로 봅니다. 원저자(Sett et al., 2024)의 분석
# 코드에 있던 정확한 수식을 그대로 재구현한 것으로, 논문이
# Mathematica로 피팅해 공개한 참값과 우리 파이프라인의 결과를 직접
# 대조 검증하는 데 사용합니다.
#
# 물리적 의미:
#   ke1, ke2 : 공진기가 두 개의 외부 전송선(포트1, 포트2)과 결합하는
#              세기(external coupling rate). 신호가 포트1로 들어와
#              공진기를 거쳐 포트2로 빠져나가는 "투과(transmission)"
#              측정 방식이라 두 개의 결합률이 따로 필요함.
#   ki       : 공진기 자체의 내부 손실률(internal loss rate). 재질
#              결함, 유전체 손실 등으로 에너지가 그냥 사라지는 정도.
#   total kappa = ke1 + ke2 + ki : 총 감쇠율(models.py에서 다뤘던
#              단일 kappa에 대응하는 값. 이 논문에서는 세 성분으로
#              분해해서 "신호가 얼마나 잘 통과하는지"와 "내부에서
#              얼마나 새는지"를 구분함).
#   A0, phi  : 전체 신호의 진폭/위상 오프셋 (증폭기 이득, 케이블 위상
#              등 실험 장비 자체의 배경 특성을 흡수하는 보정 상수).
#   tau      : 케이블 지연 (models.py의 cable_delay와 같은 역할).
def s21_single_resonance_transmission(f, f0, ke1, ke2, ki, A0, phi, tau):
    """
    단일 공진기의 투과 스펙트럼 모델 (Sett et al. 원저자 코드의
    S21fullassy 함수를 그대로 재구현).

        S21(f) = A0 * exp(-i(phi + pi/2 + 2*pi*f*tau))
                 * sqrt(ke1*ke2) / ((ke1+ke2+ki)/2 + i*(f-f0))

    models.py의 s21_anticrossing_model과의 핵심 차이:
      - 그쪽 모델: "1 - (로렌츠항)" 형태 → 공진에서 아래로 파인 딥(dip)
      - 이 모델  : 로렌츠항을 그대로 분자에 둠 → 공진에서 위로 솟은
                   피크(peak) 형태 (분모가 최소가 되는 f=f0 근처에서
                   |S21|이 최대가 됨)
    """
    denom = (ke1 + ke2 + ki) / 2.0 + 1j * (f - f0)
    return A0 * np.exp(-1j * (phi + np.pi / 2 + 2 * np.pi * f * tau)) * np.sqrt(ke1 * ke2) / denom


def power21_single_resonance(f, f0, ke1, ke2, ki, A0, phi, tau):
    """
    |S21|^2 (측정에서 흔히 "파워" 단위로 다루는 양). 원저자 코드의
    P21fullassy에 대응. amplitude로 정규화하면 공진점에서 정확히
    1(또는 그 근처)이 되도록 논문에서 amplitude 상수로 나눠서 사용함
    (단, ke1≠ke2로 비대칭이면 최대 정규화값이 1보다 작아짐 -
     4*ke1*ke2/(ke1+ke2+ki)^2 라는 "임피던스 정합 계수"가 곱해지기
     때문. 이는 버그가 아니라 두 포트 결합이 비대칭일 때 나타나는
     정상적인 물리 현상임 - 실제로 kappa=8MHz 데이터로 검증했을 때
     이 계수가 정확히 원저자 참값과 일치함을 확인함).
    """
    s21 = s21_single_resonance_transmission(f, f0, ke1, ke2, ki, A0, phi, tau)
    return np.abs(s21) ** 2


if __name__ == "__main__":
    # models.py, mock_data.py와 동일한 패턴의 자가진단 코드.
    expected_functions = [
        's21_single_resonance_transmission',
        'power21_single_resonance',
    ]
    print("zenodo_models.py 자가진단: 기대되는 함수들이 모두 있는지 확인")
    print("-" * 55)
    all_ok = True
    for name in expected_functions:
        exists = name in dir()
        status = "OK" if exists else "누락!! -> 구버전 파일일 수 있음"
        print(f"  {name:35s} : {status}")
        all_ok = all_ok and exists
    print("-" * 55)
    print("전체 상태:", "정상 (최신 버전)" if all_ok else "일부 함수 누락 - 파일을 새로 업로드하세요")
