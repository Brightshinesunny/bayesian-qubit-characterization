"""
zenodo_loader.py
=================================================================
Zenodo 10.5281/zenodo.10518320 ("Emergent Macroscopic Bistability
Induced by a Single Superconducting Qubit") 데이터셋의
S21_power_sweep_*.txt 파일을 읽는 로더.

[파일 포맷 설명]
파일은 "%%%%%%%%%% <섹션이름>" 로 구분된 여러 블록으로 구성됨:
  - "R_power_source ~dBm" : 1줄, 입력 마이크로파 전력값들 (dBm 단위)
                            예: -50.0, -49.0, ..., 10.0 (총 n_power개)
  - "~Freq. trace GHz"    : 1줄, 주파수 스캔 축 (GHz 단위, 총 n_freq개)
  - "I"                   : n_power줄, 각 줄은 그 전력에서의 S21 실수부
                            (In-phase 성분), 각 줄에 n_freq개 값
  - "Q"                   : (있다면) n_power줄, S21 허수부(Quadrature)

[왜 이 데이터가 오늘까지 다룬 avoided-crossing 모델과 다른가]
이 데이터셋은 flux(자속)를 스캔한 게 아니라, 하나의 공진기에 넣는
마이크로파 "전력(power)"을 스캔한 것입니다. 전력이 세지면 공진기 안의
광자 수가 늘어나면서, 큐빗-공진기 시스템이 비선형(Duffing 진동자와
유사한) 방식으로 행동하게 되어 공진 주파수가 이동하거나 쌍안정성
(bistability, 같은 조건에서 두 개의 서로 다른 안정 상태가 존재)이
나타날 수 있습니다. 이건 ①~④번 식(flux에 따른 avoided-crossing)과는
완전히 다른 물리 현상이므로, 그대로 s21_anticrossing_model에 넣어
피팅하면 안 됩니다 - 먼저 데이터 형태를 보고 어떤 모델이 맞는지
판단해야 합니다.
"""

import numpy as np
import re


def parse_section_header(line):
    """
    "%%%%%%%%%% 섹션이름" 형태의 줄에서 섹션 이름만 추출.
    re.match: 정규표현식(regular expression)으로 문자열 패턴을 찾는 함수.
    여기서는 앞의 %기호들과 공백을 건너뛰고 뒤의 텍스트를 뽑아냄.
    """
    m = re.match(r'%+\s*(.+)', line.strip())
        # %+  : % 문자가 1개 이상 반복
        # \s* : 공백이 0개 이상
        # (.+): 그 뒤의 나머지 문자열을 그룹으로 캡처 (섹션 이름)
    return m.group(1).strip() if m else None


def load_power_sweep_txt(filepath):
    """
    S21_power_sweep_*.txt 파일 하나를 읽어서 구조화된 딕셔너리로 반환.

    Returns
    -------
    dict: {
      'power_dbm': 1D array (n_power,) - 입력 전력값들 (dBm)
      'freq_ghz' : 1D array (n_freq,)  - 주파수 축 (GHz)
      'I' : 2D array (n_power, n_freq) - 실수부
      'Q' : 2D array (n_power, n_freq) - 허수부 (섹션이 있을 때만)
    }
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # 파일을 "%%%%%%%%%%"로 시작하는 줄을 경계로 여러 섹션으로 분리
    sections = {}   # {섹션이름: [그 섹션에 속한 데이터 줄들]}
    current_section = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('%'):
            # 새 섹션 시작 - 섹션 이름을 파싱해서 딕셔너리에 새 리스트 준비
            current_section = parse_section_header(stripped)
            if current_section is not None:
                sections[current_section] = []
        elif stripped == '':
            # 빈 줄은 섹션 안에서도 등장할 수 있음(I 블록의 각 전력별 줄
            # 사이 구분자로 보임) - 그냥 건너뜀. 실제 데이터 파싱은
            # "숫자로 시작하는 줄"만 골라내는 방식으로 하므로 안전함.
            continue
        elif current_section is not None:
            sections[current_section].append(stripped)

    def parse_number_row(line_str):
        """탭/공백으로 구분된 숫자 문자열 한 줄을 float 배열로 변환"""
        return np.array([float(x) for x in line_str.split()])
            # .split() : 인자 없이 쓰면 탭/공백/개행 등 모든 공백류 문자를
            # 구분자로 삼아 문자열을 쪼갬 (구분자가 탭인지 스페이스인지
            # 몰라도 알아서 처리되는 편리한 방식)

    result = {}

    # 전력, 주파수는 섹션 안에 "한 줄"만 있는 구조 (첫 줄만 사용)
    power_key = next((k for k in sections if 'power_source' in k.lower()), None)
        # next(제너레이터, 기본값) : 조건을 만족하는 첫 번째 항목을 찾는
        # 파이썬 관용구. 정확한 섹션 이름을 몰라도 "power_source가
        # 포함된 이름"으로 유연하게 찾아냄 (파일마다 표기가 미세하게
        # 다를 수 있는 것에 대비).
    freq_key = next((k for k in sections if 'freq' in k.lower()), None)

    if power_key and sections[power_key]:
        result['power_dbm'] = parse_number_row(sections[power_key][0])
    if freq_key and sections[freq_key]:
        result['freq_ghz'] = parse_number_row(sections[freq_key][0])

    # I, Q는 섹션 안에 "여러 줄"(전력값 하나당 한 줄)이 있는 구조
    for iq_name in ['I', 'Q']:
        if iq_name in sections and sections[iq_name]:
            rows = [parse_number_row(row) for row in sections[iq_name]]
            result[iq_name] = np.array(rows)
                # np.array(리스트의 리스트) -> 2D 배열로 자동 변환.
                # 만약 각 줄의 길이가 다르면 여기서 에러가 나므로,
                # 그 자체가 "데이터가 예상과 다르게 생겼다"는 진단이 됨.

    return result


def summarize_loaded_data(data):
    """로드된 데이터의 형태(shape)를 출력해 구조를 빠르게 확인하는 도우미"""
    print("로드된 데이터 요약:")
    print("-" * 50)
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            print(f"  {key:12s}: shape={val.shape}, "
                  f"범위=[{val.min():.4g}, {val.max():.4g}]")
    print("-" * 50)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data = load_power_sweep_txt(sys.argv[1])
        summarize_loaded_data(data)
