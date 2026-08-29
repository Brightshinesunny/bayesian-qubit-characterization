"""
fano_loader.py
=================================================================
Zenodo 10.5281/zenodo.7767046 ("Fano Interference in Microwave
Resonator Measurements", Rieger & Guenzler et al., KIT 2023)의
resonator_*_powersweep_*.npz 파일을 읽는 로더.

[파일 포맷 - 어제 .txt 포맷과의 차이]
어제 다룬 Zenodo 데이터(.txt)는 I(실수부)/Q(허수부)를 따로 저장했지만,
이 데이터셋은 numpy의 표준 압축 포맷(.npz)에 amplitude(진폭)/phase(위상)
형태로 저장되어 있습니다. 구조는 다음과 같습니다 (실제 확인된 값):

  amplitude : (n_power, n_freq) 형태, 선형 스케일 진폭 |S21|
  phase     : (n_power, n_freq) 형태, 위상 (라디안)
  frequency : (n_freq,) 형태, 주파수 축 (Hz 단위 - GHz 아님에 주의)
  power     : (n_power,) 형태, 입력 전력 (dBm)
  settings  : VNA(벡터 네트워크 분석기) 측정 세팅이 JSON 문자열로 저장된
              객체 배열. 물리 파라미터가 아니라 "이 측정을 어떻게
              했는지"에 대한 메타데이터 - 그중 port1_edel(케이블 전기
              지연, electrical delay)이 어제 계속 다룬 tau와 동일한
              물리량이라 특히 중요.

[왜 npz가 어제 .txt보다 다루기 쉬운가]
numpy의 표준 포맷이라 "빈 줄이 몇 개인지" 같은 파일 구조 규칙을 몰라도
np.load() 한 번으로 바로 배열이 나옵니다. 다만 settings 항목만 numpy
표준 숫자 타입이 아니라 "파이썬 객체"(JSON 문자열)라서 allow_pickle=True
옵션이 필요합니다.
"""

import numpy as np
import json


def load_fano_npz(filepath):
    """
    resonator_*_powersweep_*.npz 파일 하나를 읽어서 구조화된 딕셔너리로 반환.

    Returns
    -------
    dict: {
      'power_dbm'   : 1D array (n_power,) - 입력 전력 (dBm)
      'freq_hz'     : 1D array (n_freq,)  - 주파수 축 (Hz)
      'freq_ghz'    : 1D array (n_freq,)  - 주파수 축을 GHz로 환산한 것
                       (오늘까지 계속 GHz 단위로 다뤄왔으므로 편의상 추가)
      'amplitude'   : 2D array (n_power, n_freq) - |S21| 진폭
      'phase'       : 2D array (n_power, n_freq) - S21 위상 (라디안)
      's21'         : 2D array (n_power, n_freq) - 복소수로 재구성한 S21
                       (= amplitude * exp(i*phase))
      'settings'    : dict - VNA 측정 세팅 (JSON을 파싱한 딕셔너리)
      'cable_delay' : float - settings 안의 port1_edel 값 (초 단위).
                       장비가 직접 측정해 알려주는 케이블 지연으로,
                       어제 MCMC로 애먹었던 tau를 여기서는 곧바로
                       고정값으로 쓸 수 있게 해주는 값.
    }
    """
    raw = np.load(filepath, allow_pickle=True)
        # allow_pickle=True: settings 항목이 순수 숫자 배열이 아니라
        # 파이썬 객체(문자열)를 담고 있어서 필요한 옵션. 신뢰할 수 있는
        # 출처(공식 Zenodo 데이터셋)의 파일에서만 이 옵션을 쓰는 것이 안전.

    amplitude = raw['amplitude']
    phase = raw['phase']
    freq_hz = raw['frequency']
    power_dbm = raw['power']

    s21 = amplitude * np.exp(1j * phase)
        # 진폭과 위상으로부터 복소수 S21을 재구성.
        # np.exp(i*phase): 오일러 공식(e^(iθ) = cosθ + i·sinθ)에 따라
        # "위상 θ만큼 회전된 단위 복소수"를 만들고, 여기에 진폭을 곱하면
        # 극좌표(진폭,위상) 표현이 직교좌표(실수부,허수부) 복소수로 바뀜.

    # settings는 (1,) 크기의 object 배열 안에 JSON 문자열이 들어있는 구조.
    # raw['settings'][0]으로 그 문자열을 꺼낸 뒤 json.loads로 파싱.
    settings_str = raw['settings'][0]
    settings_dict = json.loads(settings_str)
        # json.loads: JSON 형식의 문자열을 파이썬 딕셔너리로 변환하는 표준 함수.
        # settings_str이 '{"vna": {...}}' 형태이므로, 파싱하면
        # settings_dict['vna']['port1_edel'] 같은 식으로 접근 가능해짐.

    cable_delay = settings_dict.get('vna', {}).get('port1_edel', None)
        # dict.get(key, default): 키가 없어도 에러 없이 default를 반환하는
        # 안전한 딕셔너리 접근 방법. 혹시 다른 파일에서 'vna' 항목 이름이나
        # 구조가 조금 다르더라도 바로 에러가 나지 않고 None을 반환하게 함.

    return {
        'power_dbm': power_dbm,
        'freq_hz': freq_hz,
        'freq_ghz': freq_hz / 1e9,
            # Hz -> GHz 변환. 오늘까지 계속 GHz 단위로 다뤄왔으므로
            # 통일성을 위해 미리 변환해서 같이 제공.
        'amplitude': amplitude,
        'phase': phase,
        's21': s21,
        'settings': settings_dict,
        'cable_delay': cable_delay,
    }


def summarize_loaded_data(data):
    """로드된 데이터의 형태(shape)와 핵심 세팅값을 요약 출력하는 도우미"""
    print("로드된 데이터 요약:")
    print("-" * 55)
    for key in ['power_dbm', 'freq_hz', 'freq_ghz', 'amplitude', 'phase', 's21']:
        val = data[key]
        print(f"  {key:12s}: shape={val.shape}, "
              f"범위=[{np.abs(val).min():.4g}, {np.abs(val).max():.4g}]")
    print("-" * 55)
    print(f"  cable_delay (port1_edel) = {data['cable_delay']} 초")
    vna = data['settings'].get('vna', {})
    print(f"  centerfreq = {vna.get('centerfreq', 'N/A')} Hz")
    print(f"  span       = {vna.get('span', 'N/A')} Hz")
    print(f"  averages   = {vna.get('averages', 'N/A')}")
    print("-" * 55)


if __name__ == "__main__":
    # models.py, zenodo_loader.py와 동일한 패턴의 자가진단 코드.
    expected_functions = ['load_fano_npz', 'summarize_loaded_data']
    print("fano_loader.py 자가진단: 기대되는 함수들이 모두 있는지 확인")
    print("-" * 55)
    all_ok = True
    for name in expected_functions:
        exists = name in dir()
        status = "OK" if exists else "누락!! -> 구버전 파일일 수 있음"
        print(f"  {name:28s} : {status}")
        all_ok = all_ok and exists
    print("-" * 55)
    print("전체 상태:", "정상 (최신 버전)" if all_ok else "일부 함수 누락")

    import sys
    if len(sys.argv) > 1:
        data = load_fano_npz(sys.argv[1])
        summarize_loaded_data(data)
