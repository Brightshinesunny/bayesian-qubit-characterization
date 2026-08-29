"""
fano_fieldsweep_loader.py
=================================================================
resonator_7_fieldsweep_overcoupled.npz 전용 로더. 지금까지 쓴
fano_loader.py와 결정적으로 다른 점은 "frequency가 2차원"이라는 것 -
자기장(B_fields) slice마다 VNA가 서로 다른 주파수 창을 스캔합니다
(공진 주파수가 자기장에 따라 이동하니, 좁은 스캔 폭(4MHz)을 유지하며
그 이동을 "쫓아가는" 방식으로 측정한 것으로 보임 - 이런 방식을
"주파수 추적(frequency tracking) 스캔"이라 부릅니다).

[물리적 배경 - 용어 정리]
  평행(in-plane) 자기장 vs 수직(out-of-plane) 자기장:
    얇은 초전도 박막에서는 자기장의 "수직 성분"만 보텍스(자속
    소용돌이)를 만듭니다. 평행 성분은 박막을 거의 뚫지 못해
    보텍스를 훨씬 적게 만들고, 그래서 평행 방향 임계자기장이
    수직 방향보다 훨씬(보통 수백 배) 큽니다.
  벡터 마그넷(vector magnet): 여러 축의 코일로 자기장의 방향과
    크기를 독립적으로 제어하는 장치.
  이 데이터셋의 실험 설계: settings의 "coil_parr"(평행 성분으로
    추정)를 크게 스윕하면서, "coil"(수직 성분으로 추정)은 아주
    작은 값으로 계속 미세조정해 0 근처로 상쇄시킵니다 - "순수한
    평행 자기장 내성"만 골라서 측정하려는 정교한 설계로 보입니다.
"""

import numpy as np
import json


def load_fano_fieldsweep_npz(filepath):
    """
    field sweep 파일을 읽어서 구조화된 딕셔너리로 반환.

    Returns
    -------
    dict: {
      'B_fields'    : 1D array (n_field,) - 자기장 값들
      'freq_hz_2d'  : 2D array (n_field, n_freq) - 슬라이스마다 다른
                       주파수 축 (기존 파일들과 달리 1D가 아님에 주의!)
      'amplitude'   : 2D array (n_field, n_freq)
      'phase'       : 2D array (n_field, n_freq)
      's21'         : 2D array (n_field, n_freq) - 복소수로 재구성
      'settings_list': 길이 n_field인 리스트, 각 원소가 그 slice의
                        VNA/코일 세팅을 담은 딕셔너리
      'cable_delay_list': 각 slice의 port1_edel 값 리스트 (지금까지
                        확인한 바로는 전부 동일한 값일 가능성이 높지만,
                        혹시 slice마다 다를 경우에 대비해 리스트로 보관)
    }
    """
    raw = np.load(filepath, allow_pickle=True)

    amplitude = raw['amplitude']
    phase = raw['phase']
    freq_hz_2d = raw['frequency']
        # [주의] 이 파일은 frequency가 (n_field, n_freq) 2차원입니다.
        # 지금까지 쓴 fano_loader.py는 frequency가 1차원(모든 slice가
        # 같은 주파수 축 공유)이라고 가정했으므로, 이 파일에는 그
        # 로더를 그대로 쓰면 안 되고 이 전용 로더를 써야 합니다.
    B_fields = raw['B_fields']

    s21 = amplitude * np.exp(1j * phase)

    settings_list = []
    cable_delay_list = []
    for i in range(raw['settings'].shape[0]):
        settings_str = raw['settings'][i, 0]
            # settings의 shape이 (n_field, 1)이므로, i번째 slice의
            # 문자열은 raw['settings'][i, 0]으로 접근 (기존 파일의
            # (n_power,) 형태와 인덱싱이 다름에 주의).
        settings_dict = json.loads(settings_str)
        settings_list.append(settings_dict)
        cable_delay_list.append(settings_dict.get('vna', {}).get('port1_edel', None))

    return {
        'B_fields': B_fields,
        'freq_hz_2d': freq_hz_2d,
        'amplitude': amplitude,
        'phase': phase,
        's21': s21,
        'settings_list': settings_list,
        'cable_delay_list': cable_delay_list,
    }


def summarize_loaded_data(data):
    """로드된 데이터 요약 출력"""
    print("로드된 데이터 요약 (field sweep):")
    print("-" * 55)
    print(f"  B_fields   : shape={data['B_fields'].shape}, "
          f"범위=[{data['B_fields'].min():.4g}, {data['B_fields'].max():.4g}]")
    print(f"  freq_hz_2d : shape={data['freq_hz_2d'].shape}")
    print(f"  amplitude  : shape={data['amplitude'].shape}, "
          f"범위=[{data['amplitude'].min():.4g}, {data['amplitude'].max():.4g}]")
    print(f"  slice 개수 : {len(data['settings_list'])}")

    delays = np.array(data['cable_delay_list'])
    print(f"  cable_delay: 전부 동일한가? {np.allclose(delays, delays[0])} "
          f"(값={delays[0]:.4e}초)")
        # cable_delay가 자기장과 무관하게 일정한지 확인 - 만약 전부
        # 같다면, 자기장이 바뀌어도 케이블/장비 자체의 지연은 안 바뀐다는
        # 뜻이라 물리적으로 당연한 결과. 만약 달랐다면 뭔가 이상 신호.

    # 첫/마지막 slice의 fr 근처 스캔 범위를 비교해, "주파수 창이
    # 실제로 자기장에 따라 따라 움직이는지"를 직접 확인
    print(f"\n  첫 번째 slice 주파수 범위: "
          f"[{data['freq_hz_2d'][0].min()/1e9:.4f}, {data['freq_hz_2d'][0].max()/1e9:.4f}] GHz")
    print(f"  마지막 slice 주파수 범위: "
          f"[{data['freq_hz_2d'][-1].min()/1e9:.4f}, {data['freq_hz_2d'][-1].max()/1e9:.4f}] GHz")
    print("-" * 55)


if __name__ == "__main__":
    expected_functions = ['load_fano_fieldsweep_npz', 'summarize_loaded_data']
    print("fano_fieldsweep_loader.py 자가진단")
    print("-" * 55)
    all_ok = True
    for name in expected_functions:
        exists = name in dir()
        status = "OK" if exists else "누락!!"
        print(f"  {name:28s} : {status}")
        all_ok = all_ok and exists
    print("-" * 55)
    print("전체 상태:", "정상" if all_ok else "일부 함수 누락")
