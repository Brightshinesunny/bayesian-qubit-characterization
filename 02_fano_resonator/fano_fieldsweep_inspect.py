"""
fano_fieldsweep_inspect.py
=================================================================
[목적] resonator_7_fieldsweep_overcoupled.npz의 파일 구조를 먼저
확인하는 스크립트입니다. 지금까지 다룬 power sweep 파일들(변수:
amplitude, phase, frequency, power, settings)과 이름 자체부터
다르므로("power" 대신 "field"일 가능성), fano_loader.py를 그대로
쓰기 전에 실제 키(key) 이름과 배열 shape을 먼저 확인해야 합니다.

[용어 정리 - 이번 실험의 물리적 배경]
  운동 인덕턴스(kinetic inductance): 초전도체 안의 전자쌍(Cooper
    pair)이 전류 방향을 바꿀 때 관성 때문에 저항하는 정도. 일반
    도선의 "저항"과 달리 에너지를 소모하지 않지만, 회로의 유효
    인덕턴스에 기여해서 공진 주파수를 결정하는 요소 중 하나임.
    외부 자기장이 걸리면 이 값이 미세하게 바뀌어, 공진 주파수(fr)가
    자기장 세기에 따라 이동하는 원인이 됨.
  보텍스(vortex, 자속 소용돌이): 초전도체가 임계 자기장을 넘어서는
    자기장에 노출되면, 자속이 실처럼 초전도체를 뚫고 들어와
    소용돌이 형태로 갇히는 현상. 이 보텍스가 마이크로파 신호의
    에너지를 흡수(손실)시켜, 자기장이 세질수록 Qi(내부 품질계수)가
    나빠지는(작아지는) 대표적인 원인이 됨.
  Field resilience(자기장 내성): 초전도 공진기/큐빗이 외부 자기장
    아래에서도 성능(Qi 등)을 얼마나 유지하는지를 나타내는 용어.
    스핀 큐빗, NV 센터 등 자기장이 필요한 다른 양자 시스템과의
    하이브리드 실험에서 중요하게 평가되는 지표.
"""

import numpy as np
import os

aa_drive_folder = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module01/Fano/overcoupled/'
aa_filename = 'resonator_7_fieldsweep_overcoupled.npz'

full_path = os.path.join(aa_drive_folder, aa_filename)

raw = np.load(full_path, allow_pickle=True)
    # allow_pickle=True: 어제와 동일하게, settings 항목이 순수 숫자
    # 배열이 아니라 파이썬 객체(문자열)를 담고 있을 가능성에 대비.

print("=" * 60)
print(f"파일: {aa_filename}")
print("=" * 60)
print(f"\n포함된 키(key) 목록: {raw.files}")
print("-" * 60)
for key in raw.files:
    arr = raw[key]
    print(f"  {key:15s}: shape={arr.shape}, dtype={arr.dtype}")
    if arr.dtype != object and arr.size <= 10:
        print(f"                 값={arr}")
    elif arr.dtype != object:
        print(f"                 범위=[{np.min(arr):.6g}, {np.max(arr):.6g}], "
              f"첫 5개={arr.flat[:5]}")

# settings가 있다면 내용도 확인 (어제와 동일한 JSON 파싱 시도)
if 'settings' in raw.files:
    import json
    try:
        settings_dict = json.loads(raw['settings'][0])
        print(f"\nsettings 내용:")
        print(json.dumps(settings_dict, indent=2))
    except Exception as e:
        print(f"\nsettings 파싱 실패 (형식이 다를 수 있음): {e}")
        print(f"raw settings 내용: {raw['settings']}")
