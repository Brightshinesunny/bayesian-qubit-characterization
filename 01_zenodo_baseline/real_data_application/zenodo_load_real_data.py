"""
load_real_zenodo_data.py
=================================================================
실제 Zenodo 데이터 파일(S21_power_sweep_*.txt)을 Google Drive
경로에서 읽어와, 구조를 확인하고 |S21|을 (전력 x 주파수) 히트맵으로
시각화하는 스크립트입니다.

이전 단계(verify_parser_equivalence.py)에서 우리 zenodo_loader.py가
원저자 방식과 완전히 동일한 결과를 낸다는 게 검증됐으므로, 이제
안심하고 실제 파일에 그대로 적용합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.getcwd())
    # os.getcwd(): 현재 작업 디렉토리(파이썬을 실행 중인 위치)를 반환.
    # 원래는 __file__(이 스크립트 자신의 경로)을 기준으로 zenodo_loader.py를
    # 찾으려 했으나, Colab처럼 셀 단위로 실행하거나 %load로 코드를
    # 불러오는 환경에서는 __file__이 제대로 정의되지 않는 경우가 흔함.
    # os.getcwd()는 이런 환경에서도 안정적으로 동작하므로, "이 노트북과
    # zenodo_loader.py가 같은 작업 폴더에 있다"는 전제 하에 이 방식을 씀.
import zenodo_loader


# =========================================================
# STEP 0. 분석자 설정값 (aa_ 접두사)
# =========================================================
aa_drive_folder = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module01/Figure1/'
    # Figure1.zip 압축을 푼 뒤 파일들을 올려둔 실제 Drive 경로.
    # 폴더 구조가 바뀌면 이 값만 고치면 됨 (다른 코드는 안 건드려도 됨).

aa_filename = 'S21_power_sweep_152_data_210823_21h50m19s.txt'
    # 지금 분석할 파일 하나. 이전에 확인한 원저자의 Mathematica 참값이
    # 이 파일(kappa≈8MHz)에 대한 것이므로, 대조 검증을 위해 이 파일을
    # 먼저 고름. 다른 파일(kappa 1.2/2.6/4.7/12.8/29.7MHz)로 바꾸려면
    # 이 값만 교체하면 됨.

aa_output_dir = './outputs/'
os.makedirs(aa_output_dir, exist_ok=True)

print("=" * 60)
print("분석자 설정값(aa_*) 요약:")
for k, v in list(globals().items()):
    if k.startswith("aa_"):
        print(f"  {k:20s} = {v}")
print("=" * 60)


# =========================================================
# STEP 1. 실제 파일 로드 (검증된 zenodo_loader.py 사용)
# =========================================================
full_path = os.path.join(aa_drive_folder, aa_filename)

if not os.path.exists(full_path):
    # 경로가 틀렸을 때, "파일이 없다"는 사실만 알려주고 멈추는 대신
    # 그 폴더 안에 실제로 어떤 파일들이 있는지 보여줘서 오타/경로
    # 실수를 바로 찾을 수 있게 함 (지난번 겪었던 "구버전 파일" 문제와
    # 비슷하게, 이런 사전 확인 코드가 있으면 디버깅 시간이 크게 줄어듦).
    print(f"\n[오류] 파일을 찾을 수 없습니다: {full_path}")
    if os.path.exists(aa_drive_folder):
        print(f"이 폴더 안에 실제로 있는 파일들:")
        for fname in os.listdir(aa_drive_folder):
            print(f"  - {fname}")
    else:
        print(f"폴더 자체가 존재하지 않습니다: {aa_drive_folder}")
        print("Drive가 마운트되어 있는지, 경로 오타는 없는지 확인하세요.")
    raise FileNotFoundError(full_path)

data = zenodo_loader.load_power_sweep_txt(full_path)
zenodo_loader.summarize_loaded_data(data)


# =========================================================
# STEP 2. 복소수 S21 = I + jQ 로 결합
# =========================================================
if 'Q' not in data:
    raise ValueError("Q(허수부) 섹션을 찾지 못했습니다. 파일 구조를 확인하세요.")

s21_complex = data['I'] + 1j * data['Q']
    # 지금까지 계속 써온 관례: 실수부(I)와 허수부(Q)를 합쳐 하나의
    # 복소수 배열로 만듦. shape은 (전력 개수, 주파수 개수).

magnitude = np.abs(s21_complex)   # |S21|
power_db2 = np.abs(s21_complex) ** 2   # |S21|^2 (원저자 코드의 P21Lin에 대응)

print(f"\n|S21| 범위: [{magnitude.min():.5f}, {magnitude.max():.5f}]")
print(f"|S21|^2 범위: [{power_db2.min():.6e}, {power_db2.max():.6e}]")


# =========================================================
# STEP 3. 시각화 - (전력 x 주파수) 히트맵
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

im0 = axes[0].pcolormesh(data['freq_ghz'], data['power_dbm'], magnitude,
                           cmap='viridis', shading='auto')
axes[0].set_xlabel('Frequency (GHz)')
axes[0].set_ylabel('Input power (dBm)')
axes[0].set_title(f'|S21| heatmap\n({aa_filename})')
plt.colorbar(im0, ax=axes[0])

# 가장 높은 전력(마지막 행)에서의 단면(1D 스펙트럼) - 원저자 코드가
# P21Lin_5[-1]로 골랐던 것과 동일한 슬라이스
highest_power_idx = -1
axes[1].plot(data['freq_ghz'], magnitude[highest_power_idx, :], '.-', ms=3)
axes[1].set_xlabel('Frequency (GHz)')
axes[1].set_ylabel('|S21|')
axes[1].set_title(f'최고 전력({data["power_dbm"][highest_power_idx]:.0f} dBm) 단면')

plt.tight_layout()
plt.savefig(os.path.join(aa_output_dir, 'zenodo_real_data_overview.png'),
            dpi=150, bbox_inches='tight')
plt.show()

print(f"\n시각화 저장 완료: {os.path.join(aa_output_dir, 'zenodo_real_data_overview.png')}")
print("\n다음 단계: 위 오른쪽 그래프(단면)를 models.s21_single_resonance_transmission")
print("모델로 우리 MCMC 파이프라인에 태워, 원저자의 Mathematica 참값")
print("(f0=10.4701, ke1/ke2/ki, A0, phi, tau)과 대조 검증할 수 있습니다.")
