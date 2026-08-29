"""
fano_visualize.py
=================================================================
실제 Fano 데이터(resonator_1_powersweep_overcoupled.npz)를 로드해서,
(전력 x 주파수) 히트맵과 대표 전력 slice의 단면 그래프를 그려
데이터 형태를 눈으로 먼저 확인하는 스크립트입니다.

[왜 모델을 만들기 전에 먼저 시각화하는가]
어제 계속 강조했던 원칙 - "모델을 세우기 전에 먼저 데이터를 눈으로
봐야 한다"를 여기서도 그대로 따릅니다. 특히 오늘은:
  - overcoupled(과대결합) 조건이라, 딥이 아주 깊고 뚜렷하게 나올
    가능성이 높음 (coupling regime에 대한 것은 어제 lineshape_gallery.py
    에서 이미 다룸 - ke2가 ke1+ki보다 훨씬 클 때가 overcoupled)
  - Fano 비대칭이 실제로 눈에 보이는지(좌우 딥 모양이 기울어져
    있는지) 확인하는 것이 이번 시각화의 핵심 목적
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.getcwd())
    # os.getcwd(): 현재 작업 디렉토리를 기준으로 fano_loader.py를 찾음.
    # (Colab에서 __file__이 불안정하게 동작하는 경우가 있어, 어제부터
    # 계속 이 방식을 씀 - 노트북이 있는 폴더와 이 스크립트가 같은
    # 위치에 있어야 함)
import fano_loader


# =========================================================
# STEP 0. 분석자 설정값 (aa_ 접두사)
# =========================================================
aa_drive_folder = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module01/Fano/overcoupled/'
aa_filename = 'resonator_1_powersweep_overcoupled.npz'
aa_output_dir = './outputs/'
os.makedirs(aa_output_dir, exist_ok=True)

aa_slice_index_to_plot = -1
    # 단면(1D) 그래프로 자세히 볼 전력 slice의 인덱스. -1은 배열의
    # 마지막(어제 zenodo_load_real_data.py에서 "최고 전력"이었던 것과
    # 같은 관례). 다만 오늘 데이터는 power 배열이 [10,...,80]으로
    # 확인됐으므로, -1이 정확히 "가장 큰 값"인지는 STEP 1에서
    # 실제 power_dbm 배열을 출력해 확인 후 필요하면 조정.

print("=" * 60)
print("분석자 설정값(aa_*) 요약:")
for k, v in list(globals().items()):
    if k.startswith("aa_"):
        print(f"  {k:24s} = {v}")
print("=" * 60)


# =========================================================
# STEP 1. 데이터 로드 (검증된 fano_loader.py 사용)
# =========================================================
full_path = os.path.join(aa_drive_folder, aa_filename)
data = fano_loader.load_fano_npz(full_path)
fano_loader.summarize_loaded_data(data)

print(f"\n전체 power_dbm 배열: {data['power_dbm']}")
    # 어제 실제 데이터에서 [10, 80] 범위로 나왔던 것을 다시 확인 -
    # "power" 라는 이름이지만 실제로는 dBm이 아니라 다른 스케일(예:
    # 감쇠기 세팅값의 절대값)일 가능성을 염두에 두고, 정확한 값들을
    # 눈으로 직접 확인해두는 것이 다음 단계(모델링)에서 헷갈리지
    # 않는 데 중요함.


# =========================================================
# STEP 2. 히트맵 시각화 - (전력 x 주파수)
# =========================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

im0 = axes[0, 0].pcolormesh(data['freq_ghz'], data['power_dbm'], data['amplitude'],
                              cmap='viridis', shading='auto')
axes[0, 0].set_xlabel('Frequency (GHz)')
axes[0, 0].set_ylabel('Power (raw units, dBm 여부 미확정)')
axes[0, 0].set_title('|S21| Amplitude heatmap')
plt.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].pcolormesh(data['freq_ghz'], data['power_dbm'], data['phase'],
                              cmap='twilight', shading='auto')
    # cmap='twilight': 위상(각도)처럼 -pi와 +pi가 "같은 지점"에서
    # 순환적으로 이어지는 데이터에 적합한, matplotlib의 순환형(cyclic)
    # 컬러맵. 일반 컬러맵을 쓰면 -pi와 +pi가 전혀 다른 색으로 보여서
    # 오해를 살 수 있음 (실제로는 거의 같은 위상인데).
axes[0, 1].set_xlabel('Frequency (GHz)')
axes[0, 1].set_ylabel('Power (raw units)')
axes[0, 1].set_title('Phase heatmap')
plt.colorbar(im1, ax=axes[0, 1])

# 대표 slice의 단면(1D) 진폭/위상 - Fano 비대칭이 실제로 보이는지 확인
freq = data['freq_ghz']
amp_slice = data['amplitude'][aa_slice_index_to_plot, :]
phase_slice = data['phase'][aa_slice_index_to_plot, :]
power_val = data['power_dbm'][aa_slice_index_to_plot]

axes[1, 0].plot(freq, amp_slice, '.-', ms=3)
axes[1, 0].set_xlabel('Frequency (GHz)')
axes[1, 0].set_ylabel('|S21|')
axes[1, 0].set_title(f'단면(진폭), power={power_val:.1f} (index={aa_slice_index_to_plot})')

axes[1, 1].plot(freq, phase_slice, '.-', ms=3, color='tab:orange')
axes[1, 1].set_xlabel('Frequency (GHz)')
axes[1, 1].set_ylabel('phase (rad)')
axes[1, 1].set_title(f'단면(위상), power={power_val:.1f}')

plt.tight_layout()
plt.savefig(os.path.join(aa_output_dir, 'fano_data_overview.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\n시각화 저장 완료: {os.path.join(aa_output_dir, 'fano_data_overview.png')}")

# =========================================================
# STEP 3. 딥의 대칭성을 눈으로 판단하는 데 도움 - 복소평면(원) 그리기
# =========================================================
# circuit.py 전체가 "S21이 복소평면에서 원을 그린다"는 사실 위에
# 세워진 방법이므로, 실제로 우리 데이터가 원 모양을 그리는지 직접
# 눈으로 확인하는 것이 이해에 큰 도움이 됨.
s21_slice = data['s21'][aa_slice_index_to_plot, :]

fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.plot(np.real(s21_slice), np.imag(s21_slice), '.-', ms=3)
ax2.set_xlabel('Re(S21)')
ax2.set_ylabel('Im(S21)')
ax2.set_title(f'복소평면 상의 S21 궤적 (power={power_val:.1f})\n'
              f'"원"에 가까운 모양이어야 circle fit이 잘 맞음')
ax2.set_aspect('equal')
    # set_aspect('equal'): x축과 y축의 스케일을 동일하게 강제.
    # 이게 없으면 실제로는 원인데 화면에서 타원처럼 찌그러져 보여서
    # "원인지 아닌지" 판단이 왜곡될 수 있음 - 복소평면 궤적을 볼 땐
    # 항상 이 옵션을 켜는 것이 표준적인 관례.
plt.tight_layout()
plt.savefig(os.path.join(aa_output_dir, 'fano_complex_plane.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"복소평면 시각화 저장 완료: {os.path.join(aa_output_dir, 'fano_complex_plane.png')}")
