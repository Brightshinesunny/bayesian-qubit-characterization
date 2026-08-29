"""
lineshape_gallery.py
=================================================================
지금까지 다룬(그리고 아직 안 다뤘지만 흔히 마주치는) 공진 lineshape
형태들을, 어떤 파라미터를 스윕했을 때 모양이 어떻게 바뀌는지
히트맵으로 나란히 보여주는 "갤러리" 스크립트입니다.

목적: 각 형태를 전부 암기하는 게 아니라, "이런 형태들이 존재하고,
어느 파라미터가 그 형태를 결정하는지" 감을 잡는 것. 실제로 새로운
데이터를 만났을 때, 여기서 본 패턴 중 무엇과 비슷한지 눈으로
판단하는 데 씁니다.

패널 구성:
  1. Avoided-crossing (딥 2개) - models.py, flux를 스윕
  2. 단일 공진 피크, 결합 세기(coupling regime) 스윕 - zenodo_models.py
     -> "undercoupled/critically coupled/overcoupled"라는 표준 개념을
        직접 눈으로 확인 (임피던스 정합 여부에 따라 피크 높이/위상
        회전 방향이 달라짐)
  3. Fano 비대칭 - models.py, Fano q를 스윕
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import models
import zenodo_models


# =========================================================
# STEP 0. 분석자 설정값 (aa_ 접두사)
# =========================================================
aa_output_dir = './outputs/'
os.makedirs(aa_output_dir, exist_ok=True)

aa_n_freq = 301

# --- 패널 1: avoided-crossing (딥) ---
aa_ac_freq_range = (4.8, 5.2)
aa_ac_flux_values = np.linspace(-0.5, 0.5, 101)
aa_ac_f_r, aa_ac_g, aa_ac_kappa = 5.0, 0.04, 0.03
aa_ac_fq_max, aa_ac_EC, aa_ac_tau = 5.15, 0.25, 0.12

# --- 패널 2: 단일 공진 피크, coupling regime 스윕 ---
aa_peak_freq_range = (10.42, 10.52)
aa_peak_f0, aa_peak_ki, aa_peak_A0, aa_peak_phi, aa_peak_tau = 10.47, 0.0005, 1.0, 0.0, 0.0
aa_peak_ke1 = 0.003   # 포트1 결합은 고정
aa_peak_ke2_values = np.linspace(0.0001, 0.01, 101)
    # [수정] 원래 0.02까지였으나, 실제로 피크가 가장 뚜렷해지는 지점
    # (critical coupling, 아래 설명)이 ke2≈0.0035 근처에 있어서 그
    # 범위가 그래프 아래쪽에 눌려 잘 안 보였음. 0.01로 좁혀서 critical
    # coupling 전후의 변화가 더 잘 보이도록 조정.
    # ke2를 아주 약한 결합(undercoupled)에서 강한 결합(overcoupled)까지
    # 스윕. 수학적으로 정규화된 피크 세기는 4*ke1*ke2/(ke1+ke2+ki)^2
    # 형태라, ke2 = ke1+ki 지점에서 최댓값을 가짐(미분해서 확인 가능) -
    # 이게 바로 "critically coupled"(임계 결합) 지점이고, 그 전후로
    # undercoupled/overcoupled 영역이 나뉨.

# --- 패널 3: Fano 비대칭 ---
aa_fano_freq_range = (4.8, 5.2)
aa_fano_f_center = 5.0
aa_fano_kappa_for_lorentzian = 0.03
aa_fano_q_values = np.linspace(-5, 5, 101)
    # q가 매우 크거나 매우 작은 음수/양수면 원래의 대칭 로렌츠에 가깝고,
    # q가 0 근처를 지나면서 비대칭이 가장 심해지고 형태가 "뒤집힘"

print("=" * 60)
print("분석자 설정값(aa_*) 요약:")
for k, v in list(globals().items()):
    if k.startswith("aa_") and not isinstance(v, np.ndarray):
        print(f"  {k:24s} = {v}")
print("=" * 60)


# =========================================================
# STEP 1. 패널 1: Avoided-crossing 히트맵 (복습 - 딥 2개)
# =========================================================
f_grid_ac = np.linspace(*aa_ac_freq_range, aa_n_freq)
s21_ac_2d = np.zeros((len(aa_ac_flux_values), aa_n_freq), dtype=complex)
for i, flux_val in enumerate(aa_ac_flux_values):
    s21_ac_2d[i, :] = models.s21_anticrossing_model(
        f_grid_ac, flux_val, aa_ac_f_r, aa_ac_g, aa_ac_kappa,
        aa_ac_fq_max, aa_ac_EC, aa_ac_tau
    )


# =========================================================
# STEP 2. 패널 2: 단일 공진 피크, coupling regime 히트맵 (신규)
# =========================================================
f_grid_peak = np.linspace(*aa_peak_freq_range, aa_n_freq)
s21_peak_2d = np.zeros((len(aa_peak_ke2_values), aa_n_freq), dtype=complex)
for i, ke2_val in enumerate(aa_peak_ke2_values):
    s21_peak_2d[i, :] = zenodo_models.s21_single_resonance_transmission(
        f_grid_peak, aa_peak_f0, aa_peak_ke1, ke2_val, aa_peak_ki,
        aa_peak_A0, aa_peak_phi, aa_peak_tau
    )


# =========================================================
# STEP 3. 패널 3: Fano 비대칭 히트맵 (신규)
# =========================================================
f_grid_fano = np.linspace(*aa_fano_freq_range, aa_n_freq)
# 기본 로렌츠 딥(참고용): "1 - 로렌츠" 형태로 avoided-crossing 없이
# 순수 단일 로렌츠 딥만 만들어, 여기에 Fano 보정을 입힘
denom = (aa_fano_kappa_for_lorentzian / 2.0) + 1j * (f_grid_fano - aa_fano_f_center)
s21_fano_base = 1.0 - (aa_fano_kappa_for_lorentzian / 2.0) / denom

s21_fano_2d = np.zeros((len(aa_fano_q_values), aa_n_freq), dtype=complex)
for i, q_val in enumerate(aa_fano_q_values):
    s21_fano_2d[i, :] = models.fano_lineshape_correction(
        s21_fano_base, f_grid_fano, aa_fano_f_center, q_val
    )


# =========================================================
# STEP 4. 세 패널을 나란히 시각화
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

im0 = axes[0].pcolormesh(f_grid_ac, aa_ac_flux_values, np.abs(s21_ac_2d),
                          cmap='viridis', shading='auto')
axes[0].set_xlabel('Frequency (GHz)')
axes[0].set_ylabel(r'Flux $\Phi/\Phi_0$')
axes[0].set_title('① Avoided-crossing\n(flux 스윕 - 딥 2개가 갈라짐)')
plt.colorbar(im0, ax=axes[0], label='|S21|')

im1 = axes[1].pcolormesh(f_grid_peak, aa_peak_ke2_values * 1e3, np.abs(s21_peak_2d),
                          cmap='viridis', shading='auto')
axes[1].axhline((aa_peak_ke1 + aa_peak_ki) * 1e3, color='red', ls='--', lw=1,
                 label=f'ke2=ke1+ki (critical coupling)')
    # [수정] critical coupling이 정확히 ke2=ke1이 아니라 ke2=ke1+ki
    # 지점에서 일어남을 수학적으로 확인(정규화 피크세기 식을 ke2로
    # 미분해 0이 되는 지점 계산) 후 반영. ki가 작으면 ke1과 거의 같은
    # 위치지만, 엄밀하게는 이 지점이 맞음.
axes[1].set_xlabel('Frequency (GHz)')
axes[1].set_ylabel('ke2 (MHz)')
axes[1].set_title('② 단일 공진 피크\n(ke2 스윕 - coupling regime 변화)')
axes[1].legend(fontsize=8)
plt.colorbar(im1, ax=axes[1], label='|S21|')

im2 = axes[2].pcolormesh(f_grid_fano, aa_fano_q_values, np.abs(s21_fano_2d),
                          cmap='viridis', shading='auto')
axes[2].axhline(0, color='red', ls='--', lw=1, label='q=0 (가장 비대칭)')
axes[2].set_xlabel('Frequency (GHz)')
axes[2].set_ylabel('Fano q')
axes[2].set_title('③ Fano 비대칭 딥\n(q 스윕 - 대칭↔비대칭↔뒤집힘)')
axes[2].legend(fontsize=8)
plt.colorbar(im2, ax=axes[2], label='|S21|')

plt.tight_layout()
plt.savefig(os.path.join(aa_output_dir, 'lineshape_gallery.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\n저장 완료: {os.path.join(aa_output_dir, 'lineshape_gallery.png')}")
print("\n[읽는 법]")
print("① 왼쪽: flux=0 근처에서 두 딥이 가장 멀리 벌어지고(강결합),")
print("   flux가 커질수록 한쪽 딥이 옅어지며 사실상 하나만 남음")
print("   (지난번 |Δ|가 클 때 g 추정이 불안정해졌던 바로 그 구간)")
print("② 가운데: ke2가 작을 때(undercoupled)는 피크가 얕고,")
print("   ke2=ke1+ki(critical coupling, 빨간선)에서 피크가 가장")
print("   뚜렷해지며, ke2가 더 커지면(overcoupled) 다시 얕아짐")
print("   (수학적으로 정규화 피크세기 = 4*ke1*ke2/(ke1+ke2+ki)^2 이")
print("   ke2=ke1+ki에서 최댓값을 가지는 것과 정확히 일치함)")
print("③ 오른쪽: q가 매우 크거나 작을 땐 대칭적인 딥, q=0 근처를")
print("   지나가며 좌우 비대칭이 극대화되고 형태가 뒤집힘")
