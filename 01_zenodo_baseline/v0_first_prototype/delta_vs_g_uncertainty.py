"""
Diagnostic: |Delta| vs g estimation uncertainty
=================================================================
"디튜닝(Δ)이 커질수록 g(결합강도) 추정이 왜/어느 정도로 무너지는가"를
정량적으로 확인하기 위한 진단 스크립트.

이전 배치 피팅(anticrossing_full_flux_sweep.py)에서 저장한
flux_sweep_fit_results.npz 를 불러와서,
  x축: |Δ| = |fq(Φ) - fr|   (디튜닝의 절대값, GHz)
  y축: g의 posterior 오차폭 (hi+lo, GHz) — "이 값이 클수록 추정이 불안정"
을 산점도로 그려, 어느 |Δ| 부근에서 오차가 급격히 커지는지
(threshold, 문턱값) 눈으로 확인합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# =========================================================
# STEP 0. 분석자 설정값
# =========================================================
aa_output_dir = '/content/drive/MyDrive/eunjunglee/SuperQuantum/outputs/'
aa_results_filename = 'flux_sweep_fit_results.npz'

# ①번 식에 쓰이는 상수 (배치 피팅 스크립트와 반드시 동일한 값을 써야 함
# -> 그래야 같은 fq(Φ) 곡선이 재현되어 Δ를 정확히 계산할 수 있음)
aa_fq_max = 5.150   # GHz
aa_EC     = 0.250   # GHz

# fr 기준값: 원칙적으로는 각 slice에서 피팅된 fr을 써야 더 정확하지만,
# 여기서는 "설계 스펙상의 fr"(참값 근처, 5.0 GHz)을 기준선으로 사용해
# Δ를 계산합니다. (fr 자체도 flux에 따라 아주 미세하게 흔들리므로,
# 어느 쪽을 쓸지는 분석자가 정하는 선택입니다 -> aa_ 처리)
aa_fr_reference = 5.000   # GHz

# =========================================================
# STEP 1. 배치 피팅 결과 로드
# =========================================================
full_path = os.path.join(aa_output_dir, aa_results_filename)
data = np.load(full_path, allow_pickle=True)
    # allow_pickle=True: status 배열처럼 문자열(object) 타입을 담은 .npz를
    # 읽으려면 파이썬 객체 직렬화(pickle)를 허용해야 함

flux = data['flux']
g_val = data['g']
g_err_lo = data['g_err_lo']
g_err_hi = data['g_err_hi']
kappa_val = data['kappa']
status = data['status']

print(f"로드 완료: {len(flux)}개 slice")

# =========================================================
# STEP 2. 각 slice의 |Δ| 계산
# =========================================================
# ①번 식을 그대로 재사용해 flux -> fq 변환
fq_arr = (aa_fq_max + aa_EC) * np.sqrt(np.abs(np.cos(np.pi * flux))) - aa_EC
delta_arr = fq_arr - aa_fr_reference
abs_delta = np.abs(delta_arr)

# g의 posterior 오차폭 (비대칭 오차의 상한+하한을 합쳐서 "전체 불확실성 폭"으로 사용)
g_err_total = g_err_lo + g_err_hi

# =========================================================
# STEP 3. 산점도 시각화 (|Δ| vs g 오차폭)
# =========================================================
status_colors = {'ok': 'tab:blue', 'unstable': 'tab:orange', 'failed': 'tab:red'}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- 왼쪽 패널: |Δ| vs g 오차폭 ---
for status_name, color in status_colors.items():
    mask = status == status_name
    if not np.any(mask):
        continue
    axes[0].scatter(abs_delta[mask], g_err_total[mask],
                     color=color, label=status_name, alpha=0.7, s=25)
        # scatter: 점만 찍는 산점도. errorbar와 달리 오차막대 없이 점 하나로 표현.

axes[0].set_xlabel(r'$|\Delta|$ = $|f_q(\Phi) - f_r|$ (GHz)')
axes[0].set_ylabel(r'$g$ posterior 오차폭 (GHz)')
axes[0].set_title('디튜닝 크기에 따른 g 추정 불확실성')
axes[0].legend()
axes[0].axhline(0.02, color='gray', ls=':', lw=1, label='unstable 판정 기준')
    # 배치 피팅에서 썼던 aa_failure_std_threshold_g=0.02 기준선을 참고용으로 표시

# --- 오른쪽 패널: flux 대 |Δ| (참고용, 물리적 맥락 확인) ---
axes[1].plot(flux, abs_delta, '.-', color='black', ms=3)
axes[1].set_xlabel(r'Flux $\Phi/\Phi_0$')
axes[1].set_ylabel(r'$|\Delta|$ (GHz)')
axes[1].set_title('Flux에 따른 디튜닝 크기 변화')

plt.tight_layout()
plt.savefig(os.path.join(aa_output_dir, 'delta_vs_g_uncertainty.png'), dpi=150, bbox_inches='tight')
plt.show()

# =========================================================
# STEP 4. 문턱값(threshold) 정량적으로 추정
# =========================================================
# "ok" 상태였던 slice들 중 |Δ|의 최댓값 = 실질적으로 안정적 추정이 가능했던 한계
ok_mask = status == 'ok'
if np.any(ok_mask):
    delta_threshold_estimate = np.max(abs_delta[ok_mask])
    print(f"\n[진단 결과] 'ok'로 판정된 slice들의 |Δ| 최댓값: {delta_threshold_estimate:.4f} GHz")
    print(f"  -> 이 실험 설정(g={aa_fq_max}대 스케일, kappa 등)에서는")
    print(f"     대략 |Δ| < {delta_threshold_estimate:.3f} GHz 영역에서만")
    print(f"     g를 안정적으로 추정할 수 있다고 잠정 결론 내릴 수 있습니다.")
else:
    print("\n'ok' 상태 slice가 없습니다.")
