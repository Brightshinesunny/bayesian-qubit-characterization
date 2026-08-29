"""
문턱값(threshold) 판단 방식 3단계 비교
=================================================================
같은 데이터(flux_sweep_fit_results.npz)를 놓고, "|Delta|가 얼마 이상이면
g 추정을 못 믿는가"를 판단하는 세 가지 방식을 나란히 비교합니다.

  [방식 A] 기존 - 이산 라벨(status)에서 극값(np.max)을 뽑음
            -> 자르기 2회, 정보 손실 큼, 이상치에 매우 취약
  [방식 B] 1차 개선 - 연속값을 |Δ| 순서로 정렬해 threshold-crossing 탐지
            -> 자르기 1회, 여전히 "0.02"라는 임의의 숫자를 사람이 정함
  [방식 C] 2차 개선 - 자르기 자체를 코드 안에서 하지 않고,
            (i) 변곡점(기울기가 급변하는 지점)을 데이터에서 자동 탐지하거나
            (ii) 아예 자르지 않고 연속 가중치(투명도)로 표현해
                 "어디서 자를지"는 사람이 그래프를 보고 최종 판단하게 남겨둠
            -> 자르기 0~1회, 임의의 상수(0.02)에 대한 의존도 최소화
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# =========================================================
# STEP 0. 설정값
# =========================================================
aa_output_dir = '/content/drive/MyDrive/eunjunglee/SuperQuantum/outputs/'
aa_results_filename = 'flux_sweep_fit_results.npz'
aa_fq_max = 5.150
aa_EC     = 0.250
aa_fr_reference = 5.000
aa_threshold_line = 0.02   # 방식 A, B에서만 사용하는 임의의 문턱값

# =========================================================
# STEP 1. 데이터 로드 및 |Δ| 계산 (공통)
# =========================================================
full_path = os.path.join(aa_output_dir, aa_results_filename)
data = np.load(full_path, allow_pickle=True)

flux = data['flux']
g_err_lo = data['g_err_lo']
g_err_hi = data['g_err_hi']
status = data['status']

fq_arr = (aa_fq_max + aa_EC) * np.sqrt(np.abs(np.cos(np.pi * flux))) - aa_EC
abs_delta = np.abs(fq_arr - aa_fr_reference)
g_err_total = g_err_lo + g_err_hi

sort_idx = np.argsort(abs_delta)
delta_sorted = abs_delta[sort_idx]
g_err_sorted = g_err_total[sort_idx]

# =========================================================
# [방식 A] 기존: 이산 라벨(status)의 극값
# =========================================================
ok_mask = status == 'ok'
threshold_A = np.max(abs_delta[ok_mask]) if np.any(ok_mask) else np.nan
    # 문제점: status라는 "이미 한 번 잘린" 라벨에서 다시 극값을 뽑음
    #   -> 자르기 2회, 이상치 하나(우연히 ok가 된 큰 |Δ|)에 결과가 좌우됨

# =========================================================
# [방식 B] 1차 개선: 연속값에서 threshold-crossing
# =========================================================
over_threshold = g_err_sorted > aa_threshold_line
threshold_B = delta_sorted[np.argmax(over_threshold)] if np.any(over_threshold) else np.nan
    # 개선점: 자르기 1회로 축소, status를 경유하지 않음
    # 한계: 여전히 "0.02"라는 사람이 정한 상수에 의존

# =========================================================
# [방식 C-i] 2차 개선: 변곡점(기울기 급변 지점) 자동 탐지
# =========================================================
# 아이디어: g 오차폭이 |Δ|에 따라 "완만 -> 급격히 증가"로 바뀌는 지점을
# 데이터 자체의 "기울기 변화"로 찾는다 (사람이 정한 0.02 같은 상수 불필요).
#
# 방법: 이동평균으로 노이즈를 줄인 뒤, 1차 미분(기울기)을 계산하고,
# 그 기울기가 "이전 구간 평균 기울기의 N배"를 처음 넘는 지점을 변곡점으로 봄.

def moving_average(x, w):
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode='same')

aa_smoothing_window_diag = 7   # 진단용 스무딩 윈도우 (분석자가 데이터 밀도를 보고 정함)
g_err_smooth = moving_average(g_err_sorted, aa_smoothing_window_diag)

# np.gradient: 배열의 국소 기울기(수치 미분)를 계산하는 함수.
# np.diff와 비슷하지만 양 끝점도 처리해주고, 배열 길이가 입력과 동일하게 유지됨.
slope = np.gradient(g_err_smooth, delta_sorted)

# 기울기가 "초반 평탄 구간 평균 기울기"의 몇 배를 넘는 첫 지점을 변곡점으로 정의
n_reference = max(len(slope) // 5, 5)   # 앞쪽 20%를 "평탄 구간 기준"으로 사용
reference_slope = np.mean(np.abs(slope[:n_reference]))
aa_slope_multiplier = 5   # "기준 기울기의 몇 배"를 변곡점으로 볼지 (분석자가 정하는 민감도)

steep_region = np.abs(slope) > aa_slope_multiplier * reference_slope
threshold_C = delta_sorted[np.argmax(steep_region)] if np.any(steep_region) else np.nan

print("=" * 60)
print("세 가지 방식의 문턱값 비교")
print("=" * 60)
print(f"[방식 A] 이산 라벨 극값(np.max)       : |Δ| ≈ {threshold_A:.4f} GHz")
print(f"[방식 B] 연속값 threshold-crossing    : |Δ| ≈ {threshold_B:.4f} GHz")
print(f"[방식 C] 변곡점 자동 탐지(기울기 기반) : |Δ| ≈ {threshold_C:.4f} GHz")
print("=" * 60)

# =========================================================
# [방식 C-ii] 아예 자르지 않고 연속 가중치로 시각화
# =========================================================
# "ok/unstable"로 이진 색칠하는 대신, g 오차폭 자체를 투명도(alpha)와
# 색상 농도로 표현. 자르기를 그래프 단계에서도 하지 않고,
# "어디서부터 못 믿을지"는 보는 사람이 최종 판단하도록 정보를 그대로 전달.

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 왼쪽: 기존 이진 색칠 방식 (비교용, 방식 A/B가 근거로 삼는 시각화)
status_colors = {'ok': 'tab:blue', 'unstable': 'tab:orange', 'failed': 'tab:red'}
for status_name, color in status_colors.items():
    mask = status == status_name
    if not np.any(mask):
        continue
    axes[0].scatter(abs_delta[mask], g_err_total[mask], color=color, label=status_name, alpha=0.7, s=25)
axes[0].axvline(threshold_A, color='tab:blue', ls='--', lw=1, label=f'A: {threshold_A:.2f}')
axes[0].axvline(threshold_B, color='tab:green', ls='--', lw=1, label=f'B: {threshold_B:.2f}')
axes[0].axvline(threshold_C, color='tab:purple', ls='--', lw=1, label=f'C: {threshold_C:.2f}')
axes[0].set_xlabel(r'$|\Delta|$ (GHz)')
axes[0].set_ylabel(r'$g$ posterior 오차폭 (GHz)')
axes[0].set_title('기존: 이진 색칠 + 세 threshold 비교')
axes[0].legend(fontsize=8)

# 오른쪽: 연속 가중치 방식 (자르지 않고 원본 정보를 그대로 표현)
# 오차폭이 클수록 점을 더 연하고 크게(=덜 신뢰) 표현하는 방식의 예시
normalized_err = (g_err_total - g_err_total.min()) / (g_err_total.max() - g_err_total.min())
    # 0~1 사이로 정규화 (min-max scaling): 오차폭의 상대적 크기를 0(가장 신뢰)~1(가장 불신)로 매핑
alpha_vals = 1.0 - 0.85 * normalized_err   # 오차폭이 클수록 투명해지도록(=흐리게) 변환
sizes = 15 + 60 * normalized_err            # 오차폭이 클수록 점을 크게 (불확실성을 시각적으로 강조)

sc = axes[1].scatter(abs_delta, g_err_total, c=g_err_total, cmap='viridis',
                      s=sizes, alpha=None)
    # alpha=None + c=값 조합으로는 개별 alpha 조절이 안 되므로, 아래처럼
    # 점 하나씩 개별 alpha를 적용하려면 루프가 필요함 (matplotlib 제약)
for i in range(len(abs_delta)):
    axes[1].scatter(abs_delta[i], g_err_total[i], color=plt.cm.viridis(normalized_err[i]),
                     s=sizes[i], alpha=alpha_vals[i])
axes[1].set_xlabel(r'$|\Delta|$ (GHz)')
axes[1].set_ylabel(r'$g$ posterior 오차폭 (GHz)')
axes[1].set_title('개선: 자르지 않고 연속 가중치(색/크기/투명도)로 표현')

plt.tight_layout()
plt.savefig(os.path.join(aa_output_dir, 'threshold_methods_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n비교 그래프 저장 완료:",
      os.path.join(aa_output_dir, 'threshold_methods_comparison.png'))
