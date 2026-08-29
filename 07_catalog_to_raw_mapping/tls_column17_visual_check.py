"""
tls_column17_visual_check.py
=================================================================
[MCMC 전에 반드시 먼저] raw_index=503(열17, ao4)의 dispamp 맵에
목표 주파수(5.097GHz) 가로선을 그어서, 이게 진짜 avoided-crossing
(전압에 따라 기우는 대각선)인지, 아니면 오늘 하루 종일 봐온
계통오차성 수평 띠인지 육안으로 먼저 확정한다.

[판정 기준]
- 진한 줄이 가로선과 거의 평행하게(수평으로) 유지되면
  -> 계통오차 가능성 높음. avoided-crossing 모델 부적합.
     multi_horizontal_lines_model(이미 만들어둔 대안 모델) 사용.
- 진한 줄이 전압에 따라 뚜렷하게 기울면(대각선)
  -> 진짜 avoided-crossing 후보. 기존 2D 모델+MCMC 그대로 진행.
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments400to640_20TLSfitted.mat'
aa_raw_index = 503
aa_target_freq_ghz = 5.0972
aa_observable = 'dispamp'


def mat_string(f, dataset_or_ref):
    try:
        if isinstance(dataset_or_ref, h5py.Reference):
            dataset_or_ref = f[dataset_or_ref]
        arr = np.array(dataset_or_ref)
        return ''.join(chr(int(c)) for c in arr.flatten())
    except Exception:
        return None


with h5py.File(aa_mat_filepath, 'r') as f:
    qdat_group = f[f['qdats']['qdat'][aa_raw_index, 0]]
    sweep1vals = np.array(qdat_group['sweep1vals']).flatten()
    sweep2vals = np.array(qdat_group['sweep2vals']).flatten()
    electrode = mat_string(f, qdat_group['sweep1name'])

    calib_field = np.array(f['qset']['calibdat']['field']).flatten()
    calib_freq_hz = np.array(f['qset']['calibdat']['qubitfreq']).flatten()
    order = np.argsort(calib_field)
    fq_axis = np.interp(sweep2vals, calib_field[order], calib_freq_hz[order]) / 1e9

    obs_val_refs = qdat_group['obs']['vals']
    obs_name_refs = qdat_group['observables']
    target_col = None
    for i in range(obs_name_refs.shape[0]):
        if mat_string(f, obs_name_refs[i, 0]) == aa_observable:
            target_col = i
            break
    data_2d = np.array(f[obs_val_refs[target_col, 0]])

col_median = np.median(data_2d, axis=0, keepdims=True)
normalized = data_2d - col_median

# =====================================================
# [핵심] 연속성 추적(continuity tracking) - 오늘 seg12에서 이미 검증된
# 방법. 고정된 좁은 창(±20MHz)만 쓰면, 카탈로그가 말하는 큰 기울기
# (452MHz/V)가 진짜라면 전압 8.8%만 지나도 창을 벗어나 버림(직접
# 확인한 버그). 대신 "이전 전압의 딥 위치 근처"에서만 다음 딥을
# 찾는 방식으로, 큰 기울기든 작은 기울기든 다 따라갈 수 있게 함.
# =====================================================
max_jump_ghz = 0.05   # 전압 한 칸(0.05V)당 최대 허용 이동폭. 카탈로그
    # gamma=452MHz/V 기준, 0.05V당 22.6MHz 이동 예상 - 여유 있게 50MHz로 설정.
center_col = len(sweep1vals) // 2
anchor_freq = aa_target_freq_ghz

def find_nearby(target, freq_axis, max_jump):
    dist = np.abs(freq_axis - target)
    nearby = np.where(dist < max_jump)[0]
    return nearby if len(nearby) > 0 else None

dip_positions = np.full(len(sweep1vals), np.nan)
idx0 = np.argmin(np.abs(fq_axis - anchor_freq))
dip_positions[center_col] = fq_axis[idx0]
last_f = fq_axis[idx0]
last_b = fq_axis[idx0]

for i in range(center_col+1, len(sweep1vals)):
    nearby = find_nearby(last_f, fq_axis, max_jump_ghz)
    if nearby is None:
        break
    idx = nearby[np.argmin(normalized[nearby, i])]
    dip_positions[i] = fq_axis[idx]
    last_f = fq_axis[idx]

for i in range(center_col-1, -1, -1):
    nearby = find_nearby(last_b, fq_axis, max_jump_ghz)
    if nearby is None:
        break
    idx = nearby[np.argmin(normalized[nearby, i])]
    dip_positions[i] = fq_axis[idx]
    last_b = fq_axis[idx]

valid = ~np.isnan(dip_positions)
print(f"전압별 딥 위치(주파수, 연속성 추적): {np.round(dip_positions[valid], 4)}")
print(f"딥 위치 표준편차: {np.nanstd(dip_positions):.5f} GHz "
      f"(작을수록 수평, 클수록 기울어짐)")
slope_estimate = np.polyfit(sweep1vals[valid], dip_positions[valid], deg=1)[0]
print(f"1차 다항식으로 추정한 기울기: {slope_estimate*1000:.3f} MHz/V "
      f"(카탈로그 gamma=452MHz/V와 비교)")

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.pcolormesh(sweep1vals, fq_axis, normalized, shading='auto', cmap='viridis')
ax.axhline(aa_target_freq_ghz, color='red', ls='--', lw=1.5, label=f'카탈로그 주파수 {aa_target_freq_ghz}GHz')
ax.plot(sweep1vals, dip_positions, 'r.', ms=4, label='실제 추적된 딥 위치')
ax.set_xlabel('게이트 전압 V')
ax.set_ylabel('큐빗 주파수 (GHz)')
ax.set_title(f'raw_index={aa_raw_index}({electrode}) - 열17 검증\n'
              f'딥 위치 표준편차={dip_positions.std()*1000:.2f}MHz, '
              f'추정기울기={slope_estimate*1000:.2f}MHz/V')
ax.legend()
plt.tight_layout()
plt.savefig('./outputs/tls_column17_visual_check.png', dpi=140)
print("\n저장 완료: ./outputs/tls_column17_visual_check.png")

print("\n[판정 가이드]")
print("  표준편차가 매우 작고(<1MHz) 추정기울기가 카탈로그 452MHz/V보다")
print("  훨씬 작으면 -> 수평 띠(계통오차) 가능성 높음, MCMC 전에 모델 재검토 필요")
print("  표준편차가 크고 추정기울기가 452MHz/V와 비슷하면 -> 진짜 대각선,")
print("  기존 avoided-crossing 모델+MCMC 그대로 진행 가능")
