"""
tls_seg12_v5_precrossing_only.py
=================================================================
[v5 - 교차 지점 이전 구간만 사용]
v4에서 확인된 사실: 대각선(TLS 후보)이 5.168GHz 근처의 계통오차
가로줄과 실제로 "교차"합니다. 교차 지점 이전(V≈-56.80~-56.35V,
앵커~약 col11)까지는 앵커+양방향 추적이 서브-MHz 정확도로 정확하게
작동했지만, 교차 이후로는 진짜 신호 자체가 배제 구간에 걸려
추적이 무너집니다.

이 스크립트는 "교차 이전 구간만" 잘라서 사용합니다 - 억지로 전체
21점을 다 쓰려다 부정확한 점을 섞는 것보다, 적더라도 확실히 맞는
점들로만 기울기를 구하는 게 더 정직하고 신뢰할 수 있는 결과를 줍니다.
"""

import h5py
import numpy as np
from scipy.optimize import minimize

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
aa_target_raw_index = 12
aa_observable = 'dispamp'
aa_freq_window = (5.12e9, 5.20e9)
aa_max_jump_freq = 8e6
aa_anchor_voltage = -56.80
aa_anchor_freq = 5.135e9
aa_new_exclusion_center = 5.168e9
aa_new_exclusion_halfwidth = 2e6
aa_crossing_voltage_cutoff = -56.60
    # [재수정] 실제 데이터로 확인한 결과, -56.35는 너무 늦어서 배제
    # 구간(5.168GHz 근처)에 갇힌 "평평한 꼬리" 5개 포인트가 섞여
    # 들어갔음(V=-56.55~-56.35에서 주파수가 5.166GHz에 고정되는 게
    # 실제로 관측됨). 상승 구간이 실제로 끝나는 지점인 V=-56.60까지만
    # 사용하도록 앞당김 - 9개 포인트만 남지만, 훨씬 깨끗한 상승 구간.


def mat_string(f, dataset_or_ref):
    try:
        if isinstance(dataset_or_ref, h5py.Reference):
            dataset_or_ref = f[dataset_or_ref]
        arr = np.array(dataset_or_ref)
        return ''.join(chr(int(c)) for c in arr.flatten())
    except Exception:
        return None


with h5py.File(aa_mat_filepath, 'r') as f:
    qdat_refs = f['qdats']['qdat']
    calib_field = np.array(f['qset']['calibdat']['field']).flatten()
    calib_freq = np.array(f['qset']['calibdat']['qubitfreq']).flatten()
    known_artifact_centers = [0.045, -0.030, -0.065]
    artifact_halfwidth = 0.006

    target_group = f[qdat_refs[aa_target_raw_index, 0]]
    sweep1vals = np.array(target_group['sweep1vals']).flatten()
    sweep2_full = np.array(target_group['sweep2vals']).flatten()

    order = np.argsort(calib_field)
    freq_full = np.interp(sweep2_full, calib_field[order], calib_freq[order])

    row_ok = np.ones(len(sweep2_full), dtype=bool)
    for center in known_artifact_centers:
        row_ok &= (np.abs(sweep2_full - center) > artifact_halfwidth)
    row_ok &= (freq_full >= aa_freq_window[0]) & (freq_full <= aa_freq_window[1])
    row_ok &= (np.abs(freq_full - aa_new_exclusion_center) > aa_new_exclusion_halfwidth)

    freq_search = freq_full[row_ok]

    obs_val_refs = target_group['obs']['vals']
    obs_name_refs = target_group['observables']
    target_col = None
    for i in range(obs_name_refs.shape[0]):
        if mat_string(f, obs_name_refs[i, 0]) == aa_observable:
            target_col = i
            break
    data_2d = np.array(f[obs_val_refs[target_col, 0]])
    col_median = np.median(data_2d, axis=0, keepdims=True)
    normalized = data_2d - col_median
    data_search = normalized[row_ok, :]

    anchor_col_idx = np.argmin(np.abs(sweep1vals - aa_anchor_voltage))

    def find_nearby(target, freq_search, max_jump):
        dist = np.abs(freq_search - target)
        nearby = np.where(dist < max_jump)[0]
        return nearby if len(nearby) > 0 else None

    nearby0 = find_nearby(aa_anchor_freq, freq_search, aa_max_jump_freq)
    idx0 = nearby0[np.argmin(data_search[nearby0, anchor_col_idx])]
    dip_freq = {anchor_col_idx: freq_search[idx0]}
    last_f = freq_search[idx0]

    # [수정] 오른쪽 방향 추적을 crossing_voltage_cutoff 이전까지만,
    # + [신규] 주파수가 정체(plateau)되는 순간(배제구간 가장자리에
    # 갇힌 신호로 의심) 자동으로 추적을 조기 종료
    aa_plateau_freq_tolerance = 0.5e6
        # [신규] 연속 두 포인트의 주파수 차이가 이보다 작으면(거의
        # 안 움직이면) "정체"로 판단. 진짜 TLS는 계속 매끄럽게
        # 이동해야 하므로, 정체는 배제구간에 갇혔다는 신호로 봄.
    for i in range(anchor_col_idx+1, len(sweep1vals)):
        if sweep1vals[i] > aa_crossing_voltage_cutoff:
            break   # 교차 지점 넘어서면 추적 중단
        nearby = find_nearby(last_f, freq_search, aa_max_jump_freq)
        if nearby is None:
            break
        idx = nearby[np.argmin(data_search[nearby, i])]
        new_f = freq_search[idx]
        if abs(new_f - last_f) < aa_plateau_freq_tolerance:
            print(f"  [자동 중단] V={sweep1vals[i]:.3f}에서 주파수 정체 감지 "
                  f"(변화량 {abs(new_f-last_f)/1e6:.3f}MHz < 기준) - 추적 조기 종료")
            break
        dip_freq[i] = new_f
        last_f = new_f

    # 왼쪽 방향은 그대로(교차와 무관한 방향)
    last_b = freq_search[idx0]
    for i in range(anchor_col_idx-1, -1, -1):
        nearby = find_nearby(last_b, freq_search, aa_max_jump_freq)
        if nearby is None:
            break
        idx = nearby[np.argmin(data_search[nearby, i])]
        dip_freq[i] = freq_search[idx]
        last_b = freq_search[idx]

    cols_used = sorted(dip_freq.keys())
    v_used = sweep1vals[cols_used]
    freq_used = np.array([dip_freq[c] for c in cols_used])

    print(f"[교차 이전 구간만 사용] 사용된 포인트 수: {len(cols_used)}/21")
    for vv, ff in zip(v_used, freq_used):
        print(f"  V={vv:.3f}, freq={ff/1e9:.5f}GHz")

    def neg_log_lik_gaussian(params, x, y, sigma):
        slope, intercept = params
        return 0.5 * np.sum(((y-(slope*x+intercept))/sigma)**2)

    def neg_log_lik_robust(params, x, y, sigma, nu=4.0):
        slope, intercept = params
        return 0.5*(nu+1) * np.sum(np.log1p(((y-(slope*x+intercept))/sigma)**2/nu))

    sigma_g = np.std(freq_used)
    res_g = minimize(neg_log_lik_gaussian, [0, np.median(freq_used)],
                       args=(v_used, freq_used, sigma_g), method='Nelder-Mead')
    slope_g, intercept_g = res_g.x

    residual_g = freq_used - (slope_g*v_used + intercept_g)
    mad = np.median(np.abs(residual_g - np.median(residual_g)))
    sigma_r = mad * 1.4826 if mad > 0 else sigma_g
    res_r = minimize(neg_log_lik_robust, res_g.x, args=(v_used, freq_used, sigma_r), method='Nelder-Mead')
    slope_r, intercept_r = res_r.x

    rel_diff = abs(slope_g-slope_r)/max(abs(slope_g),1e-10)*100
    print(f"\n[Gaussian] 기울기={slope_g/1e6:.2f} MHz/V")
    print(f"[Robust]   기울기={slope_r/1e6:.2f} MHz/V")
    print(f"기울기 상대차이: {rel_diff:.1f}%")
    print(f"(육안 추정: 약 87 MHz/V)")
