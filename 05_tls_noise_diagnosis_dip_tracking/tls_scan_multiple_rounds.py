"""
tls_scan_multiple_rounds.py
=================================================================
[목적] 오늘 확립한 검증된 방법(앵커 추적 + 정체 자동감지 + 대비
기준 3.0)을, 여러 라운드(여러 전압 구간)에 자동으로 반복 적용해서
"대비점수가 가장 높은" 새 TLS 후보를 효율적으로 찾습니다.

[효율화 포인트]
raw_index=12(alpha, V=-57~-56)에서 하나를 찾는 데 오늘 오랜 시간이
걸렸습니다. 이번엔 "라운드 하나씩 수동으로 파고들기"가 아니라,
여러 라운드(예: 10개 라운드 = 40개 segment)를 한 번에 스캔해서,
find_best_trace()가 "가장 자신 있어 하는"(대비점수 최고) 것부터
확인합니다 - 사람이 육안으로 하나씩 보는 대신, 스크리닝을 코드가
먼저 하고 사람은 최상위 후보만 검토하는 방식.
"""

import h5py
import numpy as np
from scipy.optimize import minimize

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
aa_n_rounds_to_scan = 15
    # [조정] 처음 15개 라운드(=60개 segment, raw_index 0~59)를 스캔.
    # 시간이 오래 걸리면 이 값을 2~3으로 낮춰 먼저 로직만 빠르게
    # 확인한 뒤 늘리는 것을 권장 (segment 하나당 약 6개 앵커 후보를
    # 시도하므로, 60개 segment면 360번의 추적 시도가 발생).
aa_observable = 'dispamp'
aa_freq_window = (5.02e9, 5.25e9)
    # [조정] 오늘 확인된 calibdat 전체 범위(5.035~5.258GHz)에 맞춰 넉넉히.
aa_max_jump_freq = 8e6
aa_new_exclusion_center = 5.168e9
aa_new_exclusion_halfwidth = 2e6
aa_plateau_freq_tolerance = 0.5e6
aa_min_trace_points = 6
aa_contrast_threshold = 3.5
    # [조정] 오늘 검증된 기준(3.0)보다 살짝 높여서, 상위 후보만
    # 걸러지도록 함(alpha 성공 사례의 대비점수가 더 높았던 걸 참고).
known_artifact_centers = [0.045, -0.030, -0.065]
artifact_halfwidth = 0.006


def mat_string(f, dataset_or_ref):
    try:
        if isinstance(dataset_or_ref, h5py.Reference):
            dataset_or_ref = f[dataset_or_ref]
        arr = np.array(dataset_or_ref)
        return ''.join(chr(int(c)) for c in arr.flatten())
    except Exception:
        return None


def track_from_anchor(anchor_col, anchor_freq_guess, sweep1vals, freq_search, data_search, max_jump):
    def find_nearby(target):
        dist = np.abs(freq_search - target)
        nearby = np.where(dist < max_jump)[0]
        return nearby if len(nearby) > 0 else None
    nearby0 = find_nearby(anchor_freq_guess)
    if nearby0 is None:
        return {}
    idx0 = nearby0[np.argmin(data_search[nearby0, anchor_col])]
    trace = {anchor_col: freq_search[idx0]}
    last_f = freq_search[idx0]
    for i in range(anchor_col+1, len(sweep1vals)):
        nearby = find_nearby(last_f)
        if nearby is None:
            break
        idx = nearby[np.argmin(data_search[nearby, i])]
        new_f = freq_search[idx]
        if abs(new_f - last_f) < aa_plateau_freq_tolerance:
            break
        trace[i] = new_f
        last_f = new_f
    last_b = freq_search[idx0]
    for i in range(anchor_col-1, -1, -1):
        nearby = find_nearby(last_b)
        if nearby is None:
            break
        idx = nearby[np.argmin(data_search[nearby, i])]
        new_f = freq_search[idx]
        if abs(new_f - last_b) < aa_plateau_freq_tolerance:
            break
        trace[i] = new_f
        last_b = new_f
    return trace


def find_best_trace(sweep1vals, freq_search, data_search, max_jump, min_points):
    n_cols = data_search.shape[1]
    candidate_cols = np.linspace(0, n_cols-1, min(6, n_cols)).astype(int)
    background_std = np.std(data_search)
    best_trace, best_score = {}, -1
    for anchor_col in candidate_cols:
        anchor_freq_guess = freq_search[np.argmin(data_search[:, anchor_col])]
        trace = track_from_anchor(anchor_col, anchor_freq_guess, sweep1vals,
                                     freq_search, data_search, max_jump)
        if len(trace) < min_points:
            continue
        cols = sorted(trace.keys())
        freqs = np.array([trace[c] for c in cols])
        diffs = np.diff(freqs)
        if len(diffs) == 0:
            continue
        monotonicity = abs(np.sum(np.sign(diffs))) / len(diffs)
        score = len(trace) * (0.5 + 0.5*monotonicity)
        if score > best_score:
            best_score = score
            best_trace = trace
    if len(best_trace) < min_points:
        return {}, 0.0
    cols = sorted(best_trace.keys())
    depths = []
    for c in cols:
        idx_at_dip = np.argmin(np.abs(freq_search - best_trace[c]))
        depths.append(-data_search[idx_at_dip, c])
    contrast = np.mean(depths) / (background_std + 1e-12)
    return best_trace, contrast


with h5py.File(aa_mat_filepath, 'r') as f:
    qdat_refs = f['qdats']['qdat']
    calib_field = np.array(f['qset']['calibdat']['field']).flatten()
    calib_freq = np.array(f['qset']['calibdat']['qubitfreq']).flatten()
    order = np.argsort(calib_field)

    n_segments_to_scan = aa_n_rounds_to_scan * 4
    all_results = []

    for raw_idx in range(n_segments_to_scan):
        qdat_group = f[qdat_refs[raw_idx, 0]]
        electrode = mat_string(f, qdat_group['sweep1name'])
        sweep1vals = np.array(qdat_group['sweep1vals']).flatten()
        sweep2_full = np.array(qdat_group['sweep2vals']).flatten()
        freq_full = np.interp(sweep2_full, calib_field[order], calib_freq[order])

        row_ok = np.ones(len(sweep2_full), dtype=bool)
        for center in known_artifact_centers:
            row_ok &= (np.abs(sweep2_full - center) > artifact_halfwidth)
        row_ok &= (freq_full >= aa_freq_window[0]) & (freq_full <= aa_freq_window[1])
        row_ok &= (np.abs(freq_full - aa_new_exclusion_center) > aa_new_exclusion_halfwidth)
        freq_search = freq_full[row_ok]

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
        data_search = normalized[row_ok, :]

        trace, contrast = find_best_trace(sweep1vals, freq_search, data_search,
                                             aa_max_jump_freq, aa_min_trace_points)
        if len(trace) >= aa_min_trace_points and contrast >= aa_contrast_threshold:
            cols = sorted(trace.keys())
            v_used = sweep1vals[cols]
            freq_used = np.array([trace[c] for c in cols])
            coeffs = np.polyfit(v_used, freq_used, deg=1)
            all_results.append({
                'raw_index': raw_idx, 'electrode': electrode,
                'v_range': (v_used.min(), v_used.max()),
                'n_points': len(cols), 'contrast': contrast,
                'slope_MHz_per_V': coeffs[0]/1e6,
            })

    all_results.sort(key=lambda r: -r['contrast'])
    print(f"스캔 완료: raw_index 0~{n_segments_to_scan-1} 중 "
          f"{len(all_results)}개 후보 발견 (대비≥{aa_contrast_threshold})\n")
    print(f"{'raw_idx':>8} {'전극':>6} {'전압범위':>16} {'포인트수':>8} {'대비':>8} {'기울기(MHz/V)':>14}")
    print("-"*70)
    for r in all_results[:15]:
        vr = r['v_range']
        print(f"{r['raw_index']:>8} {r['electrode']:>6} [{vr[0]:>5.1f},{vr[1]:>5.1f}] "
              f"{r['n_points']:>8} {r['contrast']:>8.2f} {r['slope_MHz_per_V']:>14.1f}")
