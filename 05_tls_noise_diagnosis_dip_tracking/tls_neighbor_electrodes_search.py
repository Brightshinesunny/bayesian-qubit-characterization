"""
tls_neighbor_electrodes_search.py
=================================================================
[목적] alpha(raw_index=12)에서 성공한 방법(앵커 추적 + 정체 자동
감지 + 교차 전 컷)을, 같은 라운드의 이웃(13=beta, 14=gamma,
15=delta)에 자동으로 적용합니다.

[알파와의 차이 - 앵커를 수동으로 안 정하는 이유]
alpha는 육안으로 확인된 시작점(V=-56.80,f=5.135GHz)이 있었지만,
beta/gamma/delta는 그런 사전 정보가 없습니다. 대신, 이 segment
안의 "가능한 모든 시작 열"을 앵커 후보로 삼아 각각 추적해보고,
가장 길게(정체 없이) 이어지는 궤적을 고르는 자동 탐색 방식을 씁니다.
"""

import h5py
import numpy as np
from scipy.optimize import minimize

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
aa_neighbor_raw_indices = [13, 14, 15]   # beta, gamma, delta (같은 라운드)
aa_observable = 'dispamp'
aa_freq_window = (5.12e9, 5.20e9)
aa_max_jump_freq = 8e6
aa_new_exclusion_center = 5.168e9
aa_new_exclusion_halfwidth = 2e6
aa_plateau_freq_tolerance = 0.5e6
aa_min_trace_points = 5
aa_contrast_threshold = 3.0
    # [신규] 딥 깊이(대비)가 배경 표준편차의 이 배수 이상이어야 "진짜
    # 신호"로 인정. 합성 데이터 검증 결과, 순수 잡음도 길이/단조성
    # 기준만으로는 통과해버려서 이 절대적 기준이 반드시 필요함.
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
    """앵커에서 시작해 양방향으로 추적하고, 정체가 감지되면 그 방향은 중단."""
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
    """
    [신규] 앵커를 수동 지정하지 않고, 배경보다 뚜렷이 어두운 지점들을
    후보 앵커로 삼아 각각 추적한 뒤, 가장 길고(포인트 수) 매끄러운
    (단조 증가/감소) 궤적을 자동으로 선택.

    [버그 수정] 길이/단조성만으로 점수를 매기면, 순수 잡음에서도
    "그럴듯한 궤적"을 찾아버리는 문제가 합성 데이터 검증에서 발견됨
    (21개 포인트가 잡음만으로도 나옴). 이를 막기 위해, 최종 선택된
    궤적의 "딥 깊이(대비, contrast)"가 배경 표준편차 대비 충분히
    커야만(quality_threshold 이상) 진짜 신호로 인정하도록 추가 검증.
    """
    n_cols = data_search.shape[1]
    candidate_cols = np.linspace(0, n_cols-1, min(8, n_cols)).astype(int)
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

    # [신규] 최종 선택된 궤적의 딥 깊이(대비) 계산
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

    results = {}
    for raw_idx in aa_neighbor_raw_indices:
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

        print(f"\n{'='*60}\nraw_index={raw_idx} (전극={electrode})\n{'='*60}")
        trace, contrast = find_best_trace(sweep1vals, freq_search, data_search, aa_max_jump_freq, aa_min_trace_points)

        if len(trace) < aa_min_trace_points or contrast < aa_contrast_threshold:
            print(f"  뚜렷한 궤적 없음 (포인트수={len(trace)}, 대비점수={contrast:.2f} "
                  f"< 기준 {aa_contrast_threshold}) -> gamma≈0으로 기록")
            results[electrode] = None
            continue

        cols = sorted(trace.keys())
        v_used = sweep1vals[cols]
        freq_used = np.array([trace[c] for c in cols])
        print(f"  자동 선택된 궤적: {len(cols)}개 포인트, 대비점수={contrast:.2f}")
        for vv, ff in zip(v_used, freq_used):
            print(f"    V={vv:.3f}, freq={ff/1e9:.5f}GHz")

        def neg_log_lik_gaussian(params, x, y, sigma):
            slope, intercept = params
            return 0.5 * np.sum(((y-(slope*x+intercept))/sigma)**2)

        sigma_g = np.std(freq_used)
        res_g = minimize(neg_log_lik_gaussian, [0, np.median(freq_used)],
                           args=(v_used, freq_used, sigma_g), method='Nelder-Mead')
        slope_g = res_g.x[0]
        print(f"  기울기(gamma_{electrode}) = {slope_g/1e6:.2f} MHz/V")
        results[electrode] = slope_g

    print(f"\n{'='*60}\n최종 요약 (alpha 결과 gamma_alpha≈109 MHz/V와 비교)\n{'='*60}")
    for electrode, slope in results.items():
        if slope is None:
            print(f"  {electrode}: 신호 없음 (gamma≈0)")
        else:
            print(f"  {electrode}: gamma≈{slope/1e6:.2f} MHz/V")
