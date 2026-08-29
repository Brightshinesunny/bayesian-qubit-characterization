"""
tls_seg12_isolated_and_crosscheck.py
=================================================================
[제안 반영]
  1. seg 12(alpha)만 단독으로 자취 추출 -> Gaussian/Robust 직선 피팅
  2. beta/gamma/delta는 alpha의 직선을 "복사"하지 않고, 각자 자기
     segment들에서 독립적으로 (seg12와 비슷한 주파수 범위,
     5.13~5.19GHz 근처에서) 뚜렷한 어두운 띠가 있는지 개별 탐색
  3. 없으면 "이 전극에서는 이번 라운드에 안 보인다"고 정직하게 보고
"""

import h5py
import numpy as np
from scipy.optimize import minimize

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
aa_target_electrode = 'ao4'
aa_target_segment_rank = 12   # alpha 전극 중 12번째(0-indexed, 전압 오름차순)
aa_observable = 'dispamp'
aa_max_jump = 0.01
aa_freq_search_range = (5.10e9, 5.22e9)
    # [조정] seg12에서 확인된 진짜 자취 범위(5.13~5.19GHz)보다 살짝
    # 넉넉하게 잡아, 다른 전극에서 비슷한 위치의 신호를 놓치지 않게 함.
aa_other_electrodes = ['ao3', 'ao5', 'ao6']   # gamma, beta, delta


def mat_string(f, dataset_or_ref):
    try:
        if isinstance(dataset_or_ref, h5py.Reference):
            dataset_or_ref = f[dataset_or_ref]
        arr = np.array(dataset_or_ref)
        return ''.join(chr(int(c)) for c in arr.flatten())
    except Exception:
        return None


def extract_trace_single_segment(f, qdat_group, observable, sweep2_search_mask,
                                    sweep2_full, calib_field, calib_freq, max_jump):
    """
    segment 하나에서 연속성 제약 딥 추적으로 (전압, 딥주파수) 궤적을 추출.
    """
    sweep1vals = np.array(qdat_group['sweep1vals']).flatten()
    obs_val_refs = qdat_group['obs']['vals']
    obs_name_refs = qdat_group['observables']
    target_col = None
    for i in range(obs_name_refs.shape[0]):
        if mat_string(f, obs_name_refs[i, 0]) == observable:
            target_col = i
            break
    data_2d = np.array(f[obs_val_refs[target_col, 0]])
    col_median = np.median(data_2d, axis=0, keepdims=True)
    normalized = data_2d - col_median

    sweep2_search = sweep2_full[sweep2_search_mask]
    data_search = normalized[sweep2_search_mask, :]

    dip_positions, last = [], None
    for i in range(data_search.shape[1]):
        col = data_search[:, i]
        if last is None:
            idx = np.argmin(col)
        else:
            dist = np.abs(sweep2_search - last)
            nearby = dist < max_jump
            if np.any(nearby):
                local = np.where(nearby)[0]
                idx = local[np.argmin(col[local])]
            else:
                idx = np.argmin(col)
        last = sweep2_search[idx]
        dip_positions.append(last)
    dip_positions = np.array(dip_positions)

    order = np.argsort(calib_field)
    dip_freq = np.interp(dip_positions, calib_field[order], calib_freq[order])

    # 신호 품질 점수: 딥 깊이(배경 대비 얼마나 어두운지)의 평균
    depths = []
    for i in range(data_search.shape[1]):
        idx_at_dip = np.argmin(np.abs(sweep2_search - dip_positions[i]))
        depths.append(-data_search[idx_at_dip, i])   # 어두울수록(값이 작을수록) 깊이가 큼(양수)
    quality_score = np.mean(depths) / (np.std(data_search) + 1e-12)
        # 배경 표준편차 대비 딥 깊이의 비율 - 일종의 신호대잡음비(SNR).
        # 값이 클수록 "진짜 뚜렷한 자취"일 가능성이 높음.

    return sweep1vals, dip_freq, quality_score


with h5py.File(aa_mat_filepath, 'r') as f:
    qdat_refs = f['qdats']['qdat']
    n_segments = qdat_refs.shape[0]

    calib_field = np.array(f['qset']['calibdat']['field']).flatten()
    calib_freq = np.array(f['qset']['calibdat']['qubitfreq']).flatten()

    known_artifact_centers = [0.045, -0.030, -0.065]
    artifact_halfwidth = 0.006

    # =====================================================
    # STEP 1. seg 12(alpha) 단독 정밀 피팅
    # =====================================================
    matching_alpha = []
    for i in range(n_segments):
        qdat_group = f[qdat_refs[i, 0]]
        if mat_string(f, qdat_group['sweep1name']) == aa_target_electrode:
            sweep1vals = np.array(qdat_group['sweep1vals']).flatten()
            matching_alpha.append({'index': i, 'v_min': sweep1vals.min()})
    matching_alpha.sort(key=lambda s: s['v_min'])

    seg12_info = matching_alpha[aa_target_segment_rank]
    seg12_group = f[qdat_refs[seg12_info['index'], 0]]
    sweep2_full = np.array(seg12_group['sweep2vals']).flatten()

    row_ok = np.ones(len(sweep2_full), dtype=bool)
    for center in known_artifact_centers:
        row_ok &= (np.abs(sweep2_full - center) > artifact_halfwidth)

    v1_seg12, freq_seg12, quality_seg12 = extract_trace_single_segment(
        f, seg12_group, aa_observable, row_ok, sweep2_full, calib_field, calib_freq, aa_max_jump
    )
    print(f"[STEP 1] seg12(alpha, index={seg12_info['index']}) 단독 추출")
    print(f"  전압범위: [{v1_seg12.min():.2f},{v1_seg12.max():.2f}]V, 포인트수: {len(v1_seg12)}")
    print(f"  품질점수(SNR 유사): {quality_seg12:.3f}")

    def neg_log_lik_gaussian(params, x, y, sigma):
        slope, intercept = params
        return 0.5 * np.sum(((y-(slope*x+intercept))/sigma)**2)

    def neg_log_lik_robust(params, x, y, sigma, nu=4.0):
        slope, intercept = params
        return 0.5*(nu+1) * np.sum(np.log1p(((y-(slope*x+intercept))/sigma)**2/nu))

    sigma_g = np.std(freq_seg12)
    res_g = minimize(neg_log_lik_gaussian, [0, np.median(freq_seg12)],
                       args=(v1_seg12, freq_seg12, sigma_g), method='Nelder-Mead')
    slope_g, intercept_g = res_g.x

    residual_g = freq_seg12 - (slope_g*v1_seg12 + intercept_g)
    mad = np.median(np.abs(residual_g - np.median(residual_g)))
    sigma_r = mad * 1.4826
    res_r = minimize(neg_log_lik_robust, res_g.x, args=(v1_seg12, freq_seg12, sigma_r), method='Nelder-Mead')
    slope_r, intercept_r = res_r.x

    rel_diff = abs(slope_g-slope_r)/max(abs(slope_g),1e-10)*100
    print(f"  [Gaussian] 기울기={slope_g:.1f} Hz/V")
    print(f"  [Robust]   기울기={slope_r:.1f} Hz/V")
    print(f"  기울기 상대차이: {rel_diff:.1f}% (단독 segment라 훨씬 작아야 정상)")

    # =====================================================
    # STEP 2. 다른 3개 전극 - "복사"가 아니라 독립적으로 탐색
    # =====================================================
    print(f"\n[STEP 2] beta/gamma/delta에서 독립적으로 비슷한 신호 탐색 "
          f"(주파수 {aa_freq_search_range[0]/1e9:.2f}~{aa_freq_search_range[1]/1e9:.2f}GHz 근처)")

    for electrode in aa_other_electrodes:
        matching = []
        for i in range(n_segments):
            qdat_group = f[qdat_refs[i, 0]]
            if mat_string(f, qdat_group['sweep1name']) == electrode:
                matching.append(i)

        best_quality, best_seg_idx, best_result = -np.inf, None, None
        for seg_idx in matching:
            qdat_group = f[qdat_refs[seg_idx, 0]]
            sweep2_this = np.array(qdat_group['sweep2vals']).flatten()
            freq_this_full = np.interp(sweep2_this,
                                          calib_field[np.argsort(calib_field)],
                                          calib_freq[np.argsort(calib_field)])
            # 이 segment가 목표 주파수 범위와 겹치는지 확인
            if freq_this_full.max() < aa_freq_search_range[0] or freq_this_full.min() > aa_freq_search_range[1]:
                continue

            row_ok_this = np.ones(len(sweep2_this), dtype=bool)
            for center in known_artifact_centers:
                row_ok_this &= (np.abs(sweep2_this - center) > artifact_halfwidth)
            row_ok_this &= (freq_this_full >= aa_freq_search_range[0]) & (freq_this_full <= aa_freq_search_range[1])

            if np.sum(row_ok_this) < 5:
                continue

            v1_this, freq_this, quality_this = extract_trace_single_segment(
                f, qdat_group, aa_observable, row_ok_this, sweep2_this, calib_field, calib_freq, aa_max_jump
            )
            if quality_this > best_quality:
                best_quality = quality_this
                best_seg_idx = seg_idx
                best_result = (v1_this, freq_this)

        if best_seg_idx is not None:
            print(f"  {electrode}: 가장 유력한 segment={best_seg_idx}, 품질점수={best_quality:.3f} "
                  f"(seg12의 {quality_seg12:.3f}과 비교)")
        else:
            print(f"  {electrode}: 목표 주파수 범위와 겹치는 segment 없음 - 이번 라운드에 데이터 없음")
