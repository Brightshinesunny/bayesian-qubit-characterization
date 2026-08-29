"""
tls_seg12_v2_raw_index.py
=================================================================
[v2 수정 - 인덱싱 혼동 해결]
  이전 코드는 "alpha 전극만 걸러서 전압순 정렬한 리스트의 12번째"
  (matching_alpha[12])를 썼는데, 전압범위가 [-48,-47]V로 나와
  원래 육안 확인했던 seg12(-57~-56V)와 달랐습니다.

  원인: 실험이 "라운드로빈"(alpha->beta->gamma->delta->alpha->...)
  구조라면, qdats 원본 배열의 raw 인덱스가 4개씩 순환합니다. 즉
  "raw 인덱스 12"가 곧 "alpha 전극의 (12/4=3번째, 즉 -57~-56V)
  segment"일 가능성이 높습니다 - 이게 원래 육안 확인 때 보신 것과
  일치합니다.

  이 스크립트는 먼저 실제 순환 구조를 데이터로 직접 확인한 뒤,
  raw 인덱스 12(및 그 바로 다음 이웃들)를 정확히 지정해서 다시
  추출합니다.
"""

import h5py
import numpy as np
from scipy.optimize import minimize

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
aa_target_raw_index = 12   # [수정] alpha 전극만 걸러낸 리스트가 아니라, qdats 원본의 raw 인덱스
aa_observable = 'dispamp'
aa_max_jump = 0.01
aa_n_check_order = 16   # 순환 구조 확인을 위해 앞의 몇 개 segment의 전극을 볼지


def mat_string(f, dataset_or_ref):
    try:
        if isinstance(dataset_or_ref, h5py.Reference):
            dataset_or_ref = f[dataset_or_ref]
        arr = np.array(dataset_or_ref)
        return ''.join(chr(int(c)) for c in arr.flatten())
    except Exception:
        return None


def extract_trace_single_segment(f, qdat_group, observable, row_ok_mask, sweep2_full,
                                     calib_field, calib_freq, max_jump):
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

    sweep2_search = sweep2_full[row_ok_mask]
    data_search = normalized[row_ok_mask, :]

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

    depths = []
    for i in range(data_search.shape[1]):
        idx_at_dip = np.argmin(np.abs(sweep2_search - dip_positions[i]))
        depths.append(-data_search[idx_at_dip, i])
    quality = np.mean(depths) / (np.std(data_search) + 1e-12)

    return sweep1vals, dip_freq, quality


with h5py.File(aa_mat_filepath, 'r') as f:
    qdat_refs = f['qdats']['qdat']
    calib_field = np.array(f['qset']['calibdat']['field']).flatten()
    calib_freq = np.array(f['qset']['calibdat']['qubitfreq']).flatten()
    known_artifact_centers = [0.045, -0.030, -0.065]
    artifact_halfwidth = 0.006

    # =====================================================
    # STEP 0. 실제 순환 구조 확인 (라운드로빈인지 직접 검증)
    # =====================================================
    print("[STEP 0] 처음 16개 raw segment의 전극 순서 확인:")
    order_check = []
    for i in range(aa_n_check_order):
        qdat_group = f[qdat_refs[i, 0]]
        name = mat_string(f, qdat_group['sweep1name'])
        sweep1vals = np.array(qdat_group['sweep1vals']).flatten()
        order_check.append(name)
        print(f"  raw_index={i}: 전극={name}, 전압범위=[{sweep1vals.min():.1f},{sweep1vals.max():.1f}]")

    # =====================================================
    # STEP 1. raw 인덱스 12(및 이웃)를 직접 지정해서 추출
    # =====================================================
    print(f"\n[STEP 1] raw_index={aa_target_raw_index} 단독 추출")
    target_group = f[qdat_refs[aa_target_raw_index, 0]]
    target_electrode = mat_string(f, target_group['sweep1name'])
    sweep2_full = np.array(target_group['sweep2vals']).flatten()
    print(f"  이 segment의 전극: {target_electrode}")

    row_ok = np.ones(len(sweep2_full), dtype=bool)
    for center in known_artifact_centers:
        row_ok &= (np.abs(sweep2_full - center) > artifact_halfwidth)

    v1_target, freq_target, quality_target = extract_trace_single_segment(
        f, target_group, aa_observable, row_ok, sweep2_full, calib_field, calib_freq, aa_max_jump
    )
    print(f"  전압범위: [{v1_target.min():.2f},{v1_target.max():.2f}]V, 품질점수={quality_target:.3f}")

    def neg_log_lik_gaussian(params, x, y, sigma):
        slope, intercept = params
        return 0.5 * np.sum(((y-(slope*x+intercept))/sigma)**2)

    def neg_log_lik_robust(params, x, y, sigma, nu=4.0):
        slope, intercept = params
        return 0.5*(nu+1) * np.sum(np.log1p(((y-(slope*x+intercept))/sigma)**2/nu))

    sigma_g = np.std(freq_target)
    res_g = minimize(neg_log_lik_gaussian, [0, np.median(freq_target)],
                       args=(v1_target, freq_target, sigma_g), method='Nelder-Mead')
    slope_g, intercept_g = res_g.x

    residual_g = freq_target - (slope_g*v1_target + intercept_g)
    mad = np.median(np.abs(residual_g - np.median(residual_g)))
    sigma_r = mad * 1.4826
    res_r = minimize(neg_log_lik_robust, res_g.x, args=(v1_target, freq_target, sigma_r), method='Nelder-Mead')
    slope_r, intercept_r = res_r.x

    rel_diff = abs(slope_g-slope_r)/max(abs(slope_g),1e-10)*100
    print(f"  [Gaussian] 기울기={slope_g:.1f} Hz/V")
    print(f"  [Robust]   기울기={slope_r:.1f} Hz/V")
    print(f"  기울기 상대차이: {rel_diff:.1f}%")

    # [신규] 잔차가 가장 큰 점들을 출력 - 어떤 점이 튀는지 직접 확인
    residual_r = freq_target - (slope_r*v1_target + intercept_r)
    sorted_idx = np.argsort(-np.abs(residual_r))
    print(f"\n  Robust 잔차가 가장 큰 상위 3개 포인트 (튀는 점 후보):")
    for idx in sorted_idx[:3]:
        print(f"    V={v1_target[idx]:.3f}, freq={freq_target[idx]/1e9:.5f}GHz, "
              f"잔차={residual_r[idx]/1e6:.3f}MHz")

    # =====================================================
    # STEP 2. 같은 라운드의 "바로 다음 이웃 raw 인덱스"로 다른 전극 확인
    # =====================================================
    print(f"\n[STEP 2] raw_index={aa_target_raw_index} 바로 다음 이웃들의 전극 확인 및 추출")
    for offset in [1, 2, 3]:
        neighbor_idx = aa_target_raw_index + offset
        neighbor_group = f[qdat_refs[neighbor_idx, 0]]
        neighbor_electrode = mat_string(f, neighbor_group['sweep1name'])
        sweep2_n = np.array(neighbor_group['sweep2vals']).flatten()

        row_ok_n = np.ones(len(sweep2_n), dtype=bool)
        for center in known_artifact_centers:
            row_ok_n &= (np.abs(sweep2_n - center) > artifact_halfwidth)

        v1_n, freq_n, quality_n = extract_trace_single_segment(
            f, neighbor_group, aa_observable, row_ok_n, sweep2_n, calib_field, calib_freq, aa_max_jump
        )
        print(f"  raw_index={neighbor_idx} (전극={neighbor_electrode}): "
              f"전압범위=[{v1_n.min():.2f},{v1_n.max():.2f}]V, 품질점수={quality_n:.3f} "
              f"(target의 {quality_target:.3f}과 비교)")
