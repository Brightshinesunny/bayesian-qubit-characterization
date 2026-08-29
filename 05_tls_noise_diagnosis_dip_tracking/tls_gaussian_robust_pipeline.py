"""
tls_gaussian_robust_pipeline.py
=================================================================
[제안된 순서 그대로 구현]
  1. segment 몇 장만 가볍게 dispamp로 확인
  2. 가벼운 이상치 마스크(3%)
  3. 가우시안 우도로 "자취(trace)/기울기"를 피팅
  4. 같은 모델에 robust sigma만 바꿔 비교
  5. 잔차의 짧은/긴 자기상관은 그 다음(이번엔 다루지 않음)

[여기서 "자취(trace) 모델"이 뭘 의미하는가]
전압(V)에 따라 신호의 딥(공명 후보) 위치가 어떻게 움직이는지를
"직선(1차 다항식)"으로 표현합니다:
    dip_position(V) = slope * V + intercept
가우시안 우도는 "딥 위치의 오차가 정규분포를 따르고 서로 독립"이라고
가정하는 표준 최소제곱법과 같습니다. Robust는 이상치(잘못 추출된
딥 위치)에 덜 흔들리는 대안입니다.
"""

import h5py
import numpy as np
from scipy.optimize import minimize

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
aa_electrode = 'ao4'
aa_n_segments_light = 5
    # [제안 반영] "세그먼트 몇 장만" - 전체 160개가 아니라 5개만 가볍게 확인.
aa_observable = 'dispamp'
aa_outlier_pct = 3.0
    # [제안 반영] "가벼운 이상치 마스크 3%".
aa_known_artifact_centers = [0.045, -0.030, -0.065]
aa_artifact_halfwidth = 0.006


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
    n_segments = qdat_refs.shape[0]

    # =====================================================
    # STEP 1. segment 5개만 가볍게 선택 (alpha 전극, 전압 낮은 순)
    # =====================================================
    matching = []
    for i in range(n_segments):
        qdat_group = f[qdat_refs[i, 0]]
        if mat_string(f, qdat_group['sweep1name']) == aa_electrode:
            sweep1vals = np.array(qdat_group['sweep1vals']).flatten()
            matching.append({'index': i, 'v_min': sweep1vals.min()})
    matching.sort(key=lambda s: s['v_min'])
    selected = matching[:aa_n_segments_light]
    print(f"선택된 segment: {[s['index'] for s in selected]}, "
          f"전압 범위: {[s['v_min'] for s in selected]}")

    first_sweep2 = np.array(f[qdat_refs[selected[0]['index'], 0]]['sweep2vals']).flatten()

    all_v1, all_data_rows = [], []
    for s in selected:
        qdat_group = f[qdat_refs[s['index'], 0]]
        sweep1vals = np.array(qdat_group['sweep1vals']).flatten()
        obs_val_refs = qdat_group['obs']['vals']
        obs_name_refs = qdat_group['observables']
        target_col = None
        for i in range(obs_name_refs.shape[0]):
            if mat_string(f, obs_name_refs[i, 0]) == aa_observable:
                target_col = i
                break
        data_2d = np.array(f[obs_val_refs[target_col, 0]])
        all_v1.extend(sweep1vals.tolist())
        all_data_rows.append(data_2d)

    stitched = np.concatenate(all_data_rows, axis=1)
    all_v1 = np.array(all_v1)
    col_median = np.median(stitched, axis=0, keepdims=True)
    normalized = stitched - col_median

    print(f"\n이번에 다룰 데이터 shape: {normalized.shape} "
          f"(전체 160 segment가 아니라 {aa_n_segments_light}개만 사용)")

    # =====================================================
    # STEP 2. 가벼운 이상치 마스크(3%) - 알려진 계통오차 구간을
    # 제외한 배경에서, 극단값 상위/하위 3%를 마스킹
    # =====================================================
    row_ok = np.ones(len(first_sweep2), dtype=bool)
    for center in aa_known_artifact_centers:
        row_ok &= (np.abs(first_sweep2 - center) > aa_artifact_halfwidth)
    sweep2_search = first_sweep2[row_ok]
    data_search = normalized[row_ok, :]

    flat_vals = data_search.flatten()
    lo_thresh, hi_thresh = np.percentile(flat_vals, [aa_outlier_pct/2, 100-aa_outlier_pct/2])
    outlier_mask = (data_search < lo_thresh) | (data_search > hi_thresh)
    n_outliers = np.sum(outlier_mask)
    print(f"\n3% 이상치 마스킹: {n_outliers}개 픽셀 제외 "
          f"(전체 {data_search.size}개 중 {n_outliers/data_search.size*100:.2f}%)")
    data_search_masked = np.where(outlier_mask, np.nan, data_search)

    # 딥 위치 추출(이상치 제외된 상태에서 최솟값 탐색)
    dip_positions = []
    for i in range(data_search_masked.shape[1]):
        col = data_search_masked[:, i]
        if np.all(np.isnan(col)):
            dip_positions.append(np.nan)
        else:
            dip_positions.append(sweep2_search[np.nanargmin(col)])
    dip_positions = np.array(dip_positions)
    valid = ~np.isnan(dip_positions)

    # =====================================================
    # STEP 3. 가우시안 우도로 직선(자취) 피팅
    # =====================================================
    def neg_log_lik_gaussian(params, x, y, sigma):
        slope, intercept = params
        residual = y - (slope*x + intercept)
        return 0.5 * np.sum((residual/sigma)**2)

    x_data = all_v1[valid]
    y_data = dip_positions[valid]
    sigma_baseline = np.std(y_data)

    res_gauss = minimize(neg_log_lik_gaussian, [0.0, np.median(y_data)],
                           args=(x_data, y_data, sigma_baseline), method='Nelder-Mead')
    slope_gauss, intercept_gauss = res_gauss.x
    print(f"\n[가우시안 우도 피팅] 기울기={slope_gauss:.8f}, 절편={intercept_gauss:.6f}")

    # =====================================================
    # STEP 4. 같은 모델, robust sigma(MAD)로 재비교
    # =====================================================
    residual_gauss = y_data - (slope_gauss*x_data + intercept_gauss)
    mad = np.median(np.abs(residual_gauss - np.median(residual_gauss)))
    sigma_robust = mad * 1.4826

    def neg_log_lik_robust(params, x, y, sigma, nu=4.0):
        slope, intercept = params
        residual = y - (slope*x + intercept)
        return 0.5*(nu+1) * np.sum(np.log1p((residual/sigma)**2/nu))

    res_robust = minimize(neg_log_lik_robust, [slope_gauss, intercept_gauss],
                            args=(x_data, y_data, sigma_robust), method='Nelder-Mead')
    slope_robust, intercept_robust = res_robust.x
    print(f"[Robust(Student-t) 피팅] 기울기={slope_robust:.8f}, 절편={intercept_robust:.6f}")

    print(f"\n[비교] 기울기 차이: {abs(slope_gauss-slope_robust):.8f} "
          f"(상대차이 {abs(slope_gauss-slope_robust)/max(abs(slope_gauss),1e-10)*100:.1f}%)")
    print(f"sigma_baseline(단순std)={sigma_baseline:.6f} vs sigma_robust(MAD)={sigma_robust:.6f}")
