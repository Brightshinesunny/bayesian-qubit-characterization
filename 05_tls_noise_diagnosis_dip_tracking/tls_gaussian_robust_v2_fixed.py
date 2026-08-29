"""
tls_gaussian_robust_v2_fixed.py
=================================================================
[v1의 버그 수정 - 사용자 진단 반영]
  버그: 2D 맵에서 미리 상/하위 3%를 마스킹한 뒤 nanargmin으로 딥을
  찾음 -> TLS 신호(=딥, 낮은 값) 자체가 이상치로 먼저 잘려나가서,
  남은 배경 요철을 따라 딥이 열마다 엉뚱한 곳으로 점프함.

  수정 1: 마스킹 없이 원본 맵에서, "이전 열의 딥 위치 근처(max_jump
    이내)에서만" 최솟값을 찾는 연속성 제약 추적으로 변경. 이렇게 하면
    자취가 한 TLS를 계속 따라가고, 배경 요철으로 튀지 않음.
  수정 2: 이상치 마스킹은 추출된 (V, dip위치) 1차원 점들에 대해서만
    적용 (2D 원본 맵이 아니라).
  수정 3: sweep2vals(플럭스 진폭)를 qset.calibdat로 실제 주파수(Hz)로
    변환 - tdat와 직접 비교 가능한 물리적 단위로 맞춤.
"""

import h5py
import numpy as np
from scipy.optimize import minimize

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
aa_electrode = 'ao4'
aa_n_segments_light = 5
aa_observable = 'dispamp'
aa_outlier_pct = 3.0
aa_known_artifact_centers = [0.045, -0.030, -0.065]
aa_artifact_halfwidth = 0.006
aa_max_jump = 0.01
    # [신규] 연속성 제약: 이전 열의 딥 위치에서 이 폭(sweep2 단위)
    # 이내에서만 다음 딥을 찾음. 너무 좁으면 진짜 TLS 이동을 놓칠 수
    # 있고, 너무 넓으면 배경으로 다시 튈 수 있음 - 절충값.


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
    # STEP 0. calibdat로 sweep2(플럭스) -> 실제 주파수(Hz) 변환 준비
    # =====================================================
    calib_field = np.array(f['qset']['calibdat']['field']).flatten()
    calib_freq = np.array(f['qset']['calibdat']['qubitfreq']).flatten()
    print(f"캘리브레이션: field 범위=[{calib_field.min():.4f},{calib_field.max():.4f}], "
          f"freq 범위=[{calib_freq.min()/1e9:.4f},{calib_freq.max()/1e9:.4f}]GHz")

    def flux_to_freq(flux_vals):
        # np.interp: 선형 보간 - calib_field/calib_freq로 만든
        # 대응표를 이용해, 임의의 flux 값에서의 주파수를 추정.
        # calib_field가 오름차순이어야 하므로 정렬 필요.
        order = np.argsort(calib_field)
        return np.interp(flux_vals, calib_field[order], calib_freq[order])

    # =====================================================
    # STEP 1. segment 5개 선택 및 이어붙이기 (v1과 동일)
    # =====================================================
    matching = []
    for i in range(n_segments):
        qdat_group = f[qdat_refs[i, 0]]
        if mat_string(f, qdat_group['sweep1name']) == aa_electrode:
            sweep1vals = np.array(qdat_group['sweep1vals']).flatten()
            matching.append({'index': i, 'v_min': sweep1vals.min()})
    matching.sort(key=lambda s: s['v_min'])
    selected = matching[:aa_n_segments_light]

    first_sweep2 = np.array(f[qdat_refs[selected[0]['index'], 0]]['sweep2vals']).flatten()
    first_freq = flux_to_freq(first_sweep2)

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

    # 계통오차 구간 제외 (마스킹이 아니라 "탐색 후보에서 제외"만)
    row_ok = np.ones(len(first_sweep2), dtype=bool)
    for center in aa_known_artifact_centers:
        row_ok &= (np.abs(first_sweep2 - center) > aa_artifact_halfwidth)
    sweep2_search = first_sweep2[row_ok]
    data_search = normalized[row_ok, :]   # [수정] 여기서 이상치 마스킹 안 함!

    # =====================================================
    # STEP 2 [수정]. 연속성 제약 딥 추적 (사용자 제안 코드 그대로 반영)
    # =====================================================
    dip_positions = []
    last = None
    for i in range(data_search.shape[1]):
        col = data_search[:, i]
        if last is None:
            idx = np.argmin(col)
        else:
            dist = np.abs(sweep2_search - last)
            nearby = dist < aa_max_jump
            if np.any(nearby):
                local = np.where(nearby)[0]
                idx = local[np.argmin(col[local])]
                    # [사용자 코드 미세 수정] col[nearby] -> col[local]:
                    # col[nearby]는 불리언 마스크로 인덱싱해 이미
                    # "nearby 안의 상대적 위치"가 되므로, argmin 결과를
                    # local(원래 인덱스 목록)에 다시 대입해야 원래
                    # sweep2_search 배열의 절대 인덱스가 됨.
            else:
                idx = np.argmin(col)
        last = sweep2_search[idx]
        dip_positions.append(last)
    dip_positions = np.array(dip_positions)

    dip_freq = flux_to_freq(dip_positions)   # [수정] 실제 주파수(Hz)로 변환

    # =====================================================
    # STEP 3 [수정]. 이상치 마스킹은 추출된 1D 점들에만 적용
    # =====================================================
    # 우선 초벌 직선 피팅으로 잔차를 구하고, 그 잔차 기준 상하위
    # 1.5%(양쪽 합쳐 3%)를 이상치로 제외
    coeffs_pre = np.polyfit(all_v1, dip_freq, deg=1)
    residual_pre = dip_freq - np.polyval(coeffs_pre, all_v1)
    lo_r, hi_r = np.percentile(residual_pre, [aa_outlier_pct/2, 100-aa_outlier_pct/2])
    keep = (residual_pre >= lo_r) & (residual_pre <= hi_r)
    print(f"\n1D 궤적에서 이상치 제외: {np.sum(~keep)}개 / {len(dip_freq)}개")

    x_data = all_v1[keep]
    y_data = dip_freq[keep]

    # =====================================================
    # STEP 4. 가우시안 우도로 직선 피팅
    # =====================================================
    def neg_log_lik_gaussian(params, x, y, sigma):
        slope, intercept = params
        residual = y - (slope*x + intercept)
        return 0.5 * np.sum((residual/sigma)**2)

    sigma_baseline = np.std(y_data)
    res_gauss = minimize(neg_log_lik_gaussian, [0.0, np.median(y_data)],
                           args=(x_data, y_data, sigma_baseline), method='Nelder-Mead')
    slope_gauss, intercept_gauss = res_gauss.x
    print(f"\n[가우시안 우도 피팅] 기울기={slope_gauss:.4f} Hz/V, 절편={intercept_gauss/1e9:.6f} GHz")

    # =====================================================
    # STEP 5. Robust sigma로 재비교
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
    print(f"[Robust 피팅] 기울기={slope_robust:.4f} Hz/V, 절편={intercept_robust/1e9:.6f} GHz")

    rel_diff = abs(slope_gauss-slope_robust)/max(abs(slope_gauss),1e-10)*100
    print(f"\n[비교] 기울기 상대차이: {rel_diff:.1f}% "
          f"(v1에서는 116.5%였음 - 이번엔 훨씬 작아야 수정이 성공한 것)")
    print(f"sigma_baseline={sigma_baseline:.2f}Hz vs sigma_robust={sigma_robust:.2f}Hz")
