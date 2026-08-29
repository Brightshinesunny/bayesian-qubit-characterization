"""
tls_seg12_v3_freq_domain.py
=================================================================
[v3 수정 - 주파수 영역에서 직접 탐색]
  v2의 문제: 연속성 제약(max_jump)을 sweep2(플럭스, 원시 단위)로
  걸었는데, 플럭스->주파수 변환이 비선형(칼리브레이션 곡선이
  구간마다 기울기가 다름)이라, "플럭스 단위로 가까운 점"이 실제
  주파수로는 전혀 안 가까울 수 있습니다. 그 결과 추적이 대각선
  띠가 아니라, 우연히 항상 낮은 값을 가지는 특정 고정 주파수
  (약 5.167GHz)의 평평한 줄을 따라갔습니다.

  수정: 1) 탐색 범위 자체를 주파수로 직접 제한(5.12~5.20GHz)
        2) 연속성 제약도 주파수 단위(10MHz)로 다시 정의
"""

import h5py
import numpy as np
from scipy.optimize import minimize

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
aa_target_raw_index = 12
aa_observable = 'dispamp'
aa_freq_window = (5.12e9, 5.20e9)   # [수정] 육안 확인된 대각선 범위로 직접 제한
aa_max_jump_freq = 10e6   # [수정] 연속성 제약을 주파수 단위(10MHz)로 정의


def mat_string(f, dataset_or_ref):
    try:
        if isinstance(dataset_or_ref, h5py.Reference):
            dataset_or_ref = f[dataset_or_ref]
        arr = np.array(dataset_or_ref)
        return ''.join(chr(int(c)) for c in arr.flatten())
    except Exception:
        return None


def extract_trace_freq_domain(f, qdat_group, observable, sweep2_full,
                                 calib_field, calib_freq, freq_window, max_jump_freq,
                                 known_artifact_centers, artifact_halfwidth):
    """[수정된 버전] 주파수 영역에서 직접 탐색 + 연속성 제약."""
    order = np.argsort(calib_field)
    freq_full = np.interp(sweep2_full, calib_field[order], calib_freq[order])

    # 계통오차(플럭스 기준) 제외 + 목표 주파수 창 제한을 동시에 적용
    row_ok = np.ones(len(sweep2_full), dtype=bool)
    for center in known_artifact_centers:
        row_ok &= (np.abs(sweep2_full - center) > artifact_halfwidth)
    row_ok &= (freq_full >= freq_window[0]) & (freq_full <= freq_window[1])

    freq_search = freq_full[row_ok]

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
    data_search = normalized[row_ok, :]

    if data_search.shape[0] == 0:
        return sweep1vals, None, None, 0

    dip_freq_positions, last = [], None
    for i in range(data_search.shape[1]):
        col = data_search[:, i]
        if last is None:
            idx = np.argmin(col)
        else:
            dist = np.abs(freq_search - last)   # [수정] 주파수 단위 거리
            nearby = dist < max_jump_freq
            if np.any(nearby):
                local = np.where(nearby)[0]
                idx = local[np.argmin(col[local])]
            else:
                idx = np.argmin(col)
        last = freq_search[idx]
        dip_freq_positions.append(last)
    dip_freq_positions = np.array(dip_freq_positions)

    depths = []
    for i in range(data_search.shape[1]):
        idx_at_dip = np.argmin(np.abs(freq_search - dip_freq_positions[i]))
        depths.append(-data_search[idx_at_dip, i])
    quality = np.mean(depths) / (np.std(data_search) + 1e-12)

    return sweep1vals, dip_freq_positions, quality, data_search.shape[0]


with h5py.File(aa_mat_filepath, 'r') as f:
    qdat_refs = f['qdats']['qdat']
    calib_field = np.array(f['qset']['calibdat']['field']).flatten()
    calib_freq = np.array(f['qset']['calibdat']['qubitfreq']).flatten()
    known_artifact_centers = [0.045, -0.030, -0.065]
    artifact_halfwidth = 0.006

    print(f"[STEP 1] raw_index={aa_target_raw_index}, 주파수 영역 직접 탐색 "
          f"({aa_freq_window[0]/1e9:.2f}~{aa_freq_window[1]/1e9:.2f}GHz)")
    target_group = f[qdat_refs[aa_target_raw_index, 0]]
    target_electrode = mat_string(f, target_group['sweep1name'])
    sweep2_full = np.array(target_group['sweep2vals']).flatten()

    v1, dip_freq, quality, n_rows_used = extract_trace_freq_domain(
        f, target_group, aa_observable, sweep2_full, calib_field, calib_freq,
        aa_freq_window, aa_max_jump_freq, known_artifact_centers, artifact_halfwidth
    )
    print(f"  전극={target_electrode}, 탐색 가능한 행 수={n_rows_used}, 품질점수={quality:.3f}")

    if dip_freq is None:
        print("  ⚠️ 이 주파수 창에서 사용 가능한 데이터가 없음")
    else:
        def neg_log_lik_gaussian(params, x, y, sigma):
            slope, intercept = params
            return 0.5 * np.sum(((y-(slope*x+intercept))/sigma)**2)

        def neg_log_lik_robust(params, x, y, sigma, nu=4.0):
            slope, intercept = params
            return 0.5*(nu+1) * np.sum(np.log1p(((y-(slope*x+intercept))/sigma)**2/nu))

        sigma_g = np.std(dip_freq)
        res_g = minimize(neg_log_lik_gaussian, [0, np.median(dip_freq)],
                           args=(v1, dip_freq, sigma_g), method='Nelder-Mead')
        slope_g, intercept_g = res_g.x

        residual_g = dip_freq - (slope_g*v1 + intercept_g)
        mad = np.median(np.abs(residual_g - np.median(residual_g)))
        sigma_r = mad * 1.4826 if mad > 0 else sigma_g
        res_r = minimize(neg_log_lik_robust, res_g.x, args=(v1, dip_freq, sigma_r), method='Nelder-Mead')
        slope_r, intercept_r = res_r.x

        rel_diff = abs(slope_g-slope_r)/max(abs(slope_g),1e-10)*100
        print(f"\n  전압-딥주파수 21개 점:")
        for vv, ff in zip(v1, dip_freq):
            print(f"    V={vv:.3f}, freq={ff/1e9:.5f}GHz")
        print(f"\n  [Gaussian] 기울기={slope_g/1e6:.2f} MHz/V")
        print(f"  [Robust]   기울기={slope_r/1e6:.2f} MHz/V")
        print(f"  기울기 상대차이: {rel_diff:.1f}%")
        print(f"  (육안 추정 기울기: 약 87 MHz/V와 비교)")
