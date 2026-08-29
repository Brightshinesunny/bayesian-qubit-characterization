"""
tls_column17_verified_start.py
=================================================================
[열17 - 오늘 처음 검증된 raw<->카탈로그 매칭] 다중열 스캔에서
열17(5.097GHz)이 이 파일 raw의 지배적 딥(A케이스 597개, 거의
160세그먼트x4전극 전체)임이 확인됨. 이번엔 그 확인된 raw
segment(들)을 실제로 특정해서, seg12 때처럼 육안검증+피팅까지
진행할 준비를 함.
"""

import h5py
import numpy as np

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments400to640_20TLSfitted.mat'
aa_target_column = 17
aa_gamma_rows = [0, 2, 9, 11]
aa_electrode_names = {0: 'ao4', 2: 'ao5', 9: 'ao3', 11: 'ao6'}
aa_freq_row = 4
aa_freq_local_halfwidth = 0.01


def mat_string(f, dataset_or_ref):
    try:
        if isinstance(dataset_or_ref, h5py.Reference):
            dataset_or_ref = f[dataset_or_ref]
        arr = np.array(dataset_or_ref)
        return ''.join(chr(int(c)) for c in arr.flatten())
    except Exception:
        return None


with h5py.File(aa_mat_filepath, 'r') as f:
    tabledata2_refs = f['tdat']['tabledata2']

    # =====================================================
    # STEP 1. 열17의 4개 전극 결합세기 확인
    # =====================================================
    gamma_vals = {}
    for row in aa_gamma_rows:
        gamma_vals[row] = np.array(f[tabledata2_refs[row, aa_target_column]]).flatten()[0]
    target_freq_hz = np.array(f[tabledata2_refs[aa_freq_row, aa_target_column]]).flatten()[0]
    target_freq_ghz = target_freq_hz / 1e9

    print(f"[STEP 1] 열{aa_target_column}: 주파수={target_freq_ghz:.4f}GHz")
    for row, name in aa_electrode_names.items():
        print(f"  {name}: {gamma_vals[row]/1e6:+.1f} MHz/V")
    dominant_row = max(gamma_vals, key=lambda r: abs(gamma_vals[r]))
    print(f"  주 전극: {aa_electrode_names[dominant_row]}\n")

    # =====================================================
    # STEP 2. 주 전극에서, A케이스 중 국소품질이 가장 높은 상위 segment 선정
    # =====================================================
    calib_field = np.array(f['qset']['calibdat']['field']).flatten()
    calib_freq_hz = np.array(f['qset']['calibdat']['qubitfreq']).flatten()
    order = np.argsort(calib_field)

    qdat_refs = f['qdats']['qdat']
    n_segments = qdat_refs.shape[0]
    dominant_electrode = aa_electrode_names[dominant_row]

    results = []
    for i in range(n_segments):
        qdat_group = f[qdat_refs[i, 0]]
        name = mat_string(f, qdat_group['sweep1name'])
        if name != dominant_electrode:
            continue
        sweep1vals = np.array(qdat_group['sweep1vals']).flatten()
        sweep2vals = np.array(qdat_group['sweep2vals']).flatten()
        fq_axis = np.interp(sweep2vals, calib_field[order], calib_freq_hz[order]) / 1e9
        local_mask = np.abs(fq_axis - target_freq_ghz) < aa_freq_local_halfwidth
        if np.sum(local_mask) < 3:
            continue

        obs_val_refs = qdat_group['obs']['vals']
        obs_name_refs = qdat_group['observables']
        target_col = None
        for j in range(obs_name_refs.shape[0]):
            if mat_string(f, obs_name_refs[j, 0]) == 'dispamp':
                target_col = j
                break
        data_2d = np.array(f[obs_val_refs[target_col, 0]])
        col_median = np.median(data_2d, axis=0, keepdims=True)
        normalized = data_2d - col_median

        local_min = normalized[local_mask, :].min()
        global_min = normalized.min()
        bg_std = np.std(normalized)
        is_A_case = abs(local_min - global_min) < 1e-9

        if is_A_case:
            results.append({'raw_index': i, 'v_range': (sweep1vals.min(), sweep1vals.max()),
                              'local_quality': -local_min/(bg_std+1e-12)})

    results.sort(key=lambda r: -r['local_quality'])
    print(f"[STEP 2] {dominant_electrode}에서 A케이스: {len(results)}개")
    print("상위 10개:")
    for r in results[:10]:
        vr = r['v_range']
        print(f"  raw_index={r['raw_index']}, 전압범위=[{vr[0]:.1f},{vr[1]:.1f}], "
              f"국소품질={r['local_quality']:.2f}")

    if results:
        best = results[0]
        print(f"\n[선정] raw_index={best['raw_index']} - 다음 단계(육안검증+피팅)에서 이걸 사용")
