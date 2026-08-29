"""
tls_multi_column_scan.py
=================================================================
[여러 열 동시 확인 - 방법 자체 검증] 열4 하나만 봐서는 "이게 열4의
특수한 문제인지, 역매핑 방법 자체의 문제인지" 구분이 안 됨. 이
파일(400to640)의 20개 카탈로그 항목 전부에 같은 검증(v4의
A케이스 탐색)을 돌려서:
  - 몇 개 항목이 raw에서 "진짜 A케이스"로 확인되는지
  - 확인되는 항목과 안 되는 항목 사이에 패턴이 있는지
    (예: 결합세기가 특히 크거나 작은 항목만 실패하는지)
을 통계적으로 확인. 이러면 "역매핑 방법 자체가 원래 이 정도 실패율을
가지는 게 정상"인지, 아니면 "뭔가 체계적으로 잘못됐다"인지 구분됨.
"""

import h5py
import numpy as np

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments400to640_20TLSfitted.mat'
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


def search_electrode(f, electrode_code, target_freq_ghz, calib_field, calib_freq_hz, order):
    qdat_refs = f['qdats']['qdat']
    n_segments = qdat_refs.shape[0]
    results = []
    for i in range(n_segments):
        qdat_group = f[qdat_refs[i, 0]]
        name = mat_string(f, qdat_group['sweep1name'])
        if name != electrode_code:
            continue
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

        results.append({'raw_index': i, 'local_quality': -local_min/(bg_std+1e-12),
                          'is_A_case': is_A_case})
    return results


with h5py.File(aa_mat_filepath, 'r') as f:
    tabledata2_refs = f['tdat']['tabledata2']
    n_rows, n_cols = tabledata2_refs.shape

    calib_field = np.array(f['qset']['calibdat']['field']).flatten()
    calib_freq_hz = np.array(f['qset']['calibdat']['qubitfreq']).flatten()
    order = np.argsort(calib_field)

    # 채워진 열(진짜 TLS 항목) 전부 찾기
    filled_columns = []
    for col in range(n_cols):
        gamma_vals = {}
        skip = False
        for row in aa_gamma_rows:
            ref = tabledata2_refs[row, col]
            val = np.array(f[ref])
            if val.dtype == np.uint16:
                skip = True
                break
            gamma_vals[row] = float(val.flatten()[0]) if val.size > 0 else 0.0
        if skip:
            continue
        if max(abs(v) for v in gamma_vals.values()) >= 1e6:
            freq_val = np.array(f[tabledata2_refs[aa_freq_row, col]]).flatten()[0]
            filled_columns.append({'col': col, 'gammas': gamma_vals, 'freq_ghz': freq_val/1e9})

    print(f"채워진 TLS 항목: {len(filled_columns)}개\n")
    print(f"{'열':>4} {'주파수(GHz)':>12} {'최대결합세기':>14} {'A케이스 개수':>12} {'A있는 전극':>20}")
    print("-"*70)

    summary = []
    for entry in filled_columns:
        col = entry['col']
        target_freq_ghz = entry['freq_ghz']
        max_gamma = max(abs(v) for v in entry['gammas'].values())

        found_electrodes = []
        total_A = 0
        for row, electrode_code in aa_electrode_names.items():
            results = search_electrode(f, electrode_code, target_freq_ghz, calib_field, calib_freq_hz, order)
            a_count = sum(1 for r in results if r['is_A_case'])
            if a_count > 0:
                found_electrodes.append(electrode_code)
            total_A += a_count

        print(f"{col:>4} {target_freq_ghz:>12.4f} {max_gamma/1e6:>13.1f}M "
              f"{total_A:>12} {','.join(found_electrodes) if found_electrodes else '없음':>20}")
        summary.append({'col': col, 'total_A': total_A, 'found': len(found_electrodes) > 0})

    n_found = sum(1 for s in summary if s['found'])
    print(f"\n[요약] {len(summary)}개 항목 중 {n_found}개는 raw에서 A케이스 확인됨, "
          f"{len(summary)-n_found}개는 확인 안 됨")
    print(f"확인 비율: {n_found/len(summary)*100:.1f}%")
