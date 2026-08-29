"""
tls_tabledata2_full_listing.py
=================================================================
[목적] "하나만 지배적으로 크다"는 좁은 필터를 버리고, tabledata2에서
값이 채워진(0이 아닌) 열 전부를 나열해서, 실제 20개 TLS의 결합
패턴이 전반적으로 어떤 모습인지 확인합니다.
"""

import h5py
import numpy as np

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
aa_gamma_rows = [0, 2, 9, 11]
aa_voltage_rows = {0: 1, 2: 3, 9: 10, 11: 12}
aa_freq_row = 4
aa_min_signal_threshold = 1e6   # 1MHz/V 이상이면 "이 전극에 결합이 있다"로 인정


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
    n_rows, n_cols = tabledata2_refs.shape

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

        max_abs = max(abs(v) for v in gamma_vals.values())
        if max_abs >= aa_min_signal_threshold:
            freq_val = np.array(f[tabledata2_refs[aa_freq_row, col]]).flatten()[0]
            filled_columns.append({'col': col, 'gammas': gamma_vals, 'freq': freq_val})

    print(f"채워진(0이 아닌) 열: {len(filled_columns)}개 / {n_cols}개 전체\n")
    print(f"{'열':>5} {'주파수(GHz)':>12} {'γ(행0)':>10} {'γ(행2)':>10} {'γ(행9)':>10} {'γ(행11)':>10}")
    print("-"*70)
    for c in filled_columns:
        g = c['gammas']
        print(f"{c['col']:>5} {c['freq']/1e9:>12.4f} "
              f"{g[0]/1e6:>10.1f} {g[2]/1e6:>10.1f} {g[9]/1e6:>10.1f} {g[11]/1e6:>10.1f}")

    print(f"\n[요약 통계 - 채워진 {len(filled_columns)}개 TLS 대상]")
    for row in aa_gamma_rows:
        vals = np.array([abs(c['gammas'][row]) for c in filled_columns])
        n_significant = np.sum(vals >= aa_min_signal_threshold)
        print(f"  행{row}: {n_significant}/{len(filled_columns)}개 TLS에서 결합 있음(≥1MHz/V), "
              f"평균 크기={vals[vals>=aa_min_signal_threshold].mean()/1e6 if n_significant>0 else 0:.1f}MHz/V")

    freqs = np.array([c['freq'] for c in filled_columns])
    print(f"\n  TLS 주파수 범위: [{freqs.min()/1e9:.4f}, {freqs.max()/1e9:.4f}] GHz")
    print(f"  (오늘 찾은 5.12~5.17GHz 대역과 겹치는 TLS가 있는지 확인용)")
