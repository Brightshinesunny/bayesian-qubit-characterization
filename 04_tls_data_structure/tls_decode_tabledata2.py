"""
tls_decode_tabledata2.py
=================================================================
[목적] 지금까지 raw 픽셀에서 직접 확인한 물리적 스케일 감각
(진짜 결합세기 gamma ~ 10^6~10^8 Hz(MHz~수백MHz), 전압 ~ -60~160V,
큐빗 주파수 ~ 5x10^9 Hz)을 활용해서, tdat.tabledata2(13행x200열,
원저자가 이미 20개 TLS를 피팅해둔 결과로 추정)의 각 행이 무엇을
의미하는지 통계적으로 역추적합니다.

[방법]
1개 열(segment 0)만 보는 대신, 여러 열(예: 30개)을 한꺼번에 보고
각 행의 값 범위(최소/최대/평균)를 통계 내면, "이 행은 전압
스케일이다/주파수 스케일이다/결합세기 스케일이다"를 훨씬 명확하게
구분할 수 있습니다.
"""

import h5py
import numpy as np

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
aa_n_columns_to_check = 30


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
    n_rows, n_cols_total = tabledata2_refs.shape
    print(f"tabledata2 전체 크기: {n_rows}행 x {n_cols_total}열\n")

    # 각 행(row)마다, 여러 열에 걸친 값들을 모아서 통계 확인
    row_values = {r: [] for r in range(n_rows)}
    row_is_string = {r: False for r in range(n_rows)}

    for col in range(min(aa_n_columns_to_check, n_cols_total)):
        for row in range(n_rows):
            ref = tabledata2_refs[row, col]
            val = np.array(f[ref])
            if val.dtype == np.uint16:
                row_is_string[row] = True
                continue
            if val.size > 0:
                row_values[row].append(float(val.flatten()[0]))

    print(f"{'행':>4} {'타입':>8} {'개수':>6} {'최소':>16} {'최대':>16} {'평균':>16} {'0인 비율':>10}")
    print("-"*90)
    for row in range(n_rows):
        if row_is_string[row]:
            print(f"{row:>4} {'문자열':>8}")
            continue
        vals = np.array(row_values[row])
        if len(vals) == 0:
            print(f"{row:>4} {'(빈값)':>8}")
            continue
        zero_frac = np.mean(np.abs(vals) < 1e-10)
        print(f"{row:>4} {'숫자':>8} {len(vals):>6} {vals.min():>16.4g} {vals.max():>16.4g} "
              f"{vals.mean():>16.4g} {zero_frac*100:>9.1f}%")

    print("\n[스케일 기반 해석 가이드]")
    print("  ~10^6~10^8 (수백만~수억) -> 결합세기(gamma, Hz/V 단위)일 가능성")
    print("  ~1~200 -> 전압(V) 스케일일 가능성")
    print("  ~10^9(수십억) -> 절대 주파수(Hz) 스케일일 가능성")
    print("  0이 아주 많은 행(50%+) -> 그 항목이 대부분의 TLS에서 결합이")
    print("    없거나(감지 안 됨) 미사용 필드일 가능성")

    # 실제 alpha에서 확인된 gamma_alpha=109MHz/V=1.09e8Hz/V와 가장
    # 비슷한 스케일을 가진 행이 어느 것인지 자동 판별
    print("\n[alpha에서 확인된 gamma_alpha≈1.09e8 Hz/V와 스케일이 가장 비슷한 행]")
    target_scale = 1.09e8
    best_row, best_ratio = None, np.inf
    for row in range(n_rows):
        if row_is_string[row] or len(row_values[row]) == 0:
            continue
        vals = np.abs(np.array(row_values[row]))
        nonzero_vals = vals[vals > 1e-10]
        if len(nonzero_vals) == 0:
            continue
        median_val = np.median(nonzero_vals)
        ratio = max(median_val, target_scale) / min(median_val, target_scale)
        if ratio < best_ratio:
            best_ratio = ratio
            best_row = row
    if best_row is not None:
        print(f"  행{best_row} (0이 아닌 값들의 중앙값이 gamma_alpha와 가장 가까운 자릿수)")
