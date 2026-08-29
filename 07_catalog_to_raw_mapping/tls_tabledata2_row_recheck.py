"""
tls_tabledata2_row_recheck.py
=================================================================
[긴급 재검증] 지난번 "행1/3/10/12=대칭점 전압"이라는 가정이, 실제로
-151V, +1240V 같은 물리적으로 불가능한 값을 냄 - 그 가정 자체가
틀렸다는 뜻. 열4의 13개 행 전부를 그대로 찍어서, 어느 행이 실제
raw 전압 범위(-60~+100V)와 맞는지 직접 재확인.

[동시에 확인] 400to640 파일의 raw sweep1vals가 정말 1파일과 같은
범위(-60~-50 근처)인지도 같이 확인 - 파일마다 전압 스캔 범위가
다를 가능성도 배제 안 됨.
"""

import h5py
import numpy as np

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments400to640_20TLSfitted.mat'
aa_target_column = 4


def mat_string(f, dataset_or_ref):
    try:
        if isinstance(dataset_or_ref, h5py.Reference):
            dataset_or_ref = f[dataset_or_ref]
        arr = np.array(dataset_or_ref)
        return ''.join(chr(int(c)) for c in arr.flatten())
    except Exception:
        return None


with h5py.File(aa_mat_filepath, 'r') as f:
    # =====================================================
    # STEP 1. tabledata2 열4의 13개 행 전부, 가공 없이 그대로 출력
    # =====================================================
    tabledata2_refs = f['tdat']['tabledata2']
    n_rows = tabledata2_refs.shape[0]
    print(f"[STEP 1] tabledata2 열{aa_target_column}의 13개 행 원본값")
    for row in range(n_rows):
        ref = tabledata2_refs[row, aa_target_column]
        val = np.array(f[ref])
        if val.dtype == np.uint16:
            decoded = mat_string(f, val)
            print(f"  행{row}: (문자열) '{decoded}'")
        else:
            v = val.flatten()[0] if val.size > 0 else None
            print(f"  행{row}: {v}")

    # =====================================================
    # STEP 2. 이 파일 raw의 ao4 segment들 sweep1vals 범위 확인
    # =====================================================
    print(f"\n[STEP 2] 이 파일(400to640) raw의 ao4 segment 전압범위 (처음 10개)")
    qdat_refs = f['qdats']['qdat']
    n_segments = qdat_refs.shape[0]
    count = 0
    for i in range(n_segments):
        qdat_group = f[qdat_refs[i, 0]]
        name = mat_string(f, qdat_group['sweep1name'])
        if name == 'ao4':
            sweep1vals = np.array(qdat_group['sweep1vals']).flatten()
            print(f"  raw_index={i}: 전압범위=[{sweep1vals.min():.2f},{sweep1vals.max():.2f}]")
            count += 1
            if count >= 10:
                break
