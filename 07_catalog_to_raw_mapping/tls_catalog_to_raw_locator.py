"""
tls_catalog_to_raw_locator.py
=================================================================
[2번 - 카탈로그 -> raw segment 역매핑] tabledata2의 특정 TLS 항목이
어느 raw segment(들)에서 나왔는지 찾는다. tabledata2의 200개 열이
639개 raw segment와 직접 인덱스로 안 맞아서(구조가 다름), "대칭점
전압"이라는 물리적 단서로 역추적한다.

[전략]
1. 카탈로그 항목에서 결합세기가 가장 큰 전극(예: 700MHz/V짜리)을 확인
2. 그 전극에 해당하는 대칭점 전압(행1/3/10/12 중 짝)을 확인
3. 그 전극, 그 전압 범위를 포함하는 raw segment를 전부 찾음(보통
   1V 폭짜리 segment 여러 개가 후보가 됨)
4. 각 후보에서 품질점수(SNR)를 계산해, 가장 뚜렷한 신호가 있는
   곳을 최종 후보로 채택
"""

import h5py
import numpy as np
import sys, os
sys.path.insert(0, os.getcwd())
import validation_robots as vr

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments400to640_20TLSfitted.mat'
aa_target_column = 4   # [조정] 아까 확인한 gamma=701MHz/V짜리 강한 TLS(열4)
aa_gamma_rows = [0, 2, 9, 11]
aa_voltage_rows = {0: 1, 2: 3, 9: 10, 11: 12}   # 결합세기 행 <-> 대칭점 전압 행 짝
aa_electrode_names = {0: 'ao4(alpha)', 2: 'ao5(beta)', 9: 'ao3(gamma)', 11: 'ao6(delta)'}
    # [오늘 확인된 매핑] raw_index%4로 결정되는 순서: 0=ao4,1=ao5,2=ao3,3=ao6


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
    # STEP 1. 목표 TLS 항목의 결합세기/대칭점 전압 확인
    # =====================================================
    print(f"[STEP 1] 열{aa_target_column}의 4개 전극 결합세기 및 대칭점 전압")
    gamma_vals, voltage_vals = {}, {}
    for row in aa_gamma_rows:
        gamma_vals[row] = np.array(f[tabledata2_refs[row, aa_target_column]]).flatten()[0]
        v_row = aa_voltage_rows[row]
        voltage_vals[row] = np.array(f[tabledata2_refs[v_row, aa_target_column]]).flatten()[0]
        print(f"  {aa_electrode_names[row]:>14}: gamma={gamma_vals[row]/1e6:>8.1f}MHz/V, "
              f"대칭점전압={voltage_vals[row]:>8.2f}V")

    # 결합세기가 가장 큰 전극을 "주 전극"으로 채택
    dominant_row = max(gamma_vals, key=lambda r: abs(gamma_vals[r]))
    dominant_electrode = aa_electrode_names[dominant_row]
    dominant_voltage = voltage_vals[dominant_row]
    print(f"\n  주 전극: {dominant_electrode}, 대칭점 전압: {dominant_voltage:.2f}V")

    # =====================================================
    # STEP 2. 그 전극, 그 전압 근처의 raw segment들을 찾음
    # =====================================================
    print(f"\n[STEP 2] {dominant_electrode}, V≈{dominant_voltage:.2f} 근처 raw segment 탐색")
    qdat_refs = f['qdats']['qdat']
    n_segments = qdat_refs.shape[0]

    target_electrode_code = dominant_electrode.split('(')[0]   # 'ao4' 부분만 추출
    candidates = []
    for i in range(n_segments):
        qdat_group = f[qdat_refs[i, 0]]
        name = mat_string(f, qdat_group['sweep1name'])
        if name != target_electrode_code:
            continue
        sweep1vals = np.array(qdat_group['sweep1vals']).flatten()
        if sweep1vals.min() - 0.5 <= dominant_voltage <= sweep1vals.max() + 0.5:
            candidates.append({'raw_index': i, 'v_range': (sweep1vals.min(), sweep1vals.max())})

    print(f"  후보 segment {len(candidates)}개 발견:")
    for c in candidates:
        print(f"    raw_index={c['raw_index']}, 전압범위={c['v_range']}")

    # =====================================================
    # STEP 3. 각 후보의 품질점수(SNR) 계산해서 최선의 후보 선정
    # =====================================================
    print(f"\n[STEP 3] 각 후보의 품질점수(SNR) 비교")
    calib_field = np.array(f['qset']['calibdat']['field']).flatten()
    calib_freq_hz = np.array(f['qset']['calibdat']['qubitfreq']).flatten()
    order = np.argsort(calib_field)

    best_candidate, best_quality = None, -np.inf
    for c in candidates:
        qdat_group = f[qdat_refs[c['raw_index'], 0]]
        sweep2vals = np.array(qdat_group['sweep2vals']).flatten()
        obs_val_refs = qdat_group['obs']['vals']
        obs_name_refs = qdat_group['observables']
        target_col = None
        for i in range(obs_name_refs.shape[0]):
            if mat_string(f, obs_name_refs[i, 0]) == 'dispamp':
                target_col = i
                break
        data_2d = np.array(f[obs_val_refs[target_col, 0]])
        col_median = np.median(data_2d, axis=0, keepdims=True)
        normalized = data_2d - col_median

        # 전체 배경 대비, 가장 어두운 지점의 깊이로 품질점수 계산
        quality = -normalized.min() / (np.std(normalized) + 1e-12)
        print(f"    raw_index={c['raw_index']}: 품질점수={quality:.2f}")
        if quality > best_quality:
            best_quality = quality
            best_candidate = c['raw_index']

    print(f"\n[최종] 가장 유력한 raw_index = {best_candidate} (품질점수={best_quality:.2f})")
    print("이 raw_index로 오늘 만든 tls_avcross_pipeline.py의 함수들을 그대로 적용해서")
    print("육안 검증 + MCMC까지 진행하면 됨(seg12 때와 같은 절차).")
