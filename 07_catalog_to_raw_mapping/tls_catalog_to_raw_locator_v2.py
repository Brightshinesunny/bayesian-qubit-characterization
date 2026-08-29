"""
tls_catalog_to_raw_locator_v2.py
=================================================================
[역매핑 v2 - 전압 대신 주파수+주전극으로] v1은 행1/3/10/12를
"대칭점 전압"으로 가정했다가 실패(-151V 등 물리적으로 불가능한 값).
이번엔 확실히 검증된 것만 씀:
  - 행4 = TLS 주파수(Hz) - 어제 seg12 딥 주파수 범위와 스케일 일치
  - 결합세기 절댓값이 가장 큰 전극 = 주 전극

[전략]
1. 목표 TLS의 주파수(행4)와 주 전극을 확인
2. 그 전극의 모든 raw segment를 훑으면서, 각 segment의 큐빗주파수
   범위(calibdat로 변환) 안에 목표 주파수가 포함되는지 확인
3. 포함되는 후보들 중, 품질점수(SNR)가 가장 높은 곳을 채택

[중요한 한계 인정] 이 파일(400to640)에는 여러 라운드(같은 전극,
다른 전압구간)가 있고, 큐빗주파수 자체도 라운드마다 조금씩 다를 수
있어서, 후보가 여러 개 나올 가능성이 있음 - 그럴 땐 품질점수로
걸러냄.
"""

import h5py
import numpy as np
import sys, os
sys.path.insert(0, os.getcwd())
import validation_robots as vr

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments400to640_20TLSfitted.mat'
aa_target_column = 4
aa_gamma_rows = [0, 2, 9, 11]
aa_electrode_names = {0: 'ao4', 2: 'ao5', 9: 'ao3', 11: 'ao6'}
aa_freq_row = 4
aa_freq_match_tolerance = 0.005   # GHz, 목표 주파수 근처 이 폭 안이면 "포함"으로 판정


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
    # STEP 1. 목표 TLS의 주파수와 주 전극 확인
    # =====================================================
    gamma_vals = {}
    for row in aa_gamma_rows:
        gamma_vals[row] = np.array(f[tabledata2_refs[row, aa_target_column]]).flatten()[0]
    target_freq_hz = np.array(f[tabledata2_refs[aa_freq_row, aa_target_column]]).flatten()[0]
    target_freq_ghz = target_freq_hz / 1e9

    dominant_row = max(gamma_vals, key=lambda r: abs(gamma_vals[r]))
    dominant_electrode = aa_electrode_names[dominant_row]

    print(f"[STEP 1] 목표: 주파수={target_freq_ghz:.4f}GHz, 주전극={dominant_electrode} "
          f"(gamma={gamma_vals[dominant_row]/1e6:.1f}MHz/V)")

    # =====================================================
    # STEP 2. 그 전극의 모든 raw segment를 훑어서, 목표 주파수를
    # 포함하는 것 찾기
    # =====================================================
    calib_field = np.array(f['qset']['calibdat']['field']).flatten()
    calib_freq_hz = np.array(f['qset']['calibdat']['qubitfreq']).flatten()
    order = np.argsort(calib_field)

    qdat_refs = f['qdats']['qdat']
    n_segments = qdat_refs.shape[0]

    candidates = []
    for i in range(n_segments):
        qdat_group = f[qdat_refs[i, 0]]
        name = mat_string(f, qdat_group['sweep1name'])
        if name != dominant_electrode:
            continue
        sweep2vals = np.array(qdat_group['sweep2vals']).flatten()
        fq_axis = np.interp(sweep2vals, calib_field[order], calib_freq_hz[order]) / 1e9
        if fq_axis.min() - aa_freq_match_tolerance <= target_freq_ghz <= fq_axis.max() + aa_freq_match_tolerance:
            sweep1vals = np.array(qdat_group['sweep1vals']).flatten()
            candidates.append({'raw_index': i, 'v_range': (sweep1vals.min(), sweep1vals.max()),
                                 'fq_range': (fq_axis.min(), fq_axis.max())})

    print(f"\n[STEP 2] 목표 주파수({target_freq_ghz:.4f}GHz)를 포함하는 {dominant_electrode} segment: "
          f"{len(candidates)}개")
    for c in candidates:
        print(f"  raw_index={c['raw_index']}, 전압범위={c['v_range']}, "
              f"주파수범위=[{c['fq_range'][0]:.4f},{c['fq_range'][1]:.4f}]")

    # =====================================================
    # STEP 3. 품질점수로 최종 후보 선정
    # =====================================================
    if len(candidates) == 0:
        print("\n[결과] 후보 없음 - target_freq_ghz 자체가 이 파일의 calibdat 범위 밖일 수 있음.")
        print("  calibdat 전체 범위 재확인 필요.")
    else:
        best_candidate, best_quality = None, -np.inf
        for c in candidates:
            qdat_group = f[qdat_refs[c['raw_index'], 0]]
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
            quality = -normalized.min() / (np.std(normalized) + 1e-12)
            print(f"  raw_index={c['raw_index']}: 품질점수={quality:.2f}")
            if quality > best_quality:
                best_quality = quality
                best_candidate = c['raw_index']

        print(f"\n[최종] 가장 유력한 raw_index = {best_candidate} (품질점수={best_quality:.2f})")
