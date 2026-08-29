"""
tls_catalog_to_raw_locator_v3.py
=================================================================
[역매핑 v3 - v2의 결함 수정] v2의 문제: 이 파일은 "같은 주파수
창(5.035~5.257GHz)을 전압 조각만 바꿔가며 반복"하는 구조라서,
"주파수가 창 안에 있는가"라는 조건이 사실상 필터 역할을 전혀
못 함(ao4 160개 전부 통과). 그 상태에서 품질점수를 "맵 전체
최솟값"으로 계산해버려서, 진짜 5.164GHz 근처가 아니라 "가장 시끄러운
배경"을 고른 것과 같아짐(가짜 1등 문제, B/C 케이스).

[수정 핵심]
품질점수를 "맵 전체"가 아니라 "목표 주파수(5.164GHz) 근처 좁은
띠 안"에서만 계산. 이러면 진짜로 "그 주파수에 어두운 줄이 있는
조각"만 높은 점수를 받게 됨.

[검증 방법도 명시적으로 추가]
1등, 2등 후보(raw_index=503, 84)의 dispamp 맵에서, 목표 주파수
지점의 값과 그 주파수를 "제외한" 나머지 배경 값을 비교 - 그
주파수에서 유독 어두운지, 아니면 다른 곳(예: 5.09GHz)이 더
어두운데 우연히 최댓값에 걸린 건지 구분.
"""

import h5py
import numpy as np

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments400to640_20TLSfitted.mat'
aa_target_column = 4
aa_gamma_rows = [0, 2, 9, 11]
aa_electrode_names = {0: 'ao4', 2: 'ao5', 9: 'ao3', 11: 'ao6'}
aa_freq_row = 4
aa_freq_local_halfwidth = 0.01
    # [핵심 수정] 목표 주파수 근처 ±10MHz 안에서만 품질점수를 계산.
    # 이 폭 밖(배경 전체)은 "이 주파수에 어두운 줄이 있는가"와
    # 무관하므로 아예 안 봄.


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
    gamma_vals = {}
    for row in aa_gamma_rows:
        gamma_vals[row] = np.array(f[tabledata2_refs[row, aa_target_column]]).flatten()[0]
    target_freq_hz = np.array(f[tabledata2_refs[aa_freq_row, aa_target_column]]).flatten()[0]
    target_freq_ghz = target_freq_hz / 1e9
    dominant_row = max(gamma_vals, key=lambda r: abs(gamma_vals[r]))
    dominant_electrode = aa_electrode_names[dominant_row]

    print(f"[목표] 주파수={target_freq_ghz:.4f}GHz, 주전극={dominant_electrode}\n")

    calib_field = np.array(f['qset']['calibdat']['field']).flatten()
    calib_freq_hz = np.array(f['qset']['calibdat']['qubitfreq']).flatten()
    order = np.argsort(calib_field)

    qdat_refs = f['qdats']['qdat']
    n_segments = qdat_refs.shape[0]

    # =====================================================
    # [핵심 수정] "목표 주파수 근처에서만" 품질점수 계산
    # =====================================================
    all_candidates = []
    for i in range(n_segments):
        qdat_group = f[qdat_refs[i, 0]]
        name = mat_string(f, qdat_group['sweep1name'])
        if name != dominant_electrode:
            continue
        sweep1vals = np.array(qdat_group['sweep1vals']).flatten()
        sweep2vals = np.array(qdat_group['sweep2vals']).flatten()
        fq_axis = np.interp(sweep2vals, calib_field[order], calib_freq_hz[order]) / 1e9

        # 목표 주파수 근처 좁은 띠만 골라냄 (배경 전체가 아니라)
        local_mask = np.abs(fq_axis - target_freq_ghz) < aa_freq_local_halfwidth
        if np.sum(local_mask) < 3:
            continue   # 이 세그먼트가 목표 주파수를 아예 포함 안 하면 건너뜀

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

        # [핵심] 품질점수를 "목표 주파수 근처"만으로 계산
        local_min = normalized[local_mask, :].min()
        background_std = np.std(normalized)   # 배경 표준편차는 전체로 계산(정규화 기준)
        local_quality = -local_min / (background_std + 1e-12)

        # [검증용] 목표 주파수 근처가 아닌 다른 곳(전체 맵)의 최솟값도
        # 같이 기록 - "다른 곳이 더 어두운데 우연히 걸린 건 아닌지" 대조
        global_min = normalized.min()
        global_quality = -global_min / (background_std + 1e-12)

        all_candidates.append({
            'raw_index': i, 'v_range': (sweep1vals.min(), sweep1vals.max()),
            'local_quality': local_quality, 'global_quality': global_quality,
            'is_local_the_global_min': abs(local_min - global_min) < 1e-9,
                # True면 "목표 주파수 근처가 실제로 이 맵에서 가장
                # 어두운 곳"이라는 뜻(B/C 케이스가 아니라 A케이스일
                # 가능성이 높음). False면 다른 곳이 더 어두운데,
                # 목표 주파수 근처는 상대적으로 평범하다는 뜻(B케이스 의심).
        })

    all_candidates.sort(key=lambda c: -c['local_quality'])
    print(f"목표 주파수를 포함하는 {dominant_electrode} segment: {len(all_candidates)}개\n")
    print(f"{'raw':>5} {'전압범위':>14} {'국소품질(목표주파수)':>16} {'전체품질(배경최댓값)':>16} {'목표=전체최소?':>12}")
    print("-"*75)
    for c in all_candidates[:10]:
        vr = c['v_range']
        print(f"{c['raw_index']:>5} [{vr[0]:>5.1f},{vr[1]:>5.1f}] "
              f"{c['local_quality']:>16.2f} {c['global_quality']:>16.2f} "
              f"{'예(A케이스 유력)' if c['is_local_the_global_min'] else '아니오(B케이스 의심)':>12}")

    print("\n[해석 가이드]")
    print("  국소품질이 높고 '목표=전체최소'가 '예'면: 목표 주파수 자체가")
    print("    이 맵에서 가장 어두운 지점 - 진짜 후보(A케이스)일 가능성 높음")
    print("  국소품질은 낮은데 전체품질만 높으면: 목표 주파수와 무관한")
    print("    다른 배경 띠(예:5.09GHz)가 진짜 딥 - 이 후보는 가짜(B케이스)")
