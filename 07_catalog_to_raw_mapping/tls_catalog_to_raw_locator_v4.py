"""
tls_catalog_to_raw_locator_v4.py
=================================================================
[역매핑 v4] v3 실행 결과 - ao4(주전극, 결합세기 701MHz/V로 가장
큼)의 목표주파수(5.164GHz) 포함 segment 160개 중, 국소품질 상위
10개를 전부 확인한 결과:
    raw=84  (V=-39~-38): 국소4.98, 전체6.79, 목표=전체최소? 아니오
    raw=503 (V=66~67)  : 국소4.33, 전체6.88, 목표=전체최소? 아니오
    raw=483 (V=61~62)  : 국소4.02, 전체5.08, 목표=전체최소? 아니오
    (이하 상위 10개 전부 "아니오")
즉 ao4에서는 "목표=전체최소=예"(진짜 A케이스)가 상위 10개 안에
하나도 없었음 - 전부 B케이스(다른 배경 띠가 진짜 최저점이고, 목표
주파수 5.164GHz는 그 사이에 낀 평범한 지점일 뿐).

[결론 - 이게 왜 "실패"가 아니라 "검증이 제 역할을 한 것"인가]
v2에서는 "전체품질"(맵 전체 최솟값)만 보고 raw_index=503을 골랐는데,
그건 5.164GHz와 무관한 다른 배경 띠(아마 5.09GHz 근처)가 만든
가짜 1등이었음. v3의 "국소품질 vs 전체품질 비교"가 이 가짜 1등을
정확히 걸러냄 - 즉 "raw_index=503이 열4의 원본이다"라고 잘못
결론짓는 걸 미리 막은 것.

[v4가 하는 일 - 다음 단계]
ao4 하나에서 A케이스가 안 나왔다고 포기하지 않고, 나머지 3개
전극(ao5=beta 439MHz/V, ao3=gamma 138MHz/V, ao6=delta 22.5MHz/V)도
같은 방식(국소품질 vs 전체품질 비교)으로 전부 확인:
  1. ao4, ao5, ao3, ao6 순서로 전부 순회
  2. 각 전극에서 "목표=전체최소=예"(진짜 A케이스)만 골라 저장
  3. 4개 전극 중 A케이스가 하나라도 나오면 -> 그게 진짜 후보로 채택
  4. 4개 전극 전부 A케이스가 없으면 -> "이 카탈로그 항목(열4)은
     이 파일의 raw 원본 맵에서 뚜렷하게 안 보인다"는 결론을
     명시적으로 확정. 이유는 두 가지로 추정 가능:
       (a) 원저자의 자체 프로그램("quin")이 우리보다 훨씬 정교한
           알고리즘(예: 여러 segment를 이어붙여서 보는 방식,
           시간축까지 활용하는 방식)으로 찾아냈을 가능성
       (b) 결합세기가 피팅상으로는 크게(701MHz/V) 나왔어도, 실제
           원시 맵에서는 노이즈에 묻혀 육안/SNR로는 확인이 안 될
           정도로 약한 신호였을 가능성

[핵심 교훈]
"결합세기(피팅값)가 가장 큰 전극이 곧 원시 맵에서 가장 잘 보이는
전극은 아닐 수 있다" - 피팅된 파라미터와 원시 데이터의 가시성은
서로 다른 정보라서, 하나만 보고 포기하면 안 되고 체계적으로 전부
확인해야 함.
"""

import h5py
import numpy as np

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments400to640_20TLSfitted.mat'
aa_target_column = 4
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
    """특정 전극의 모든 segment를 훑어서 (raw_index, 국소품질, A케이스여부) 반환."""
    qdat_refs = f['qdats']['qdat']
    n_segments = qdat_refs.shape[0]
    results = []
    for i in range(n_segments):
        qdat_group = f[qdat_refs[i, 0]]
        name = mat_string(f, qdat_group['sweep1name'])
        if name != electrode_code:
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

        results.append({'raw_index': i, 'v_range': (sweep1vals.min(), sweep1vals.max()),
                          'local_quality': -local_min/(bg_std+1e-12), 'is_A_case': is_A_case})
    return results


with h5py.File(aa_mat_filepath, 'r') as f:
    tabledata2_refs = f['tdat']['tabledata2']
    gamma_vals = {}
    for row in aa_gamma_rows:
        gamma_vals[row] = np.array(f[tabledata2_refs[row, aa_target_column]]).flatten()[0]
    target_freq_hz = np.array(f[tabledata2_refs[aa_freq_row, aa_target_column]]).flatten()[0]
    target_freq_ghz = target_freq_hz / 1e9

    calib_field = np.array(f['qset']['calibdat']['field']).flatten()
    calib_freq_hz = np.array(f['qset']['calibdat']['qubitfreq']).flatten()
    order = np.argsort(calib_field)

    print(f"[목표] 열{aa_target_column}, 주파수={target_freq_ghz:.4f}GHz\n")
    print("전극별 결합세기(참고용):")
    for row, name in aa_electrode_names.items():
        print(f"  {name}: {gamma_vals[row]/1e6:+.1f} MHz/V")

    # 4개 전극 전부 체계적으로 확인
    all_A_cases = []
    for row, electrode_code in aa_electrode_names.items():
        print(f"\n[{electrode_code} 탐색 - 결합세기 {gamma_vals[row]/1e6:+.1f}MHz/V]")
        results = search_electrode(f, electrode_code, target_freq_ghz, calib_field, calib_freq_hz, order)
        results.sort(key=lambda r: -r['local_quality'])

        a_cases = [r for r in results if r['is_A_case']]
        print(f"  전체 {len(results)}개 중 A케이스(진짜 목표주파수가 최저점): {len(a_cases)}개")
        for r in a_cases[:5]:
            vr = r['v_range']
            print(f"    raw_index={r['raw_index']}, 전압범위=[{vr[0]:.1f},{vr[1]:.1f}], "
                  f"국소품질={r['local_quality']:.2f}")
            all_A_cases.append({'electrode': electrode_code, **r})

    print("\n" + "="*60)
    if all_A_cases:
        best = max(all_A_cases, key=lambda r: r['local_quality'])
        print(f"[최종] 전극 무관 최선의 A케이스: {best['electrode']}, "
              f"raw_index={best['raw_index']}, 국소품질={best['local_quality']:.2f}")
    else:
        print("[최종] 4개 전극 어디에서도 A케이스(진짜 목표주파수 딥) 없음")
        print("       -> 이 카탈로그 항목(열4)은 raw 원본 맵에서 뚜렷하게 안 보임")
        print("          (원저자의 자체 알고리즘이 우리보다 더 정교하게 걸러냈을 가능성,")
        print("           또는 여러 segment를 이어붙인 형태로만 보이는 신호일 가능성)")
