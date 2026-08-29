"""
tls_cross_validate_electrodes.py
=================================================================
[목적] 방금 alpha(ao4) 전극 + dispamp 관측량에서 "딥 위치가 거의
안 움직인다(계통오차 의심)"는 결론이 나왔습니다. 이게 정말
계통오차(예: 구동선 공진, 케이블 정재파)라면, 물리적으로 완전히
무관한 다른 전극(beta,gamma,delta)이나 다른 관측량(dispphase)에서도
"똑같은 sweep2 위치"에 딥이 나타나야 합니다 - 전극이나 관측 방식과
무관하게 항상 같은 곳에 있다는 건, TLS(전극별로 다르게 반응해야
정상)가 아니라 "이 주파수(sweep2) 자체에 내재된 하드웨어 문제"라는
뜻이기 때문입니다.

반대로, 전극마다 딥 위치가 서로 다르다면 - 그건 각 전극이 국소
전기장으로 서로 다른 TLS(또는 같은 TLS라도 다른 결합세기)를
건드린다는 뜻이라 실제 TLS 신호일 가능성이 높아집니다.
"""

import h5py
import numpy as np

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
aa_electrodes_to_check = ['ao3', 'ao4', 'ao5', 'ao6']   # gamma, alpha, beta, delta
aa_observables_to_check = ['dispamp', 'dispphase']


def mat_string(f, dataset_or_ref):
    try:
        if isinstance(dataset_or_ref, h5py.Reference):
            dataset_or_ref = f[dataset_or_ref]
        arr = np.array(dataset_or_ref)
        return ''.join(chr(int(c)) for c in arr.flatten())
    except Exception:
        return None


def get_dip_stats(f, qdat_refs, electrode_name, observable_name):
    """특정 전극+관측량 조합으로 딥 위치를 추적하고 통계를 반환."""
    n_segments = qdat_refs.shape[0]
    matching_segments = []
    for i in range(n_segments):
        qdat_group = f[qdat_refs[i, 0]]
        name = mat_string(f, qdat_group['sweep1name'])
        if name == electrode_name:
            sweep1vals = np.array(qdat_group['sweep1vals']).flatten()
            matching_segments.append({'index': i, 'v_min': sweep1vals.min()})
    if len(matching_segments) == 0:
        return None
    matching_segments.sort(key=lambda s: s['v_min'])

    first_sweep2 = np.array(f[qdat_refs[matching_segments[0]['index'], 0]]['sweep2vals']).flatten()

    all_v1 = []
    all_data_rows = []
    for s in matching_segments:
        qdat_group = f[qdat_refs[s['index'], 0]]
        sweep1vals = np.array(qdat_group['sweep1vals']).flatten()
        obs_val_refs = qdat_group['obs']['vals']
        obs_name_refs = qdat_group['observables']
        target_col = None
        for i in range(obs_name_refs.shape[0]):
            name = mat_string(f, obs_name_refs[i, 0])
            if name == observable_name:
                target_col = i
                break
        if target_col is None:
            return None
        data_2d = np.array(f[obs_val_refs[target_col, 0]])
        all_v1.extend(sweep1vals.tolist())
        all_data_rows.append(data_2d)

    stitched_data = np.concatenate(all_data_rows, axis=1)
    all_v1 = np.array(all_v1)

    col_median = np.median(stitched_data, axis=0, keepdims=True)
    normalized = stitched_data - col_median

    dip_positions = np.array([first_sweep2[np.argmin(normalized[:, i])]
                                for i in range(stitched_data.shape[1])])

    return {
        'n_segments': len(matching_segments),
        'v_range': (all_v1.min(), all_v1.max()),
        'dip_median': np.median(dip_positions),
        'dip_std': dip_positions.std(),
        'correlation': np.corrcoef(all_v1, dip_positions)[0, 1],
    }


with h5py.File(aa_mat_filepath, 'r') as f:
    qdat_refs = f['qdats']['qdat']

    print(f"{'전극':>6} {'관측량':>10} {'segment수':>10} {'전압범위':>16} "
          f"{'딥위치 중앙값':>14} {'딥위치 std':>12} {'상관계수':>10}")
    print("-"*80)

    results = {}
    for electrode in aa_electrodes_to_check:
        for obs in aa_observables_to_check:
            stats = get_dip_stats(f, qdat_refs, electrode, obs)
            if stats is None:
                print(f"{electrode:>6} {obs:>10}  (데이터 없음 또는 실패)")
                continue
            results[(electrode, obs)] = stats
            vr = stats['v_range']
            print(f"{electrode:>6} {obs:>10} {stats['n_segments']:>10} "
                  f"[{vr[0]:>5.0f},{vr[1]:>5.0f}] {stats['dip_median']:>14.4f} "
                  f"{stats['dip_std']:>12.4f} {stats['correlation']:>10.4f}")

    print("\n[해석 가이드]")
    print("  모든 전극/관측량 조합에서 '딥위치 중앙값'이 거의 같은 값(예: -0.065 근처)에")
    print("  몰려있다면, 이건 특정 TLS가 아니라 sweep2(주파수) 축 자체의 계통적 특징")
    print("  (예: 구동선 공진)일 가능성이 매우 높습니다.")
    print("  반대로 전극마다 딥위치가 서로 다르다면, 각 전극이 서로 다른 실제 신호를")
    print("  건드리고 있다는 뜻이라 TLS일 가능성이 높아집니다.")
