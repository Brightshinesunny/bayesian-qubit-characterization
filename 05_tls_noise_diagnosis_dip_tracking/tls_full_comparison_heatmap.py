"""
tls_full_comparison_heatmap.py
=================================================================
[목적] 4개 전극(alpha,beta,gamma,delta)의 전체 데이터를 하나의
그림 안에 같은 컬러스케일로 나란히 배치해서, ASCII 히트맵으로
확인한 "전극과 무관하게 똑같은 패턴"이 실제로도 그런지 육안으로
최종 확인합니다.
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
aa_electrodes = ['ao3', 'ao4', 'ao5', 'ao6']
aa_electrode_labels = {'ao3': 'gamma(ao3)', 'ao4': 'alpha(ao4)', 'ao5': 'beta(ao5)', 'ao6': 'delta(ao6)'}
aa_observable = 'dispamp'
aa_output_dir = './outputs/'


def mat_string(f, dataset_or_ref):
    try:
        if isinstance(dataset_or_ref, h5py.Reference):
            dataset_or_ref = f[dataset_or_ref]
        arr = np.array(dataset_or_ref)
        return ''.join(chr(int(c)) for c in arr.flatten())
    except Exception:
        return None


def get_stitched_data(f, qdat_refs, electrode, observable):
    n_segments = qdat_refs.shape[0]
    matching_segments = []
    for i in range(n_segments):
        qdat_group = f[qdat_refs[i, 0]]
        name = mat_string(f, qdat_group['sweep1name'])
        if name == electrode:
            sweep1vals = np.array(qdat_group['sweep1vals']).flatten()
            matching_segments.append({'index': i, 'v_min': sweep1vals.min()})
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
            if name == observable:
                target_col = i
                break
        data_2d = np.array(f[obs_val_refs[target_col, 0]])
        all_v1.extend(sweep1vals.tolist())
        all_data_rows.append(data_2d)

    stitched_data = np.concatenate(all_data_rows, axis=1)
    all_v1 = np.array(all_v1)

    # 열별 중앙값 제거(배경 정규화) - 전극 간 절대밝기 차이를 없애고
    # 순수한 "국소 구조"만 비교하기 위함
    col_median = np.median(stitched_data, axis=0, keepdims=True)
    normalized = stitched_data - col_median

    return all_v1, first_sweep2, normalized


with h5py.File(aa_mat_filepath, 'r') as f:
    qdat_refs = f['qdats']['qdat']

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True, sharey=True)

    all_normalized = []
    all_v1_list = []
    sweep2_ref = None
    for electrode in aa_electrodes:
        v1, sweep2, normalized = get_stitched_data(f, qdat_refs, electrode, aa_observable)
        all_normalized.append(normalized)
        all_v1_list.append(v1)
        sweep2_ref = sweep2

    # 4개 전극이 모두 같은 컬러스케일을 쓰도록 전체 범위에서 vmin/vmax 계산
    vmin = np.percentile(np.concatenate([n.flatten() for n in all_normalized]), 2)
    vmax = np.percentile(np.concatenate([n.flatten() for n in all_normalized]), 98)

    for ax, electrode, v1, normalized in zip(axes, aa_electrodes, all_v1_list, all_normalized):
        im = ax.pcolormesh(v1, sweep2_ref, normalized, shading='auto',
                             cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_ylabel('sweep2\n(주파수축)')
        ax.set_title(f'{aa_electrode_labels[electrode]} - {aa_observable} (배경제거됨, 공통 컬러스케일)')

    axes[-1].set_xlabel('게이트 전압 (V)')
    plt.colorbar(im, ax=axes, label=aa_observable, fraction=0.02)
    plt.savefig(f'{aa_output_dir}/tls_full_comparison.png', dpi=130, bbox_inches='tight')
    print(f"저장 완료: {aa_output_dir}/tls_full_comparison.png")
