"""
tls_stitch_alpha_segments.py
=================================================================
[목적] sweep1name='ao4'(alpha 전극)인 모든 segment를 찾아서,
V_alpha 값 순서대로 정렬한 뒤 이어붙여, segment 0 하나(1V 폭)에서
"수직선"으로 보였던 패턴이 더 넓은 전압 범위에서 실제로 곡선
(TLS 공명 패턴)으로 휘는지 확인합니다.

[왜 이렇게 하는가 - 물리적 근거]
TLS 공명 주파수 공식(omega_TLS = sqrt(Delta^2+(epsilon+2p.E)^2)/hbar)은
쌍곡선 형태입니다. 좁은 전압 구간(1V)만 보면 어떤 매끄러운 곡선이든
거의 직선처럼 보이는 게 정상입니다(국소적 선형근사). 훨씬 넓은
범위를 이어붙여야 진짜 곡선 모양(또는 avoided-crossing의 특징적인
쌍곡선 휘어짐)을 확인할 수 있습니다.

[여러 segment를 이어붙일 때 sweep2vals(주파수 축)가 다를 수 있다는
점에 유의]
각 segment의 sweep2vals(플럭스/주파수 축)이 매번 정확히 같은 401개
값이라는 보장은 없습니다(연속 segment마다 스캔 범위가 미세하게
다를 수 있음) - 이어붙이기 전에 이것도 확인합니다.
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
aa_target_electrode = 'ao4'   # [조정] alpha 전극. 다른 전극(ao3=gamma, ao5=beta, ao6=delta)도 시도 가능.
aa_observable_to_plot = 'dispamp'
aa_output_dir = './outputs/'


def mat_string(f, dataset_or_ref):
    try:
        if isinstance(dataset_or_ref, h5py.Reference):
            dataset_or_ref = f[dataset_or_ref]
        arr = np.array(dataset_or_ref)
        return ''.join(chr(int(c)) for c in arr.flatten())
    except Exception:
        return None


with h5py.File(aa_mat_filepath, 'r') as f:
    qdat_refs = f['qdats']['qdat']
    n_segments = qdat_refs.shape[0]
    print(f"전체 segment 개수: {n_segments}")

    # =====================================================
    # STEP 1. sweep1name이 목표 전극인 segment들을 전부 찾기
    # =====================================================
    matching_segments = []
    for i in range(n_segments):
        qdat_group = f[qdat_refs[i, 0]]
        name = mat_string(f, qdat_group['sweep1name'])
        if name == aa_target_electrode:
            sweep1vals = np.array(qdat_group['sweep1vals']).flatten()
            matching_segments.append({
                'index': i,
                'v_min': sweep1vals.min(),
                'v_max': sweep1vals.max(),
                'n_points': len(sweep1vals),
            })

    print(f"\n'{aa_target_electrode}' 전극을 스캔한 segment 개수: {len(matching_segments)}")
    if len(matching_segments) > 0:
        v_ranges = [(s['v_min'], s['v_max']) for s in matching_segments]
        print(f"각 segment의 전압 범위(처음 10개): {v_ranges[:10]}")

    # 전압 시작값(v_min) 기준으로 정렬
    matching_segments.sort(key=lambda s: s['v_min'])
    overall_v_min = matching_segments[0]['v_min']
    overall_v_max = matching_segments[-1]['v_max']
    print(f"\n전체 이어붙인 전압 범위: [{overall_v_min:.2f}, {overall_v_max:.2f}] "
          f"({overall_v_max-overall_v_min:.2f}V 폭, segment 1개(1V)의 "
          f"{(overall_v_max-overall_v_min):.0f}배)")

    # =====================================================
    # STEP 2. sweep2vals가 segment마다 동일한지 확인
    # =====================================================
    first_sweep2 = np.array(f[qdat_refs[matching_segments[0]['index'], 0]]['sweep2vals']).flatten()
    all_same_sweep2 = True
    for s in matching_segments[1:5]:   # 처음 5개만 빠르게 확인
        s2 = np.array(f[qdat_refs[s['index'], 0]]['sweep2vals']).flatten()
        if not np.allclose(s2, first_sweep2):
            all_same_sweep2 = False
            break
    print(f"\nsweep2vals(주파수축)이 segment마다 동일한가? {all_same_sweep2}")

    # =====================================================
    # STEP 3. 전체 segment를 이어붙여 큰 2D 배열 생성
    # =====================================================
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
            if name == aa_observable_to_plot:
                target_col = i
                break
        if target_col is None:
            continue

        data_2d = np.array(f[obs_val_refs[target_col, 0]])
            # shape=(401, 21) 형태로 확인됨 - (sweep2, sweep1) 순서

        all_v1.extend(sweep1vals.tolist())
        all_data_rows.append(data_2d)   # (401, 21) 블록들을 모음

    # 가로(전압축)로 이어붙임: 모든 블록이 (401, n_i) 형태이므로 axis=1로 concat
    stitched_data = np.concatenate(all_data_rows, axis=1)   # (401, 전체전압포인트수)
    all_v1 = np.array(all_v1)

    print(f"\n이어붙인 데이터 shape: {stitched_data.shape}")
    print(f"이어붙인 전압축 포인트 수: {len(all_v1)}")

    # =====================================================
    # STEP 4. 시각화
    # =====================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.pcolormesh(all_v1, first_sweep2, stitched_data, shading='auto', cmap='viridis')
    ax.set_xlabel(f'V_{aa_target_electrode} (게이트 전압, V) - 여러 segment 이어붙임')
    ax.set_ylabel('sweep2 (플럭스/주파수 축)')
    ax.set_title(f'{aa_target_electrode} 전극 전체 범위 TLS 공명 지도 ({aa_observable_to_plot})\n'
                  f'수직선이 곡선으로 휘는지 확인 - 진짜 TLS vs 계통오차 판별')
    plt.colorbar(im, ax=ax, label=aa_observable_to_plot)
    plt.tight_layout()
    plt.savefig(f'{aa_output_dir}/tls_stitched_alpha_map.png', dpi=140, bbox_inches='tight')
    print(f"\n이어붙인 지도 저장 완료: {aa_output_dir}/tls_stitched_alpha_map.png")
