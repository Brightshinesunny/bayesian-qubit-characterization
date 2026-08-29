"""
tls_visualize_and_tabledata2.py
=================================================================
[목적] segment 0의 실제 2D 스캔 데이터를 시각화해서 논문 Fig.2a와
같은 "TLS 공명 지도"를 재현하고, tdat.tabledata2(진짜 TLS 피팅
결과로 추정)의 13개 행이 무엇을 의미하는지 확인합니다.
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
aa_segment_index = 0
aa_output_dir = './outputs/'


def mat_string(f, dataset_or_ref):
    try:
        if isinstance(dataset_or_ref, h5py.Reference):
            dataset_or_ref = f[dataset_or_ref]
        arr = np.array(dataset_or_ref)
        return ''.join(chr(int(c)) for c in arr.flatten())
    except Exception as e:
        return f"(실패:{e})"


with h5py.File(aa_mat_filepath, 'r') as f:
    # =====================================================
    # STEP 1. segment 0의 obs.vals 10개 항목을 전부 역참조
    # =====================================================
    qdat_group = f[f['qdats']['qdat'][aa_segment_index, 0]]
    sweep1vals = np.array(qdat_group['sweep1vals']).flatten()
    sweep2vals = np.array(qdat_group['sweep2vals']).flatten()

    obs_group = qdat_group['obs']
    obs_val_refs = obs_group['vals']
    obs_name_refs = qdat_group['observables']

    obs_dict = {}
    for i in range(obs_val_refs.shape[0]):
        name = mat_string(f, obs_name_refs[i, 0])
        val_ref = obs_val_refs[i, 0]
        val_arr = np.array(f[val_ref])
        obs_dict[name] = val_arr
        print(f"obs['{name}']: shape={val_arr.shape}, dtype={val_arr.dtype}")

    # =====================================================
    # STEP 2. dispamp 또는 dispphase를 2D로 시각화
    # (sweep1=21개 x sweep2=401개 그리드와 일치하는지 확인)
    # =====================================================
    print(f"\nsweep1vals(전압) 개수: {len(sweep1vals)}, sweep2vals(주파수축) 개수: {len(sweep2vals)}")

    key_to_plot = 'dispamp' if 'dispamp' in obs_dict else list(obs_dict.keys())[0]
    data_2d = obs_dict[key_to_plot]
    print(f"\n'{key_to_plot}' 원본 shape: {data_2d.shape}")

    # shape이 (21,401) 또는 (401,21) 또는 다른 형태일 수 있으므로 확인 후 필요시 전치
    if data_2d.shape == (len(sweep1vals), len(sweep2vals)):
        data_for_plot = data_2d
    elif data_2d.shape == (len(sweep2vals), len(sweep1vals)):
        data_for_plot = data_2d.T
    else:
        print(f"⚠️ 예상한 (21,401) 또는 (401,21) 형태가 아님 - 그대로 출력만")
        data_for_plot = None

    if data_for_plot is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.pcolormesh(sweep2vals, sweep1vals, data_for_plot, shading='auto', cmap='viridis')
        ax.set_xlabel('sweep2 (플럭스/주파수 축)')
        ax.set_ylabel('sweep1 (V_alpha, 게이트 전압)')
        ax.set_title(f'Segment 0 TLS 공명 지도 ({key_to_plot})\n'
                      f'(어두운/밝은 줄무늬 = TLS 공명 - 논문 Fig.2a와 비교)')
        plt.colorbar(im, ax=ax, label=key_to_plot)
        plt.tight_layout()
        plt.savefig(f'{aa_output_dir}/tls_segment0_map.png', dpi=140, bbox_inches='tight')
        print(f"\n지도 저장 완료: {aa_output_dir}/tls_segment0_map.png")

    # =====================================================
    # STEP 3. tdat.tabledata2 구조 확인 (진짜 TLS 피팅 결과로 추정)
    # =====================================================
    print("\n" + "="*70)
    print("tdat.tabledata2 확인 (13행 x 200열)")
    print("="*70)
    tabledata2_refs = f['tdat']['tabledata2']
    for row in range(tabledata2_refs.shape[0]):
        ref = tabledata2_refs[row, 0]
        val = np.array(f[ref])
        if val.dtype == np.uint16:
            decoded = mat_string(f, f[ref])
            print(f"  행{row}: (문자열) '{decoded}'")
        else:
            print(f"  행{row}: shape={val.shape}, 값(일부)={val.flatten()[:5]}")
