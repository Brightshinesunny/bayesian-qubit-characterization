"""
tls_ascii_heatmap.py
=================================================================
[목적] matplotlib 이미지 대신, 텍스트(ASCII 문자)로 히트맵을 그려서
Claude가 이미지 없이도 직접 패턴을 읽을 수 있게 합니다. 이번엔
확인된 계통오차(sweep2≈0.0465 dispamp / 0.0020 dispphase 근처)를
제외한 뒤, 남은 데이터에서 다른 패턴(진짜 TLS 후보)이 보이는지
확인합니다.

[ASCII 히트맵 원리]
2D 데이터를 작은 격자(예: 세로 40칸, 가로 60칸)로 다운샘플링한 뒤,
값의 크기에 따라 문자를 다르게 배정합니다(공백=배경, #=가장 어두운
값 등). 텍스트로 출력되므로 이미지 업로드 없이 그대로 채팅에
복사해서 보여줄 수 있습니다.
"""

import h5py
import numpy as np

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
aa_electrodes = ['ao3', 'ao4', 'ao5', 'ao6']
aa_observable = 'dispamp'
aa_grid_rows = 30   # sweep2(주파수) 축 다운샘플 크기
aa_grid_cols = 70   # 전압 축 다운샘플 크기
aa_mask_center = 0.0465   # [확인된 계통오차 위치] 이 근처를 제외
aa_mask_halfwidth = 0.004  # 이 폭만큼 양옆을 제외


def mat_string(f, dataset_or_ref):
    try:
        if isinstance(dataset_or_ref, h5py.Reference):
            dataset_or_ref = f[dataset_or_ref]
        arr = np.array(dataset_or_ref)
        return ''.join(chr(int(c)) for c in arr.flatten())
    except Exception:
        return None


def text_heatmap(data_2d, rows, cols, chars=" .:-=+*#%@"):
    """
    2D 배열을 (rows x cols) 격자로 다운샘플링(블록 평균)한 뒤, 값 크기에
    따라 문자를 배정해 텍스트로 반환. 어두운 값(작은 값)일수록 진한
    문자(#,%,@)를, 밝은 값(큰 값)일수록 연한 문자(공백,.)를 배정.
    """
    h, w = data_2d.shape
    row_edges = np.linspace(0, h, rows+1).astype(int)
    col_edges = np.linspace(0, w, cols+1).astype(int)
    downsampled = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            block = data_2d[row_edges[i]:row_edges[i+1], col_edges[j]:col_edges[j+1]]
            downsampled[i, j] = np.nanmean(block) if block.size > 0 else np.nan

    vmin, vmax = np.nanpercentile(downsampled, [2, 98])
    normalized = np.clip((downsampled - vmin) / (vmax - vmin + 1e-12), 0, 1)

    lines = []
    for i in range(rows):
        line = ""
        for j in range(cols):
            if np.isnan(normalized[i, j]):
                line += "?"
            else:
                idx = int(normalized[i, j] * (len(chars)-1))
                line += chars[idx]
        lines.append(line)
    return lines, vmin, vmax


with h5py.File(aa_mat_filepath, 'r') as f:
    qdat_refs = f['qdats']['qdat']
    n_segments = qdat_refs.shape[0]

    for electrode in aa_electrodes:
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
                if name == aa_observable:
                    target_col = i
                    break
            data_2d = np.array(f[obs_val_refs[target_col, 0]])
            all_v1.extend(sweep1vals.tolist())
            all_data_rows.append(data_2d)

        stitched_data = np.concatenate(all_data_rows, axis=1)
        all_v1 = np.array(all_v1)

        # 열(전압)별 중앙값 제거 (배경 밝기 정규화)
        col_median = np.median(stitched_data, axis=0, keepdims=True)
        normalized = stitched_data - col_median

        # [핵심] 확인된 계통오차 위치를 NaN으로 마스킹
        mask_rows = np.abs(first_sweep2 - aa_mask_center) < aa_mask_halfwidth
        masked_data = normalized.copy()
        masked_data[mask_rows, :] = np.nan

        print(f"\n{'='*80}")
        print(f"전극: {electrode}, 관측량: {aa_observable} (계통오차 마스킹 후)")
        print(f"전압범위: [{all_v1.min():.0f}, {all_v1.max():.0f}]V, "
              f"sweep2범위: [{first_sweep2.min():.4f}, {first_sweep2.max():.4f}]")
        print(f"{'='*80}")

        lines, vmin, vmax = text_heatmap(masked_data, aa_grid_rows, aa_grid_cols)
        print(f"(진한 문자 # = 어두운값(딥) 쪽, 공백 = 밝은값 쪽, ? = 마스킹된 부분, "
              f"값범위=[{vmin:.4f},{vmax:.4f}])")
        print(f"세로축: sweep2 위쪽이 {first_sweep2.max():.3f}(위), 아래쪽이 {first_sweep2.min():.3f}")
        print(f"가로축: 전압 왼쪽이 {all_v1.min():.0f}V, 오른쪽이 {all_v1.max():.0f}V")
        for line in lines:
            print(line)
