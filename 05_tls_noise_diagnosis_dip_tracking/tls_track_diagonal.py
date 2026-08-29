"""
tls_track_diagonal.py
=================================================================
[목적] 히트맵에서 육안으로 확인된 "대각선 줄무늬"(전압에 따라
sweep2 위치가 서서히 이동하는 패턴)를 정량적으로 추적합니다. 이미
확인된 수평 계통오차(sweep2≈0.045,-0.03,-0.065 근처)는 제외하고,
그 "사이 구간"(배경 영역)에서만 국소적인 극값을 추적합니다.

[왜 "사이 구간"만 보는가]
수평 계통오차가 워낙 강해서, 그냥 전체 범위에서 argmin을 하면
매번 그 강한 계통오차만 잡힙니다(지난번 겪은 문제). 대각선 신호는
그보다 훨씬 약하므로, 계통오차 띠들을 먼저 제외한 "배경 구간"
안에서만 극값을 찾아야 대각선을 제대로 추적할 수 있습니다.
"""

import h5py
import numpy as np

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
aa_electrodes = ['ao3', 'ao4', 'ao5', 'ao6']
aa_observable = 'dispamp'
aa_known_artifact_centers = [0.045, -0.030, -0.065]   # 확인된 계통오차 위치들
aa_artifact_halfwidth = 0.006   # 이 폭만큼 각 계통오차 주변을 제외
aa_search_window = (-0.015, 0.020)   # [조정] 대각선이 보이는 것으로 추정되는
aa_smoothing_window = 15
    # [신규] 인접한 전압 열을 이 개수만큼 평균 내서 잡음을 줄임. 개별
    # 픽셀 단위로는 대각선(미세한 신호)이 잡음에 묻혀 안 보였을 수
    # 있으므로, 스무딩으로 신호대잡음비를 높이는 표준적인 방법.
    # sweep2≈0 부근의 "배경 구간"만 좁혀서 탐색 (그림에서 중앙 부근)


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

    all_v1, all_data_rows = [], []
    for s in matching_segments:
        qdat_group = f[qdat_refs[s['index'], 0]]
        sweep1vals = np.array(qdat_group['sweep1vals']).flatten()
        obs_val_refs = qdat_group['obs']['vals']
        obs_name_refs = qdat_group['observables']
        target_col = None
        for i in range(obs_name_refs.shape[0]):
            if mat_string(f, obs_name_refs[i, 0]) == observable:
                target_col = i
                break
        data_2d = np.array(f[obs_val_refs[target_col, 0]])
        all_v1.extend(sweep1vals.tolist())
        all_data_rows.append(data_2d)

    stitched = np.concatenate(all_data_rows, axis=1)
    col_median = np.median(stitched, axis=0, keepdims=True)
    return np.array(all_v1), first_sweep2, stitched - col_median


with h5py.File(aa_mat_filepath, 'r') as f:
    qdat_refs = f['qdats']['qdat']

    print(f"{'전극':>6} {'기울기(sweep2/V)':>18} {'상관계수':>10} {'대각선 여부':>12}")
    print("-"*55)

    results = {}
    for electrode in aa_electrodes:
        all_v1, sweep2, normalized = get_stitched_data(f, qdat_refs, electrode, aa_observable)

        # STEP 1: 알려진 계통오차 위치를 제외한 "탐색 가능한 행" 마스크 생성
        row_ok = (sweep2 >= aa_search_window[0]) & (sweep2 <= aa_search_window[1])
        for center in aa_known_artifact_centers:
            row_ok &= (np.abs(sweep2 - center) > aa_artifact_halfwidth)

        sweep2_search = sweep2[row_ok]
        data_search = normalized[row_ok, :]

        # [신규] 스무딩: 인접한 전압 열끼리 이동평균을 취해 잡음을 줄임
        kernel = np.ones(aa_smoothing_window) / aa_smoothing_window
        data_smoothed = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode='same'), axis=1, arr=data_search
        )
            # np.apply_along_axis: 2D 배열의 각 "행"(여기서는 sweep2 한
            # 줄, 즉 전압 축 방향)에 대해 1차원 함수(convolve)를 적용.
            # 이렇게 하면 전압 방향으로 이웃한 aa_smoothing_window개
            # 열끼리 평균내는 효과가 남 - 개별 픽셀 잡음은 상쇄되고,
            # 여러 열에 걸쳐 일관되게 나타나는 미세한 추세(대각선)만
            # 살아남게 됨.

        # STEP 2: 스무딩된 데이터에서 각 전압(열)마다 최솟값 위치 추적
        diag_positions = np.array([sweep2_search[np.argmin(data_smoothed[:, i])]
                                     for i in range(data_smoothed.shape[1])])

        # STEP 3: 1차 다항식(직선) 피팅으로 기울기 추정
        coeffs = np.polyfit(all_v1, diag_positions, deg=1)
        slope, intercept = coeffs
        correlation = np.corrcoef(all_v1, diag_positions)[0, 1]

        is_diagonal = abs(correlation) > 0.3   # [조정] 대각선으로 판정할 상관계수 기준

        results[electrode] = {'slope': slope, 'correlation': correlation, 'positions': diag_positions}
        print(f"{electrode:>6} {slope:>18.8f} {correlation:>10.4f} "
              f"{'예 (대각선 감지)' if is_diagonal else '아니오':>12}")

    print("\n[해석 가이드]")
    print("  기울기가 0에 가깝고 상관계수도 낮으면: 대각선이 아니라 그냥 배경 잡음")
    print("  기울기가 뚜렷하고(0이 아님) 상관계수도 높으면: 진짜 전압-의존적 신호(TLS 후보)")
    print("  전극마다 기울기 크기가 다르면: 각 전극이 서로 다른 세기로 TLS와 결합한다는")
    print("  논문의 핵심 주장과 일치 - 위치 추정에 사용할 수 있는 유력한 증거")
