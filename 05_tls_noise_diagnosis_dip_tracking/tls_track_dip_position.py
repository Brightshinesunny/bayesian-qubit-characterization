"""
tls_track_dip_position.py
=================================================================
[목적] 시각적으로 그래프를 판독하는 대신, "딥(공명 지점)의 위치가
전압에 따라 실제로 이동하는지"를 코드로 직접, 정량적으로 확인합니다.

[방법]
이어붙인 데이터(401 x 3360)에서, 각 전압(열) 위치마다 sweep2(주파수)
축을 따라 신호가 가장 어두운(또는 가장 밝은) 지점을 찾습니다. 이
"극값의 위치"를 전압 전체 범위(160V)에 대해 그려보면:
  - 위치가 거의 안 변함(수평선) -> 전압과 무관한 계통오차 가능성
  - 위치가 매끄럽게 이동함(곡선) -> 진짜 TLS가 전압에 반응하는 것
    (Stark 편이 - 전기장에 의해 TLS 공명 주파수가 이동하는 현상)
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
aa_target_electrode = 'ao4'
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

    matching_segments = []
    for i in range(n_segments):
        qdat_group = f[qdat_refs[i, 0]]
        name = mat_string(f, qdat_group['sweep1name'])
        if name == aa_target_electrode:
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
            if name == aa_observable_to_plot:
                target_col = i
                break
        data_2d = np.array(f[obs_val_refs[target_col, 0]])
        all_v1.extend(sweep1vals.tolist())
        all_data_rows.append(data_2d)

    stitched_data = np.concatenate(all_data_rows, axis=1)
    all_v1 = np.array(all_v1)

    # =====================================================
    # STEP: 각 전압(열)마다, sweep2 축에서 신호가 가장 낮은(어두운)
    # 지점의 위치를 찾음 - "딥 위치 추적"
    # =====================================================
    # 전역 배경 편차(전압에 따라 전체 밝기가 변하는 효과)를 먼저
    # 제거하기 위해, 각 열(전압)에서 중앙값을 빼서 정규화
    col_median = np.median(stitched_data, axis=0, keepdims=True)
    normalized = stitched_data - col_median

    dip_positions = []
    for col_idx in range(stitched_data.shape[1]):
        col = normalized[:, col_idx]
        min_idx = np.argmin(col)   # 가장 어두운(최소값) 지점의 인덱스
        dip_positions.append(first_sweep2[min_idx])
    dip_positions = np.array(dip_positions)

    print(f"전압 범위: [{all_v1.min():.1f}, {all_v1.max():.1f}]V")
    print(f"딥 위치(sweep2) 범위: [{dip_positions.min():.4f}, {dip_positions.max():.4f}]")
    print(f"딥 위치의 표준편차: {dip_positions.std():.4f}")
    print(f"(딥 위치가 거의 고정이면 std가 작고, 전압에 따라 크게 움직이면 std가 큼)")

    # 상관계수: 전압과 딥 위치 사이에 선형(또는 단조) 관계가 있는지
    correlation = np.corrcoef(all_v1, dip_positions)[0, 1]
    print(f"\n전압 vs 딥위치 상관계수: {correlation:.4f}")
    print(f"(0에 가까우면 무관, ±1에 가까우면 강한 선형관계)")

    # 시각화: 전압에 따른 딥 위치의 이동 궤적
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    im = axes[0].pcolormesh(all_v1, first_sweep2, stitched_data, shading='auto', cmap='viridis')
    axes[0].plot(all_v1, dip_positions, 'r.', ms=2, alpha=0.5, label='추적된 딥 위치')
    axes[0].set_ylabel('sweep2 (플럭스/주파수 축)')
    axes[0].legend()
    axes[0].set_title(f'{aa_target_electrode} 전극 전체범위 - 딥 위치 추적 (빨간 점)')

    axes[1].plot(all_v1, dip_positions, '.', ms=3)
    axes[1].set_xlabel(f'V_{aa_target_electrode} (V)')
    axes[1].set_ylabel('추적된 딥 위치')
    axes[1].set_title(f'딥 위치 vs 전압 (상관계수={correlation:.4f})')

    plt.tight_layout()
    plt.savefig(f'{aa_output_dir}/tls_dip_tracking.png', dpi=140, bbox_inches='tight')
    print(f"\n그래프 저장 완료: {aa_output_dir}/tls_dip_tracking.png")

    # =====================================================
    # [추가] 선형 상관계수가 0이어도 U자형(쌍곡선) 곡선일 수 있으므로,
    # 2차 다항식(포물선)을 피팅해서 곡률과 최저점을 확인
    # =====================================================
    print("\n" + "="*60)
    print("[추가 검증] 2차 다항식 피팅 - U자형(쌍곡선) 패턴 확인")
    print("="*60)

    coeffs = np.polyfit(all_v1, dip_positions, deg=2)
    a_coef, b_coef, c_coef = coeffs
    print(f"2차 다항식: y = {a_coef:.8f}*x^2 + {b_coef:.6f}*x + {c_coef:.4f}")
    print(f"곡률(2차항 계수) a = {a_coef:.8f}")

    if abs(a_coef) > 1e-8:
        vertex_v = -b_coef / (2*a_coef)   # 포물선의 꼭짓점(최저점 또는 최고점) 위치
        print(f"\n포물선 꼭짓점(대칭점 후보) 위치: V = {vertex_v:.2f}")
        if all_v1.min() <= vertex_v <= all_v1.max():
            print(f"  -> 이 꼭짓점이 스캔 범위 [{all_v1.min():.0f}, {all_v1.max():.0f}]V 안에 있음!")
            print(f"     이건 진짜 TLS의 '대칭점'(symmetry point)일 가능성을 시사합니다.")
        else:
            print(f"  -> 꼭짓점이 스캔 범위 밖에 있음 - 이 구간 안에서는 단조 변화로 보임")

    # 잔차 확인: 2차 다항식이 얼마나 잘 맞는지(R^2)
    fitted = np.polyval(coeffs, all_v1)
    ss_res = np.sum((dip_positions - fitted)**2)
    ss_tot = np.sum((dip_positions - np.mean(dip_positions))**2)
    r_squared = 1 - ss_res/ss_tot
    print(f"\n2차 다항식의 결정계수(R^2): {r_squared:.4f}")
    print(f"(1에 가까울수록 U자형 모델이 딥 위치 변화를 잘 설명한다는 뜻)")
