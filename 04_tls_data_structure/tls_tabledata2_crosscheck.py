"""
tls_tabledata2_crosscheck.py
=================================================================
[목적] tabledata2의 200개 열 전체를 스캔해서, "행{0,2,9,11} 중 하나만
109MHz/V 근처로 크고 나머지 3개는 작은" 열을 찾습니다. 그런 열이
있으면 (1) 어느 행이 alpha 전극인지 확정되고, (2) 그 열의 다른 정보
(대칭점 전압, TLS 주파수)가 오늘 수동으로 찾은 결과(V=-57~-56,
f=5.12~5.17GHz)와 일치하는지 최종 교차검증합니다.
"""

import h5py
import numpy as np

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
aa_target_gamma = 1.093e8   # 오늘 확인된 gamma_alpha (Gaussian/Robust 평균, 109.34/109.46 MHz/V)
aa_gamma_rows = [0, 2, 9, 11]
aa_voltage_rows = {0: 1, 2: 3, 9: 10, 11: 12}   # 각 결합세기 행과 짝을 이루는 전압 행
aa_freq_row = 4
aa_dominance_ratio_threshold = 5.0
    # [조정] "하나만 크고 나머지는 작다"의 기준: 최댓값이 두번째로
    # 큰 값보다 이 배수 이상 커야 "확실히 한 전극만 강하다"고 판정.
aa_min_signal_threshold = 1e6
    # [버그 수정용 신규] 최댓값 자체가 이 값(1MHz/V) 이상이어야 "진짜
    # 신호"로 인정. 이게 없으면, 4개 값이 전부 0인 빈 TLS 항목(200개
    # 열 중 실제로 채워진 건 20개뿐일 가능성이 높음 - "20TLSfitted"
    # 파일명과 일치)도 "0 하나만 있으니 지배적"이라고 잘못 판정되어
    # 결과가 전부 0인 항목들로 뒤덮이는 문제가 있었음.


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
    n_rows, n_cols = tabledata2_refs.shape

    candidates = []
    for col in range(n_cols):
        gamma_vals = {}
        for row in aa_gamma_rows:
            ref = tabledata2_refs[row, col]
            val = np.array(f[ref])
            if val.dtype == np.uint16:
                gamma_vals = None
                break
            gamma_vals[row] = float(val.flatten()[0]) if val.size > 0 else 0.0
        if gamma_vals is None:
            continue

        abs_vals = {r: abs(v) for r, v in gamma_vals.items()}
        sorted_rows = sorted(abs_vals, key=lambda r: -abs_vals[r])
        largest_row = sorted_rows[0]
        largest_val = abs_vals[largest_row]
        second_val = abs_vals[sorted_rows[1]] if len(sorted_rows) > 1 else 0.0

        # "하나만 지배적으로 크다" 판정 - [수정] 최댓값 자체가 유의미한
        # 크기(aa_min_signal_threshold 이상)일 때만 "진짜 신호"로 인정.
        # 이게 없으면 4개 값이 전부 0인 빈 TLS 항목도 잘못 통과함.
        is_dominant = (largest_val >= aa_min_signal_threshold) and (
            (second_val < 1e-6) or (largest_val / max(second_val, 1e-6) > aa_dominance_ratio_threshold)
        )

        # 목표 gamma_alpha와 얼마나 가까운지
        closeness = abs(largest_val - aa_target_gamma) / aa_target_gamma

        if is_dominant:
            candidates.append({
                'col': col, 'largest_row': largest_row, 'largest_val': largest_val,
                'closeness': closeness, 'all_gammas': gamma_vals,
            })

    candidates.sort(key=lambda c: c['closeness'])
    n_nonempty = sum(1 for c in candidates if c['largest_val'] >= aa_min_signal_threshold)
    print(f"'하나만 지배적으로 큰' 패턴을 보이는 열: {len(candidates)}개 / {n_cols}개")
    print(f"  (그중 실제로 값이 채워진(0이 아닌) 열: {n_nonempty}개 - "
          f"'20TLSfitted' 파일명과 비슷한 규모인지 확인)\n")
    print(f"gamma_alpha={aa_target_gamma/1e6:.1f}MHz/V와 가장 가까운 상위 5개:")
    print(f"{'열':>5} {'지배행':>7} {'값(MHz/V)':>12} {'차이(%)':>10}")
    print("-"*45)
    for c in candidates[:5]:
        print(f"{c['col']:>5} {c['largest_row']:>7} {c['largest_val']/1e6:>12.2f} "
              f"{c['closeness']*100:>10.2f}")

    if len(candidates) > 0:
        best = candidates[0]
        col, row = best['col'], best['largest_row']
        print(f"\n[최상위 후보 상세] 열={col}, 지배행={row} (이 행이 alpha일 가능성)")
        print(f"  4개 결합세기: {[f'{r}:{v/1e6:.2f}MHz/V' for r,v in best['all_gammas'].items()]}")

        v_row = aa_voltage_rows[row]
        v_val = np.array(f[tabledata2_refs[v_row, col]]).flatten()[0]
        freq_val = np.array(f[tabledata2_refs[aa_freq_row, col]]).flatten()[0]
        print(f"  대칭점 전압(행{v_row}): {v_val:.2f} V "
              f"(오늘 찾은 범위 -57~-56V와 비교)")
        print(f"  TLS 주파수(행{aa_freq_row}): {freq_val/1e9:.4f} GHz "
              f"(오늘 찾은 범위 5.12~5.17GHz와 비교)")
