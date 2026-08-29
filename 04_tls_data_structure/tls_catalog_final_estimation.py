"""
tls_catalog_final_estimation.py
=================================================================
[오늘의 최종 결론 - "밑작업 없이 가능한 것"]
raw 픽셀 데이터는 원저자의 "quin" 프로그램 없이는 계통오차/비선형
매핑/딥추적 문제로 사실상 다루기 어렵다는 게 오늘 하루로 확인됨.
대신, 원저자가 이미 공개한 tdat.tabledata2(20개 TLS의 결합세기
카탈로그)를 "그 자체로 하나의 작은 데이터셋"으로 삼아, 오늘 확립한
노이즈 처리/매개변수 추정 절차(Gaussian vs Robust 교차검증 + 피셔
행렬 신뢰구간)를 적용합니다 - 이건 quin 프로그램 없이도 완전히
가능한, "정상 속도"에 맞는 작업입니다.

[분석 대상]
tabledata2의 4개 결합세기 행(0,2,9,11 = 4개 게이트 전극)에서, 20개
채워진 TLS 항목의 값들을 각 전극별로 모아, "이 전극의 전형적인
결합세기가 얼마인지"를 강건하게 추정합니다.
"""

import h5py
import numpy as np
import sys, os
sys.path.insert(0, os.getcwd())
import fano_bayesian_toolkit as bt   # 오늘 이미 검증된 라이브러리 재사용

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
aa_gamma_rows = {0: 'alpha(row0)', 2: 'beta(row2)', 9: 'gamma(row9)', 11: 'delta(row11)'}
    # [주의] 어느 행이 물리적으로 어느 게이트(alpha/beta/gamma/delta)에
    # 대응하는지는 오늘 확정하지 못했습니다 - 그래서 이름표는 임시로
    # "row 번호"를 그대로 붙였습니다. 4개의 "서로 다른 게이트"라는
    # 사실만 확실하고, 어느 게이트인지는 다음 과제로 남겨둡니다.
aa_outlier_trim_pct = 3.0
    # [오늘 확립한 표준] 5% 대신 여기서는 데이터가 20개뿐이라 3%로
    # (실질적으로 상하위 1개 정도만) 완화 - 표본이 작을수록 과도한
    # 제거는 오히려 정보를 너무 많이 잃게 됨.


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
    freq_row = 4

    # =====================================================
    # STEP 1. 채워진 20개 TLS의 4개 결합세기 값 추출
    # =====================================================
    all_gammas = {row: [] for row in aa_gamma_rows}
    for col in range(n_cols):
        gamma_vals = {}
        skip = False
        for row in aa_gamma_rows:
            ref = tabledata2_refs[row, col]
            val = np.array(f[ref])
            if val.dtype == np.uint16:
                skip = True
                break
            gamma_vals[row] = float(val.flatten()[0]) if val.size > 0 else 0.0
        if skip:
            continue
        max_abs = max(abs(v) for v in gamma_vals.values())
        if max_abs >= 1e6:   # 채워진 항목 판정 기준(어제 확립)
            for row, v in gamma_vals.items():
                all_gammas[row].append(v)

    print(f"채워진 TLS 항목 수: {len(all_gammas[0])}개\n")

    # =====================================================
    # STEP 2. 전극별 Gaussian vs Robust 교차검증 + 신뢰구간
    # =====================================================
    print(f"{'전극':>16} {'n':>4} {'Gaussian평균':>14} {'Robust(중앙값)':>16} "
          f"{'MAD sigma':>12} {'68%CI':>28}")
    print("-"*100)

    for row, label in aa_gamma_rows.items():
        vals = np.array(all_gammas[row])

        # --- 이상치 3% 제거 (양쪽 합쳐서) ---
        lo_thresh, hi_thresh = np.percentile(vals, [aa_outlier_trim_pct/2, 100-aa_outlier_trim_pct/2])
        trimmed = vals[(vals >= lo_thresh) & (vals <= hi_thresh)]

        # --- Gaussian(단순 평균) ---
        mean_gaussian = np.mean(trimmed)
        std_gaussian = np.std(trimmed, ddof=1)

        # --- Robust(중앙값 + MAD) ---
        median_robust = np.median(trimmed)
        mad = np.median(np.abs(trimmed - median_robust))
        sigma_mad = mad * 1.4826

        # --- [라이브러리 재사용] 피셔/베이지안 방식 신뢰구간 ---
        # 평균 추정의 표준오차(SEM)를 이용한 몬테카를로 샘플링으로
        # "전극의 진짜 평균 결합세기"에 대한 68% 신뢰구간을 구함.
        sem = std_gaussian / np.sqrt(len(trimmed))
        mc_samples = np.random.default_rng(0).normal(mean_gaussian, sem, size=20000)
        lo, med, hi = bt.credible_interval(mc_samples, level=0.68)
            # [라이브러리 재사용] fano_bayesian_toolkit.credible_interval -
            # 오늘 오전 Fano 데이터 분석에서 이미 검증된 함수를 그대로 사용.

        ci_str = f"[{lo/1e6:.1f}, {hi/1e6:.1f}] MHz/V"
        print(f"{label:>16} {len(trimmed):>4} {mean_gaussian/1e6:>13.1f} {median_robust/1e6:>15.1f} "
              f"{sigma_mad/1e6:>11.1f} {ci_str:>28}")

        # 이상치로 제거된 값이 있으면 보고
        n_removed = len(vals) - len(trimmed)
        if n_removed > 0:
            removed_vals = vals[(vals < lo_thresh) | (vals > hi_thresh)]
            print(f"  -> 이상치 {n_removed}개 제거: {np.round(removed_vals/1e6,1)} MHz/V")

    print("\n[해석]")
    print("  Gaussian평균과 Robust(중앙값)이 서로 가까우면, 그 전극의 결합세기")
    print("  추정치가 이상치에 크게 흔들리지 않는 안정적인 값이라는 뜻입니다.")
    print("  둘이 크게 다르면, 그 전극에는 극단적으로 강하게 결합된 TLS가")
    print("  하나(또는 소수) 존재해서 평균을 왜곡시키고 있다는 신호입니다.")
