"""
tls_combined_catalog_analysis.py
=================================================================
[B단계 1번] 3개 파일(Segments1to200, 200to400, 400to640)의
tabledata2 카탈로그를 합쳐서 59개 TLS로 통계 재분석. 표본이 20->59
개로 3배 가까이 커지니, 신뢰구간이 얼마나 좁아지는지가 핵심 확인점.

[시각화 방향]
"20개일 때 vs 59개일 때" 신뢰구간을 나란히 비교하는 그림을 만들어서,
표본 크기가 커질수록 불확실성이 실제로 줄어드는지 눈으로 보여줌 -
통계학 교과서적인 "표본 크기 효과"를 실측 데이터로 직접 확인.
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.getcwd())
import fano_bayesian_toolkit as bt   # 오늘 검증된 라이브러리 재사용

aa_mat_filepaths = [
    '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat',
    '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments200to400_20TLSfitted.mat',
    '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments400to640_20TLSfitted.mat',
]
aa_gamma_rows = {0: 'alpha(row0)', 2: 'beta(row2)', 9: 'gamma(row9)', 11: 'delta(row11)'}
aa_outlier_trim_pct = 3.0


def mat_string(f, dataset_or_ref):
    try:
        if isinstance(dataset_or_ref, h5py.Reference):
            dataset_or_ref = f[dataset_or_ref]
        arr = np.array(dataset_or_ref)
        return ''.join(chr(int(c)) for c in arr.flatten())
    except Exception:
        return None


def extract_gammas_from_file(mat_filepath):
    """파일 하나에서 채워진 TLS 항목들의 gamma 값(4개 전극)을 추출."""
    with h5py.File(mat_filepath, 'r') as f:
        tabledata2_refs = f['tdat']['tabledata2']
        n_rows, n_cols = tabledata2_refs.shape
        results = {row: [] for row in aa_gamma_rows}
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
            if max(abs(v) for v in gamma_vals.values()) >= 1e6:
                for row, v in gamma_vals.items():
                    results[row].append(v)
    return results


# =========================================================
# STEP 1. 파일별로 따로 추출한 뒤, "20개짜리(첫 파일만)"와
# "59개짜리(전부 합침)" 두 버전을 모두 만듦 - 비교를 위해
# =========================================================
gammas_first_file_only = extract_gammas_from_file(aa_mat_filepaths[0])

gammas_all_combined = {row: [] for row in aa_gamma_rows}
for filepath in aa_mat_filepaths:
    partial = extract_gammas_from_file(filepath)
    for row in aa_gamma_rows:
        gammas_all_combined[row].extend(partial[row])

n_first = len(gammas_first_file_only[0])
n_all = len(gammas_all_combined[0])
print(f"[STEP 1] 첫 파일만: {n_first}개 TLS / 3개 합침: {n_all}개 TLS")


# =========================================================
# STEP 2. 두 버전 각각 Gaussian/Robust + 신뢰구간 계산
# =========================================================
def analyze_gammas(gammas_dict):
    summary = {}
    for row, label in aa_gamma_rows.items():
        vals = np.array(gammas_dict[row])
        lo_thresh, hi_thresh = np.percentile(vals, [aa_outlier_trim_pct/2, 100-aa_outlier_trim_pct/2])
        trimmed = vals[(vals >= lo_thresh) & (vals <= hi_thresh)]

        mean_gaussian = np.mean(trimmed)
        median_robust = np.median(trimmed)
        std_gaussian = np.std(trimmed, ddof=1)
        sem = std_gaussian / np.sqrt(len(trimmed))

        mc_samples = np.random.default_rng(0).normal(mean_gaussian, sem, size=20000)
        lo, med, hi = bt.credible_interval(mc_samples, level=0.68)

        summary[row] = {'n': len(trimmed), 'mean': mean_gaussian, 'median_robust': median_robust,
                          'ci_lo': lo, 'ci_hi': hi, 'ci_width': hi-lo}
    return summary

summary_first = analyze_gammas(gammas_first_file_only)
summary_all = analyze_gammas(gammas_all_combined)

print("\n[STEP 2] 20개 vs 59개 비교")
print(f"{'전극':>16} {'20개 CI':>28} {'59개 CI':>28} {'폭 감소율':>10}")
for row, label in aa_gamma_rows.items():
    s1, s2 = summary_first[row], summary_all[row]
    ci1 = f"[{s1['ci_lo']/1e6:.1f},{s1['ci_hi']/1e6:.1f}]"
    ci2 = f"[{s2['ci_lo']/1e6:.1f},{s2['ci_hi']/1e6:.1f}]"
    reduction = (1 - s2['ci_width']/s1['ci_width']) * 100
    print(f"{label:>16} {ci1:>28} {ci2:>28} {reduction:>9.1f}%")


# =========================================================
# STEP 3. 시각화 - 표본 크기 효과를 한눈에
# =========================================================
fig, ax = plt.subplots(figsize=(10, 6))
labels = list(aa_gamma_rows.values())
x_pos = np.arange(len(labels))

for i, row in enumerate(aa_gamma_rows):
    s1, s2 = summary_first[row], summary_all[row]
    # 20개 결과 (왼쪽, 옅은 색)
    ax.errorbar(x_pos[i]-0.15, s1['mean']/1e6,
                 yerr=[[  (s1['mean']-s1['ci_lo'])/1e6 ], [ (s1['ci_hi']-s1['mean'])/1e6 ]],
                 fmt='o', color='lightblue', capsize=5, markersize=10, label='20개(1파일)' if i==0 else "")
    # 59개 결과 (오른쪽, 진한 색)
    ax.errorbar(x_pos[i]+0.15, s2['mean']/1e6,
                 yerr=[[ (s2['mean']-s2['ci_lo'])/1e6 ], [ (s2['ci_hi']-s2['mean'])/1e6 ]],
                 fmt='s', color='darkblue', capsize=5, markersize=10, label='59개(3파일 합침)' if i==0 else "")

ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylabel('결합세기 gamma (MHz/V)')
ax.set_title('표본 크기 효과: 20개 TLS vs 59개 TLS 신뢰구간 비교\n'
              '(표본이 커질수록 신뢰구간이 좁아지는지 확인)')
ax.legend()
ax.axhline(0, color='gray', lw=0.5)
plt.tight_layout()
plt.savefig('./outputs/tls_combined_catalog_comparison.png', dpi=140)
print(f"\n그래프 저장 완료: ./outputs/tls_combined_catalog_comparison.png")
