"""
tls_heterogeneity_check.py
=================================================================
[이질성(heterogeneity) 확인] 3개 파일을 그냥 합쳐서 59개로 봤을 때
일부 전극(gamma, delta)에서 신뢰구간이 오히려 넓어지는 현상이
관찰됨. "3개 파일이 정말 같은 모집단인가, 아니면 파일마다 서로
다른 특성을 가진 별개 집단인가"를 직접 확인.

[통계 원리]
전체 분산 = 파일"내" 분산 + 파일 "간" 분산
파일 간 평균이 서로 크게 다르면(파일 간 분산이 큼), 그냥 합쳐서
보면 "가짜로 퍼진" 것처럼 보일 수 있음 - 이걸 나눠서 봐야 진짜
원인을 알 수 있음.
"""

import h5py
import numpy as np
import sys, os
sys.path.insert(0, os.getcwd())
import fano_bayesian_toolkit as bt

aa_mat_filepaths = {
    '파일1(1-200)': '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat',
    '파일2(200-400)': '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments200to400_20TLSfitted.mat',
    '파일3(400-640)': '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments400to640_20TLSfitted.mat',
}
    # [수정] 오늘 계속 써온 Module02/TLS 경로로 통일 (이전 실수 수정).
aa_gamma_rows = {0: 'alpha(row0)', 2: 'beta(row2)', 9: 'gamma(row9)', 11: 'delta(row11)'}


def mat_string(f, dataset_or_ref):
    try:
        if isinstance(dataset_or_ref, h5py.Reference):
            dataset_or_ref = f[dataset_or_ref]
        arr = np.array(dataset_or_ref)
        return ''.join(chr(int(c)) for c in arr.flatten())
    except Exception:
        return None


def extract_gammas_from_file(mat_filepath):
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
# STEP 1. 파일별로 각각 통계 계산
# =========================================================
per_file_stats = {}
for label, filepath in aa_mat_filepaths.items():
    gammas = extract_gammas_from_file(filepath)
    per_file_stats[label] = {}
    for row, elec_label in aa_gamma_rows.items():
        vals = np.array(gammas[row])
        mean_val = np.mean(vals)
        sem = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
        per_file_stats[label][row] = {'n': len(vals), 'mean': mean_val, 'sem': sem}


# =========================================================
# STEP 2. 전극별로, 파일 간 평균이 얼마나 다른지 표로 확인
# =========================================================
print(f"{'전극':>16} " + " ".join(f"{label:>18}" for label in aa_mat_filepaths))
print("-" * (16 + 19*len(aa_mat_filepaths)))
for row, elec_label in aa_gamma_rows.items():
    row_str = f"{elec_label:>16} "
    means = []
    for label in aa_mat_filepaths:
        s = per_file_stats[label][row]
        row_str += f"{s['mean']/1e6:>8.1f}±{s['sem']/1e6:<8.1f} "
        means.append(s['mean'])
    print(row_str)

    # 파일 간 평균의 퍼짐(표준편차)과, 각 파일 내 SEM의 평균을 비교
    between_file_std = np.std(means, ddof=1)
    within_file_sem_avg = np.mean([per_file_stats[label][row]['sem'] for label in aa_mat_filepaths])
    ratio = between_file_std / (within_file_sem_avg + 1e-10)
    verdict = "이질성 큼(파일 간 차이가 지배적)" if ratio > 2 else "동질적(파일 간 차이가 SEM 수준)"
    print(f"  -> 파일간 표준편차={between_file_std/1e6:.1f}, 파일내 평균SEM={within_file_sem_avg/1e6:.1f}, "
          f"비율={ratio:.2f} [{verdict}]\n")
