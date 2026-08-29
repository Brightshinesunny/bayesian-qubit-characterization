"""
tls_B_fast_start.py
=================================================================
[B단계 - 빠른 시작] 오늘 만든 도구 전부를 재사용해서, 새 TLS
구간(Segments200to400 또는 400to640)을 빠르게 훑어봄. "오늘 배운 게
진짜 빨라졌는지" 확인하는 것 자체가 이 스크립트의 목적.

[오늘과 다른 점 - 이번엔 처음부터 검증 로봇을 같이 씀]
오늘은 문제를 겪은 "뒤에" 검증 도구를 만들었지만, 이번엔 처음부터
validation_robots.py를 같이 돌려서, 계통오차/prior/모델 문제를
조기에 잡아낼 수 있는지 확인.
"""

import h5py
import numpy as np
import sys, os
sys.path.insert(0, os.getcwd())
import tls_avcross_pipeline as pipeline
import tls_avcross_models as models
import validation_robots as vr

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments200to400_20TLSfitted.mat'
    # [조정] 400to640을 먼저 보고 싶으면 파일명만 바꾸면 됨.


def mat_string(f, dataset_or_ref):
    try:
        if isinstance(dataset_or_ref, h5py.Reference):
            dataset_or_ref = f[dataset_or_ref]
        arr = np.array(dataset_or_ref)
        return ''.join(chr(int(c)) for c in arr.flatten())
    except Exception:
        return None


# =========================================================
# STEP 1. 구조가 오늘 파일과 같은지 빠르게 확인
# =========================================================
with h5py.File(aa_mat_filepath, 'r') as f:
    print("[STEP 1] 최상위 구조 확인")
    print("  키 목록:", list(f.keys()))

    n_segments = f['qdats']['qdat'].shape[0]
    print(f"  segment 개수: {n_segments}")

    tabledata2_shape = f['tdat']['tabledata2'].shape
    print(f"  tabledata2 shape: {tabledata2_shape} (오늘 파일은 13x200이었음)")

    # 처음 몇 개 segment의 전극 순서도 확인 (라운드로빈 구조 재확인)
    print("\n  처음 8개 segment 전극 순서:")
    for i in range(min(8, n_segments)):
        qdat_group = f[f['qdats']['qdat'][i, 0]]
        name = mat_string(f, qdat_group['sweep1name'])
        sweep1vals = np.array(qdat_group['sweep1vals']).flatten()
        print(f"    raw_index={i}: 전극={name}, 전압범위=[{sweep1vals.min():.1f},{sweep1vals.max():.1f}]")


# =========================================================
# STEP 2. [빠른 경로] tabledata2 카탈로그 즉시 해독
# =========================================================
print("\n[STEP 2] tabledata2 카탈로그 (원저자가 이미 찾은 TLS들)")
with h5py.File(aa_mat_filepath, 'r') as f:
    tabledata2_refs = f['tdat']['tabledata2']
    n_rows, n_cols = tabledata2_refs.shape
    gamma_rows = [0, 2, 9, 11]
    freq_row = 4

    filled_columns = []
    for col in range(n_cols):
        gamma_vals = {}
        skip = False
        for row in gamma_rows:
            ref = tabledata2_refs[row, col]
            val = np.array(f[ref])
            if val.dtype == np.uint16:
                skip = True
                break
            gamma_vals[row] = float(val.flatten()[0]) if val.size > 0 else 0.0
        if skip:
            continue
        max_abs = max(abs(v) for v in gamma_vals.values())
        if max_abs >= 1e6:
            freq_val = np.array(f[tabledata2_refs[freq_row, col]]).flatten()[0]
            filled_columns.append({'col': col, 'gammas': gamma_vals, 'freq': freq_val})

    print(f"  채워진 TLS 항목: {len(filled_columns)}개 / {n_cols}개 (오늘 파일은 20개였음)")
    for c in filled_columns[:5]:
        g = c['gammas']
        print(f"    열{c['col']}: freq={c['freq']/1e9:.4f}GHz, "
              f"gamma(행0,2,9,11)=[{g[0]/1e6:.1f},{g[2]/1e6:.1f},{g[9]/1e6:.1f},{g[11]/1e6:.1f}]MHz/V")


# =========================================================
# STEP 3. [검증 로봇] prior 범위 자동 점검 - 오늘 같은 버그 조기 발견
# =========================================================
print("\n[STEP 3] 검증 로봇 - prior 범위 사전 점검")
prior_bounds_check = {'f_TLS0': (5.10, 5.20), 'gamma_stark': (-0.2, 0.2), 'g': (1e-6, 0.03)}
    # 오늘 최종 확정한 안전한 범위. 혹시 새 데이터가 다른 스케일이면
    # 여기서 미리 경고가 뜰 것.
warnings = vr.check_prior_sanity(prior_bounds_check)
print("  경고:", warnings if warnings else "없음(정상)")

print("\n[요약] STEP 1~3까지 몇 분 안에 끝났다면, 오늘 배운 도구 재사용이")
print("실제로 시간을 크게 절약한다는 증거. 다음은 특정 TLS 항목 하나를")
print("골라서 raw 픽셀 검증(오늘 seg12처럼)까지 갈지 결정하면 됨.")
