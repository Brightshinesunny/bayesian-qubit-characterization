"""
tls_inspect_v3_deref.py
=================================================================
[목적] qdats(639개 segment)의 첫 번째 원소와, tdat.tabledata(TLS
피팅 결과 요약표)를 실제로 역참조해서 숫자 데이터를 확인합니다.

[논문과의 대응 - 오늘 확인한 물리적 배경]
  qdats[i].qdat.sweep1vals : 4개 게이트 전극(alpha,beta,gamma,delta)
    중 하나의 전압 스캔값. 논문 Fig.2a에서 "각 segment마다 전극 하나의
    전압을 1V씩 올린다"는 설명과 대응.
  qdats[i].qdat.sweep2vals : 플럭스 펄스 진폭(=큐빗 주파수 조정).
  qdats[i].qdat.obs.vals   : 큐빗 들뜬상태 확률(P(|1>)) 측정값.
    TLS와 공명하는 지점에서 어두운 픽셀(최솟값)로 나타남(논문 설명).
  tdat.tabledata           : 639개 segment 전체에 대한 피팅 결과
    요약표(7행). 논문의 결합세기 gamma_i에 해당하는 값이 포함되어
    있을 것으로 추정 - 실제 값을 보고 행 의미를 역추적함.
"""

import h5py
import numpy as np

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
aa_segment_index = 0   # [조정] 몇 번째 segment(0~638)를 자세히 볼지


def mat_string(f, dataset_or_ref):
    """MATLAB의 uint16 코드 배열 문자열을 사람이 읽을 수 있는 문자열로 변환."""
    try:
        if isinstance(dataset_or_ref, h5py.Reference):
            dataset_or_ref = f[dataset_or_ref]
        arr = np.array(dataset_or_ref)
        return ''.join(chr(int(c)) for c in arr.flatten())
    except Exception as e:
        return f"(문자열 변환 실패: {e})"


with h5py.File(aa_mat_filepath, 'r') as f:
    print("=" * 70)
    print(f"[STEP 1] qdats의 segment {aa_segment_index}번 역참조")
    print("=" * 70)

    qdat_refs = f['qdats']['qdat']   # shape=(639,1), 각 원소가 참조
    ref = qdat_refs[aa_segment_index, 0]
    qdat_group = f[ref]   # 역참조 - 실제 그룹으로 진입

    print(f"이 segment의 필드 목록: {list(qdat_group.keys())}")

    # --- sweep1name: 어떤 게이트 전극을 스캔했는지 ---
    sweep1name = mat_string(f, qdat_group['sweep1name'])
    print(f"\nsweep1name (스캔한 전극): '{sweep1name}'")

    # --- sweep1vals: 그 전극의 전압값들 ---
    sweep1vals = np.array(qdat_group['sweep1vals']).flatten()
    print(f"sweep1vals: shape={sweep1vals.shape}, "
          f"범위=[{sweep1vals.min():.4f}, {sweep1vals.max():.4f}]")
    print(f"  첫 5개 값: {sweep1vals[:5]}")

    # --- sweep2vals: 플럭스 펄스 진폭(큐빗 주파수 조정 축) ---
    sweep2vals = np.array(qdat_group['sweep2vals']).flatten()
    print(f"\nsweep2vals: shape={sweep2vals.shape}, "
          f"범위=[{sweep2vals.min():.4f}, {sweep2vals.max():.4f}]")

    # --- obs: 실제 측정 신호 ---
    obs_group = qdat_group['obs']
    print(f"\nobs 그룹의 필드 목록: {list(obs_group.keys())}")
    obs_vals = np.array(obs_group['vals'])
    print(f"obs.vals: shape={obs_vals.shape}, dtype={obs_vals.dtype}")
    if obs_vals.size < 50:
        print(f"  값 미리보기: {obs_vals.flatten()[:10]}")

    # --- observables: obs.vals 안의 각 열이 무엇을 의미하는지 이름표일 가능성 ---
    if 'observables' in qdat_group:
        obs_names = qdat_group['observables']
        print(f"\nobservables (obs.vals 각 열의 이름표로 추정): shape={obs_names.shape}, dtype={obs_names.dtype}")
        try:
            if obs_names.dtype == object:
                for i in range(min(obs_names.shape[0], 5)):
                    name = mat_string(f, obs_names[i, 0] if obs_names.ndim==2 else obs_names[i])
                    print(f"  [{i}]: '{name}'")
        except Exception as e:
            print(f"  이름 해독 실패: {e}")

    print("\n" + "=" * 70)
    print("[STEP 2] tdat.tabledata 구조 확인 (639개 segment의 피팅 결과 요약)")
    print("=" * 70)
    tabledata_refs = f['tdat']['tabledata']
    print(f"tabledata shape: {tabledata_refs.shape} (7행 x 639열로 추정)")

    # 첫 번째 열(segment 0)의 7개 행 값을 전부 확인
    print(f"\nsegment 0의 7개 행 값:")
    for row in range(tabledata_refs.shape[0]):
        ref = tabledata_refs[row, 0]
        val = f[ref]
        val_arr = np.array(val)
        if val_arr.dtype == np.uint16:
            # 문자열일 가능성
            decoded = mat_string(f, val)
            print(f"  행{row}: (문자열로 추정) '{decoded}'")
        else:
            print(f"  행{row}: shape={val_arr.shape}, "
                  f"값={val_arr.flatten()[:5] if val_arr.size>0 else '(비어있음)'}")
