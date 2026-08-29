"""
tls_inspect_v2.py
=================================================================
[목적] TLS 데이터셋(.mat, MATLAB v7.3/HDF5 형식)의 구조를 h5py로
탐색합니다. Readme.txt에서 확인된 구조:
  qdats: 구조체 "배열" (4개 게이트 전압 스윕 각각이 1개 segment)
  qset:  구조체 "하나" (실험 전반 설정)
  tdat:  피팅된 TLS trace 기울기

[MATLAB v7.3 구조체 배열의 HDF5 저장 방식 - 왜 "역참조"가 필요한가]
MATLAB이 struct "배열"(qdats처럼 여러 개)을 HDF5로 저장할 때는,
보통 각 원소를 별도의 HDF5 그룹으로 만들고, qdats 자체는 그 그룹들을
"가리키는 참조(reference)"들의 배열로 저장합니다. 즉 qdats[0]을
그냥 읽으면 실제 데이터가 아니라 "어디를 봐야 하는지"를 나타내는
포인터만 나옵니다 - h5py에서는 이걸 h5py.Reference 타입으로 표현하고,
f[reference] 형태로 실제 위치를 다시 열어야(역참조) 진짜 데이터에
접근할 수 있습니다. 문자열도 uint16 배열로 저장되는 경우가 많아
(MATLAB의 char 배열이 UTF-16 유사 형태), 사람이 읽을 수 있는 문자열로
따로 변환해야 합니다.
"""

import h5py
import numpy as np

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
    # [조정] 실제 업로드하신 경로로 바꿔주세요.


def mat_string_to_str(f, dataset):
    """
    MATLAB이 문자열을 uint16 코드 배열로 저장한 경우, 사람이 읽을 수
    있는 파이썬 문자열로 변환. 실패하면 원본을 그대로 반환(다른
    형태일 수 있으므로).
    """
    try:
        arr = np.array(dataset)
        return ''.join(chr(c) for c in arr.flatten())
    except Exception:
        return dataset


def explore_group(f, obj, name, depth=0, max_depth=3):
    """HDF5 그룹/데이터셋을 재귀적으로 탐색하며 구조를 출력."""
    indent = "  " * depth
    if depth > max_depth:
        print(f"{indent}{name}: (더 깊이 들어가지 않음 - max_depth 도달)")
        return

    if isinstance(obj, h5py.Dataset):
        print(f"{indent}{name}: Dataset, shape={obj.shape}, dtype={obj.dtype}")
        # 작은 데이터셋이면 실제 값 일부를 미리보기
        if obj.size > 0 and obj.size < 20:
            try:
                print(f"{indent}  값 미리보기: {np.array(obj).flatten()[:10]}")
            except Exception:
                pass
        # dtype이 object면 참조(reference) 배열일 가능성이 높음
        if obj.dtype == h5py.special_dtype(ref=h5py.Reference) or obj.dtype == object:
            print(f"{indent}  [참조 배열로 추정 - 역참조 시도]")
            try:
                first_ref = obj[0, 0] if obj.ndim == 2 else obj[0]
                if isinstance(first_ref, h5py.Reference):
                    dereferenced = f[first_ref]
                    print(f"{indent}  -> 역참조된 첫 원소의 타입: {type(dereferenced)}")
                    if isinstance(dereferenced, h5py.Group):
                        print(f"{indent}  -> 역참조된 첫 원소의 키들: {list(dereferenced.keys())}")
            except Exception as e:
                print(f"{indent}  역참조 실패: {e}")
    elif isinstance(obj, h5py.Group):
        print(f"{indent}{name}: Group, 키들={list(obj.keys())}")
        for key in obj.keys():
            explore_group(f, obj[key], key, depth+1, max_depth)


with h5py.File(aa_mat_filepath, 'r') as f:
    print("=" * 70)
    print("최상위 키 목록:", list(f.keys()))
    print("=" * 70)

    for top_key in f.keys():
        if top_key.startswith('#'):
            continue
        print(f"\n{'='*20} {top_key} {'='*20}")
        explore_group(f, f[top_key], top_key, depth=0, max_depth=2)
