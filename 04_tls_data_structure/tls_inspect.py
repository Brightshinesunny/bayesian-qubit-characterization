"""
tls_inspect.py
=================================================================
[목적] TLS 매핑 데이터셋(.mat 파일)의 구조를 처음 확인하는 스크립트.
MATLAB 프로그램 자체는 필요 없고, 파이썬만으로 읽습니다.

[.mat 파일의 두 가지 버전 - 왜 두 가지 방법을 다 준비하는가]
  - MATLAB v7 이하(구버전): scipy.io.loadmat로 바로 읽힘. 내부적으로
    일반적인 이진 파일 형식.
  - MATLAB v7.3 이상(신버전): 내부적으로 HDF5(Hierarchical Data
    Format) 형식으로 저장됨 - 대용량 데이터(이 데이터셋처럼 500만
    측정치 이상)를 다룰 때 MATLAB이 자동으로 이 형식을 씀. 이 경우
    scipy.io.loadmat이 에러를 내고, 대신 h5py 라이브러리로 읽어야
    함. 어느 버전인지는 파일을 열어보기 전엔 알 수 없으므로, 이
    스크립트는 먼저 scipy로 시도하고 실패하면 자동으로 h5py로
    넘어가도록 만들었습니다.
"""

import os

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/PUT_FILENAME_HERE.mat'
    # [조정] 실제 다운받은 .mat 파일 경로로 바꿔주세요.


def inspect_mat_file(filepath):
    print(f"파일: {filepath}")
    print("=" * 60)

    # --- 1차 시도: scipy.io.loadmat (구버전 MATLAB 형식) ---
    try:
        from scipy.io import loadmat
        data = loadmat(filepath)
        print("[성공] scipy.io.loadmat으로 읽힘 (구버전 MATLAB 형식)")
        print("\n포함된 변수 목록:")
        for key in data.keys():
            if key.startswith('__'):
                continue   # __header__, __version__ 같은 메타정보는 건너뜀
            val = data[key]
            shape = getattr(val, 'shape', 'N/A')
            dtype = getattr(val, 'dtype', type(val))
            print(f"  {key:20s}: shape={shape}, dtype={dtype}")
        return data, 'scipy'
    except NotImplementedError:
        print("[실패] scipy로 읽기 실패 - v7.3(HDF5) 형식으로 추정됨. h5py로 재시도...")
    except Exception as e:
        print(f"[실패] scipy 시도 중 다른 에러: {e}")
        print("h5py로 재시도...")

    # --- 2차 시도: h5py (신버전 MATLAB v7.3, HDF5 형식) ---
    try:
        import h5py
        with h5py.File(filepath, 'r') as f:
            print("\n[성공] h5py로 읽힘 (v7.3/HDF5 형식)")
            print("\n최상위 그룹/데이터셋 목록:")

            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  {name:30s}: shape={obj.shape}, dtype={obj.dtype}")
                else:
                    print(f"  {name:30s}: (그룹)")

            f.visititems(print_structure)
        return None, 'h5py'
    except Exception as e:
        print(f"[실패] h5py로도 읽기 실패: {e}")
        return None, None


if __name__ == "__main__":
    if os.path.exists(aa_mat_filepath):
        inspect_mat_file(aa_mat_filepath)
    else:
        print(f"파일을 찾을 수 없습니다: {aa_mat_filepath}")
        print("aa_mat_filepath를 실제 파일 경로로 수정해주세요.")
