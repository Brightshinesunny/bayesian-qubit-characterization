"""
verify_loader_equivalence.py
=================================================================
원저자(Sett et al.)의 "고정 줄번호 인덱싱" 방식과, 우리 zenodo_loader.py의
"%%%%%%%%%% 마커 기반" 방식이 정말로 같은 결과를 내는지 직접 대조 검증.

두 단계로 구성:
  STEP 1. 작은 가짜 파일(power_size=5, freq_size=8)을 만들어, 두 방식의
          결과가 완전히 일치하는지 확인 (원리 검증)
  STEP 2. 실제 Zenodo 데이터 파일(S21_power_sweep_152...txt 등)에 대해
          두 방식을 동시에 돌려서, 실제 데이터에서도 일치하는지 확인
          (실전 검증) - 파일 경로만 바꿔서 사용
"""

import numpy as np
from io import StringIO
import zenodo_loader


# =========================================================
# 원저자 방식을 함수로 감싼 것 (PBB_paper_figures_Figure1.py의
# 로딩 로직을 그대로 재현 - 로직 자체는 전혀 바꾸지 않음)
# =========================================================
def load_author_style(filepath):
    """
    원저자 코드와 동일하게, 고정된 줄 번호(인덱스)를 계산해서
    power, freq, I, Q를 읽어들이는 함수.
    """
    with open(filepath) as f:
        data = f.readlines()
    data = [line[:-1] if line.endswith('\n') else line for line in data]
        # 원저자 코드의 [line[:-1] for line in data] 와 동일:
        # 각 줄 끝의 개행문자(\n)를 제거.

    power = np.genfromtxt(StringIO(data[4]), delimiter='\t')
    freq = np.genfromtxt(StringIO(data[10]), delimiter='\t')
    power_size = np.size(power)
    freq_trace_size = np.size(freq)

    I_old = np.genfromtxt(StringIO(data[16]), delimiter='\t')
    for i in range(power_size - 1):
        I1 = np.genfromtxt(StringIO(data[16 + (i + 1) * 2]), delimiter='\t')
        I_old = np.append(I_old, I1)
    I21 = I_old.reshape([power_size, freq_trace_size])

    init = 16 + power_size * 2 + 4
    Q_old = np.genfromtxt(StringIO(data[init]), delimiter='\t')
    for i in range(power_size - 1):
        Q1 = np.genfromtxt(StringIO(data[init + (i + 1) * 2]), delimiter='\t')
        Q_old = np.append(Q_old, Q1)
    Q21 = Q_old.reshape([power_size, freq_trace_size])

    return {'power_dbm': power, 'freq_ghz': freq, 'I': I21, 'Q': Q21}


def compare_two_methods(filepath, label=""):
    """
    두 방식으로 같은 파일을 읽어서, 각 항목이 일치하는지 하나씩 출력.
    np.allclose: 부동소수점은 완전히 똑같지 않을 수 있어(계산 순서에 따른
    미세한 반올림 오차), "거의 같다(아주 작은 오차 이내)"를 판정하는 함수.
    """
    print(f"\n{'='*55}")
    print(f"검증 대상: {label if label else filepath}")
    print('='*55)

    result_author = load_author_style(filepath)
    result_ours = zenodo_loader.load_power_sweep_txt(filepath)

    checks = [
        ('power_dbm', 'power_dbm'),
        ('freq_ghz', 'freq_ghz'),
        ('I', 'I'),
        ('Q', 'Q'),
    ]

    all_match = True
    for key_a, key_o in checks:
        if key_a not in result_author or key_o not in result_ours:
            print(f"  {key_a:12s}: 한쪽에만 존재 (건너뜀)")
            continue
        a, o = result_author[key_a], result_ours[key_o]
        shape_ok = a.shape == o.shape
        value_ok = shape_ok and np.allclose(a, o)
        status = "일치" if value_ok else "불일치!!"
        print(f"  {key_a:12s}: shape 원저자={a.shape}, 우리={o.shape}  ->  {status}")
        all_match = all_match and value_ok

    print(f"\n  최종 결과: {'모두 일치 (검증 성공)' if all_match else '불일치 발견 - 확인 필요'}")
    return all_match


# =========================================================
# STEP 1. 작은 가짜 파일로 원리 검증
# =========================================================
def make_fake_file(power_size, freq_size, out_path, seed=0):
    """원저자 파일과 동일한 섹션/줄 구조를 갖는 가짜 파일을 생성"""
    rng = np.random.default_rng(seed)
    lines = []
    lines.append('')
    lines.append('%%%%%%%%%% R_power_source ~dBm')
    lines.append('')
    lines.append('')
    lines.append('\t'.join(str(float(p)) for p in range(power_size)))
    lines.append('')
    lines.append('')
    lines.append('%%%%%%%%%% ~Freq. trace GHz')
    lines.append('')
    lines.append('')
    lines.append('\t'.join(str(10.0 + 0.1 * i) for i in range(freq_size)))
    lines.append('')
    lines.append('')
    lines.append('%%%%%%%%%% I')
    lines.append('')
    lines.append('')

    I_true = rng.standard_normal((power_size, freq_size))
    for row in I_true:
        lines.append('\t'.join(str(v) for v in row))
        lines.append('')

    lines.append('')
    lines.append('%%%%%%%%%% Q')
    lines.append('')
    lines.append('')

    Q_true = rng.standard_normal((power_size, freq_size))
    for row in Q_true:
        lines.append('\t'.join(str(v) for v in row))
        lines.append('')

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    import os

    print("#" * 55)
    print("# STEP 1. 가짜 파일로 원리 검증")
    print("#" * 55)
    fake_path = '/tmp/verify_fake.txt'
    make_fake_file(power_size=5, freq_size=8, out_path=fake_path)
    compare_two_methods(fake_path, label="가짜 파일 (power=5, freq=8)")

    print("\n\n" + "#" * 55)
    print("# STEP 2. 실제 Zenodo 데이터 파일로 실전 검증")
    print("#" * 55)
    print("아래 aa_real_data_path를 실제 다운로드한 .txt 파일 경로로")
    print("바꾼 뒤 이 스크립트를 다시 실행하세요.\n")

    aa_real_data_path = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module01/Figure1/S21_power_sweep_152_data_210823_21h50m19s.txt'

    if os.path.exists(aa_real_data_path):
        compare_two_methods(aa_real_data_path, label="실제 Zenodo 데이터 (kappa=8MHz 파일)")
    else:
        print(f"  [안내] 파일을 찾을 수 없습니다: {aa_real_data_path}")
        print("  aa_real_data_path 변수를 실제 파일 위치로 수정 후 다시 실행하세요.")
