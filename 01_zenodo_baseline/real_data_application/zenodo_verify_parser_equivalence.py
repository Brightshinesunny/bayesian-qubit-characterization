"""
verify_parser_equivalence.py
=================================================================
Zenodo 데이터를 읽는 두 가지 방식이 "정말 똑같은 결과"를 내는지
직접 검증하는 스크립트입니다.

  [방식 A] 원저자 방식 (고정 인덱스)
    원저자(Sett et al.)의 PBB_paper_figures_Figure1.py 코드가 실제로
    사용한 방식. "데이터가 몇 번째 줄에 있는지"를 미리 손으로 계산해서
    (예: data[4], data[16+i*2]) 그 줄만 정확히 집어서 읽음.

  [방식 B] 우리 방식 (마커 기반, zenodo_loader.py)
    "%%%%%%%%%% 섹션이름"이라는 이름표를 실시간으로 인식해서, 그
    이름표 아래에 있는 줄들을 자동으로 모으는 방식. 줄 번호를 몰라도
    되고, 빈 줄이 몇 개든 상관없이 동작.

이 스크립트는 원저자 파일과 똑같은 구조를 가진 "가짜 검증용 파일"을
직접 만들고, 두 방식으로 각각 읽어서 결과(전력값, 주파수값, I, Q
배열)가 완전히 일치하는지 확인합니다. 이렇게 "같은 입력에 대해 두
독립적인 방법이 같은 출력을 내는지" 확인하는 것을 교차검증
(cross-validation)이라고 하며, 오늘까지 계속 사용해온 검증 원칙
("독립적인 방법 두 개가 일치하면 신뢰도가 올라간다" - 예: MCMC vs
피셔 행렬 비교와 같은 발상)을 파일 파싱 단계에도 적용한 것입니다.
"""

import numpy as np
from io import StringIO
import sys
import os

# zenodo_loader.py를 같은 폴더에서 불러오기 위한 경로 설정
sys.path.insert(0, os.getcwd())
    # os.getcwd(): 현재 작업 디렉토리를 기준으로 zenodo_loader.py를 찾음.
    # (Colab 등에서 __file__이 안정적으로 동작하지 않는 경우가 있어
    #  os.path.dirname(os.path.abspath(__file__)) 대신 이 방식을 사용)
import zenodo_loader


# =========================================================
# STEP 0. 분석자 설정값 (aa_ 접두사)
# =========================================================
aa_power_size = 5    # 가짜 검증 파일에 넣을 전력(power) 값의 개수.
                      # 실제 파일은 61개지만, 여기서는 사람이 눈으로
                      # 결과를 확인하기 쉽도록 작은 수로 축소해서 검증.
aa_freq_size = 8      # 가짜 검증 파일에 넣을 주파수 값의 개수 (실제는 101개)
aa_random_seed = 0    # 가짜 I, Q 데이터를 무작위로 채울 때 쓰는 난수 시드.
                       # 시드를 고정해두면 스크립트를 몇 번을 다시 돌려도
                       # 항상 같은 가짜 데이터가 생성되어 결과 재현이 보장됨.
aa_temp_file_path = '/tmp/verify_parser_test.txt'
    # 가짜 검증 파일을 임시로 저장할 경로. 이 파일 자체는 실제 실험
    # 데이터가 아니라 "파서 두 개를 비교하기 위한 소품"일 뿐이므로,
    # 검증이 끝나면 지워도 무방함.


# =========================================================
# STEP 1. 원저자 파일 구조를 그대로 흉내낸 가짜 검증 파일 만들기
# =========================================================
# 원저자 파일의 실제 구조 (앞서 확인한 원저자 코드의 인덱스 계산에서
# 역으로 추론한 것):
#
#   줄0  : (빈 줄)
#   줄1  : '%%%%%%%%%% R_power_source ~dBm'   <- 섹션 이름표(헤더)
#   줄2,3: (빈 줄 2개)
#   줄4  : 전력값들 (data[4]로 원저자가 직접 참조)
#   줄5,6: (빈 줄 2개)
#   줄7  : '%%%%%%%%%% ~Freq. trace GHz'
#   줄8,9: (빈 줄 2개)
#   줄10 : 주파수값들 (data[10])
#   줄11,12: (빈 줄 2개)
#   줄13 : '%%%%%%%%%% I'
#   줄14,15: (빈 줄 2개)
#   줄16 : 첫 번째 I 데이터 줄 (data[16])
#   줄17 : (빈 줄 1개 - 이후 I 데이터 줄들 사이는 빈 줄 "1개"씩만 있음)
#   줄18 : 두 번째 I 데이터 줄 (data[16+1*2])
#   ... (이런 식으로 I 데이터가 2줄 간격으로 aa_power_size개 반복)
#   그 다음 : (빈줄, 빈줄, '%%%%%%%%%% Q' 헤더, 빈줄, 빈줄) 순서로
#             동일한 "헤더 앞뒤 2줄 공백" 패턴이 한 번 더 나온 뒤 Q 데이터 시작
#
# 이 구조를 그대로 재현하지 않으면 원저자 방식(고정 인덱스)이 엉뚱한
# 줄을 읽어버리므로, 검증을 위해서는 정확히 이 패턴을 지켜야 함.

rng = np.random.default_rng(aa_random_seed)
    # np.random.default_rng: numpy가 권장하는 최신 난수 생성기. 시드를
    # 넣으면 항상 같은 순서의 난수를 재현할 수 있음.

lines = []
lines.append('')                                              # 줄0
lines.append('%%%%%%%%%% R_power_source ~dBm')                # 줄1
lines.append('')                                              # 줄2
lines.append('')                                              # 줄3
lines.append('\t'.join(str(float(p)) for p in range(aa_power_size)))
    # 줄4: 전력값 0,1,2,...,(aa_power_size-1)을 탭으로 이어붙인 문자열
    #      (실제 파일처럼 -50.0 ~ 10.0 dBm으로 안 해도, 파싱이 맞는지만
    #       확인하는 목적이라 단순한 정수 0,1,2,...를 씀)
lines.append('')                                              # 줄5
lines.append('')                                              # 줄6
lines.append('%%%%%%%%%% ~Freq. trace GHz')                   # 줄7
lines.append('')                                              # 줄8
lines.append('')                                              # 줄9
lines.append('\t'.join(str(10.0 + 0.1 * i) for i in range(aa_freq_size)))
    # 줄10: 주파수값 10.0, 10.1, 10.2, ...
lines.append('')                                              # 줄11
lines.append('')                                              # 줄12
lines.append('%%%%%%%%%% I')                                  # 줄13
lines.append('')                                              # 줄14
lines.append('')                                              # 줄15

I_true = rng.standard_normal((aa_power_size, aa_freq_size))
    # rng.standard_normal(shape): 평균0, 표준편차1인 정규분포에서
    # (aa_power_size, aa_freq_size) 크기의 난수 배열을 뽑음.
    # 이 값 자체는 의미 없는 "테스트용 숫자"일 뿐이며, 나중에
    # 두 파서가 이 값을 "똑같이" 읽어내는지가 검증의 핵심.
for row in I_true:
    lines.append('\t'.join(str(v) for v in row))   # 줄16, 18, 20, ...
    lines.append('')                                # 줄17, 19, 21, ...

# I 섹션의 마지막 데이터 줄 다음에, Q 섹션 헤더가 나오기까지 정확히
# "빈줄,빈줄,헤더,빈줄,빈줄" 패턴(다른 섹션 전환부와 동일한 패턴)이
# 오도록 맞춤. 이 부분의 줄 개수가 하나라도 어긋나면 원저자의 고정
# 인덱스 방식(아래 STEP 2)이 엉뚱한 줄을 읽어 에러가 남 - 실제로
# 이 스크립트를 처음 만들 때 이 지점에서 빈 줄 개수를 잘못 맞춰
# "Empty input file" 에러가 났던 적이 있음 (원저자 방식이 얼마나
# "정확한 줄 번호"에 의존하는 취약한 구조인지 보여주는 실제 사례).
lines.append('')                          # 헤더 앞 빈 줄 1
lines.append('%%%%%%%%%% Q')              # Q 섹션 헤더
lines.append('')                          # 헤더 뒤 빈 줄 1
lines.append('')                          # 헤더 뒤 빈 줄 2

Q_true = rng.standard_normal((aa_power_size, aa_freq_size))
for row in Q_true:
    lines.append('\t'.join(str(v) for v in row))
    lines.append('')

full_text = '\n'.join(lines)
with open(aa_temp_file_path, 'w') as f:
    f.write(full_text)

print(f"가짜 검증 파일 생성 완료: {aa_temp_file_path}")
print(f"  (전력 {aa_power_size}개 x 주파수 {aa_freq_size}개 크기로 축소된 테스트용 데이터)")


# =========================================================
# STEP 2. [방식 A] 원저자의 고정 인덱스 파싱 방식을 그대로 재현
# =========================================================
# 아래 코드는 원저자의 PBB_paper_figures_Figure1.py에 있던 로직을
# 최대한 그대로 옮긴 것 (변수 이름과 계산식까지 동일하게 유지해서,
# "다른 방식으로 바꿔치기 한 게 아니라 정말 같은 로직"임을 보여줌).

data = full_text.split('\n')
    # 원저자는 open(...).readlines() 후 각 줄 끝의 '\n'을 잘라내는
    # 방식을 썼는데, 우리는 이미 메모리에 있는 문자열을 split('\n')
    # 하는 것으로 동일한 효과를 냄 (파일을 다시 열지 않아도 됨).

power_author = np.genfromtxt(StringIO(data[4]), delimiter='\t')
    # np.genfromtxt: 텍스트에서 숫자 배열을 읽어오는 numpy 함수.
    # StringIO: 문자열을 "파일인 것처럼" 다룰 수 있게 감싸는 도구
    # (원래 genfromtxt는 파일 경로나 파일 객체를 기대하므로, 이미
    #  메모리에 있는 문자열 한 줄을 파일처럼 속여서 넣어주는 트릭).
freq_author = np.genfromtxt(StringIO(data[10]), delimiter='\t')

power_size_A = np.size(power_author)   # np.size: 배열의 원소 개수
freq_size_A = np.size(freq_author)

# I 데이터: 16번째 줄부터 2줄 간격으로 power_size_A개를 순서대로 읽어
# 1차원으로 쭉 이어붙인 뒤, 마지막에 2차원 (전력개수 x 주파수개수)로 변형
I_old = np.genfromtxt(StringIO(data[16]), delimiter='\t')
for i in range(power_size_A - 1):
    I1 = np.genfromtxt(StringIO(data[16 + (i + 1) * 2]), delimiter='\t')
    I_old = np.append(I_old, I1)
        # np.append: 두 배열을 이어붙이는 함수 (매번 새 배열을 만들어
        # 반환하므로, 반복문 안에서 쓰면 데이터가 많을 때 느릴 수 있음 -
        # 원저자 코드 그대로를 재현하는 게 목적이라 그대로 사용)
I21_author = I_old.reshape([power_size_A, freq_size_A])
    # .reshape([행,열]): 1차원으로 쭉 이어진 배열을 원하는 2차원
    # 형태로 다시 접는 함수. 순서대로 잘라 넣으므로, 원래 데이터가
    # "전력0의 전체 주파수, 전력1의 전체 주파수, ..." 순서로 쌓여
    # 있어야 올바른 모양이 나옴 (I_old를 만든 순서가 정확히 이 순서).

# Q 데이터: I 섹션이 끝난 뒤 "빈줄,빈줄,헤더,빈줄,빈줄"만큼 더 간
# 지점(init)부터 시작. 이 +4라는 숫자 자체가 "빈 줄이 정확히 몇 개
# 있는지"에 대한 원저자의 암묵적 가정이며, 파일 구조가 조금이라도
# 다르면 이 숫자부터 다시 계산해야 함 (마커 기반 방식에는 없는 취약점).
init = 16 + power_size_A * 2 + 4
Q_old = np.genfromtxt(StringIO(data[init]), delimiter='\t')
for i in range(power_size_A - 1):
    Q1 = np.genfromtxt(StringIO(data[init + (i + 1) * 2]), delimiter='\t')
    Q_old = np.append(Q_old, Q1)
Q21_author = Q_old.reshape([power_size_A, freq_size_A])

print("\n[방식 A: 원저자 고정 인덱스] 파싱 완료")
print(f"  power shape={power_author.shape}, freq shape={freq_author.shape}")
print(f"  I shape={I21_author.shape}, Q shape={Q21_author.shape}")


# =========================================================
# STEP 3. [방식 B] 우리 마커 기반 파서 (zenodo_loader.py)
# =========================================================
result_ours = zenodo_loader.load_power_sweep_txt(aa_temp_file_path)

print("\n[방식 B: 마커 기반 파서] 파싱 완료")
print(f"  power shape={result_ours['power_dbm'].shape}, "
      f"freq shape={result_ours['freq_ghz'].shape}")
print(f"  I shape={result_ours['I'].shape}, Q shape={result_ours['Q'].shape}")


# =========================================================
# STEP 4. 두 방식의 결과 비교 (교차검증)
# =========================================================
print("\n" + "=" * 55)
print("두 방식의 결과가 완전히 일치하는지 비교")
print("=" * 55)

checks = {
    '전력값(power)': np.allclose(power_author, result_ours['power_dbm']),
        # np.allclose(a, b): 두 배열의 모든 원소가 (부동소수점 오차
        # 범위 안에서) 사실상 같은지 확인하는 함수. ==으로 직접 비교하면
        # 아주 작은 계산 오차 때문에 False가 나올 수 있어, 수치 비교에는
        # allclose를 쓰는 것이 표준적인 관례.
    '주파수값(freq)': np.allclose(freq_author, result_ours['freq_ghz']),
    'I 배열 shape':  I21_author.shape == result_ours['I'].shape,
    'I 배열 값':      np.allclose(I21_author, result_ours['I']),
    'Q 배열 shape':  Q21_author.shape == result_ours['Q'].shape,
    'Q 배열 값':      np.allclose(Q21_author, result_ours['Q']),
}

all_passed = True
for name, passed in checks.items():
    status = "✅ 일치" if passed else "❌ 불일치!"
    print(f"  {name:15s}: {status}")
    all_passed = all_passed and passed

print("=" * 55)
if all_passed:
    print("결론: 두 파싱 방식이 완전히 동일한 결과를 냅니다.")
    print("      -> zenodo_loader.py를 실제 파일에 안심하고 사용해도 됩니다.")
else:
    print("결론: 두 방식이 어긋난 부분이 있습니다. 위에서 '불일치'로")
    print("      표시된 항목을 확인하고, 가짜 파일의 구조(STEP 1)나")
    print("      zenodo_loader.py의 파싱 로직을 다시 점검하세요.")

# 검증용 임시 파일 정리 (원하면 이 줄을 주석 처리해서 파일을 남겨둘 수 있음)
os.remove(aa_temp_file_path)
