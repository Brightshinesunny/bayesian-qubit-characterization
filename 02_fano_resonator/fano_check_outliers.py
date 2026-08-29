"""
fano_check_outliers.py
=================================================================
[문제 상황] fano_fit_real_data.py로 최고 전력(-10dBm) slice에
circle fit을 돌렸더니, 결과(fr, Ql 등)가 실측 데이터와 전혀 다른
곳을 가리켰습니다. 진폭 그래프를 보면 공진 딥 근처(4.9955~4.9958
GHz)에 유독 듬성듬성하고 튀는 점들이 보였는데, 이게 정말 이상치
(outlier)인지, 아니면 그냥 그래프가 작아서 그렇게 보인 것뿐인지
확대해서 직접 확인하는 스크립트입니다.

[왜 이상치가 circle fit을 이렇게 크게 망가뜨리는가 - 복습]
fit_circle_algebraic()은 "최소제곱법"을 씁니다. 최소제곱법은 모든
데이터 점의 오차를 "제곱해서" 합산하는데, 제곱을 취하면 오차가 큰
점(=이상치)의 영향이 더 크게(기하급수적으로) 부풀려집니다. 예를 들어
오차가 2배 큰 점은, 단순히 2배가 아니라 4배 더 큰 비중으로 결과에
영향을 줍니다. 그래서 이상치 몇 개만 섞여 있어도 원의 중심/반지름
추정이 크게 틀어질 수 있습니다 - 이게 첫날 배운 "Gaussian 우도가
이상치에 취약하다"는 것과 정확히 같은 원리입니다(다만 circle fit은
MCMC의 robust 우도 같은 "덜 믿는" 대안이 구조적으로 없다는 게 다름).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.getcwd())
import fano_loader


# =========================================================
# STEP 0. 분석자 설정값 (aa_ 접두사)
# =========================================================
aa_drive_folder = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module01/Fano/overcoupled/'
aa_filename = 'resonator_1_powersweep_overcoupled.npz'
aa_output_dir = './outputs/'
os.makedirs(aa_output_dir, exist_ok=True)

aa_slice_index = -1
    # 문제가 생겼던 바로 그 slice(최고 전력, -10dBm)를 그대로 지정해서
    # 재현 확인.

aa_zoom_freq_range = (4.9950, 4.9962)
    # [분석자가 지정] 이전 그래프에서 이상치로 의심됐던 구간
    # (4.9955~4.9958 GHz)을 여유 있게 포함하도록 확대할 주파수 범위.
    # 이 값을 조정하면서 "정확히 어디까지가 이상한 구간인지" 좁혀갈 수 있음.

print("=" * 60)
print("분석자 설정값(aa_*) 요약:")
for k, v in list(globals().items()):
    if k.startswith("aa_"):
        print(f"  {k:24s} = {v}")
print("=" * 60)


# =========================================================
# STEP 1. 데이터 로드 및 대상 slice 추출
# =========================================================
full_path = os.path.join(aa_drive_folder, aa_filename)
data = fano_loader.load_fano_npz(full_path)

freq_ghz = data['freq_ghz']
s21_slice = data['s21'][aa_slice_index, :]
amp_slice = data['amplitude'][aa_slice_index, :]
power_val = data['power_dbm'][aa_slice_index]

print(f"\n확인 대상: power={power_val:.1f} dBm (index={aa_slice_index})")


# =========================================================
# STEP 2. 확대 구간만 잘라내기 (boolean indexing)
# =========================================================
zoom_mask = (freq_ghz >= aa_zoom_freq_range[0]) & (freq_ghz <= aa_zoom_freq_range[1])
    # boolean indexing(불리언 인덱싱): freq_ghz 배열의 각 원소가 조건을
    # 만족하는지(True/False)를 담은 같은 길이의 배열(zoom_mask)을 만든
    # 뒤, 이 mask를 인덱스처럼 사용해 조건을 만족하는 원소들만 쏙
    # 뽑아내는 numpy의 표준적인 필터링 방법. "&"는 두 조건을 동시에
    # 만족해야 한다는 뜻(파이썬 논리연산자 and의 배열 버전).

freq_zoom = freq_ghz[zoom_mask]
amp_zoom = amp_slice[zoom_mask]
s21_zoom = s21_slice[zoom_mask]

n_points_in_zoom = np.sum(zoom_mask)
    # np.sum(불리언 배열): True를 1, False를 0으로 세어 합산하므로,
    # "조건을 만족하는 원소 개수"를 세는 관용적인 방법.
print(f"확대 구간 안의 데이터 포인트 수: {n_points_in_zoom}개")


# =========================================================
# STEP 3. 이상치 후보를 통계적으로 자동 탐지
# =========================================================
# "눈으로 보기에 튄다"는 느낌을 숫자로 확인하기 위해, 이 구간 안에서
# 이웃한 점들끼리 값 차이가 얼마나 큰지를 계산해봄. 정상적인(매끈한)
# 곡선이라면 이웃한 점끼리 값이 크게 안 다를 텐데, 이상치가 있다면
# 그 이상치 전후로 값이 갑자기 크게 뛸 것.
diffs = np.abs(np.diff(amp_zoom))
    # np.diff: 배열에서 "바로 이웃한 두 원소끼리의 차이"를 계산해
    # (길이가 원래보다 1 짧은) 새 배열로 반환하는 함수. 예를 들어
    # [1,2,5,6]의 diff는 [1,3,1] (5-2=3처럼 급격한 변화가 여기서 두드러짐).

median_diff = np.median(diffs)
    # np.median: 중앙값(데이터를 크기순으로 줄 세웠을 때 정가운데 값).
    # 평균(mean) 대신 중앙값을 쓰는 이유: 평균은 이상치 자체에 의해
    # 쉽게 영향받지만(이상치가 평균을 확 끌어올리거나 내림), 중앙값은
    # "대부분의 정상적인 이웃 간 차이가 어느 정도인지"를 이상치에
    # 흔들리지 않고 안정적으로 알려줌 - 그래서 "이상치를 찾는 기준선"
    # 으로 중앙값을 쓰는 것이 통계학에서 표준적인 관행.

threshold = median_diff * 5
    # 이웃 간 차이가 "전형적인 차이(중앙값)의 5배"를 넘으면 이상치로
    # 간주. 5라는 배수는 엄밀한 통계 기준이 아니라 실전에서 자주 쓰는
    # 경험적(heuristic) 기준값 - 더 엄격하게 하려면 배수를 낮추고,
    # 너무 많은 정상 점을 이상치로 오판하면 배수를 높이면 됨.

outlier_indices = np.where(diffs > threshold)[0]
    # np.where(조건)[0]: 조건을 만족하는 위치(인덱스)들을 배열로 반환.
    # diffs가 amp_zoom보다 1 짧으므로, 이 인덱스는 "그 지점과 다음
    # 지점 사이에서 급격한 변화가 있었다"는 뜻.

print(f"\n이웃 간 진폭 차이의 중앙값: {median_diff:.6f}")
print(f"이상치 판정 기준(중앙값의 5배): {threshold:.6f}")
print(f"이상치로 의심되는 지점 개수: {len(outlier_indices)}개")
if len(outlier_indices) > 0:
    print(f"이상치 의심 위치의 주파수: "
          f"{np.round(freq_zoom[outlier_indices], 5)} GHz")
    print(f"해당 지점들의 진폭값: {np.round(amp_zoom[outlier_indices], 5)}")


# =========================================================
# STEP 4. 시각화 - 확대해서 눈으로 직접 확인
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(freq_zoom, amp_zoom, 'o-', ms=4, alpha=0.7, label='진폭 데이터')
if len(outlier_indices) > 0:
    axes[0].plot(freq_zoom[outlier_indices], amp_zoom[outlier_indices],
                 'rx', ms=12, mew=2, label='이상치 의심 지점')
axes[0].set_xlabel('Frequency (GHz)')
axes[0].set_ylabel('|S21|')
axes[0].set_title(f'확대된 진폭 (power={power_val:.1f} dBm)')
axes[0].legend(fontsize=9)

axes[1].plot(np.real(s21_zoom), np.imag(s21_zoom), 'o-', ms=4, alpha=0.7,
             label='복소평면 궤적')
if len(outlier_indices) > 0:
    axes[1].plot(np.real(s21_zoom[outlier_indices]), np.imag(s21_zoom[outlier_indices]),
                 'rx', ms=12, mew=2, label='이상치 의심 지점')
axes[1].set_xlabel('Re(S21)')
axes[1].set_ylabel('Im(S21)')
axes[1].set_aspect('equal')
axes[1].set_title('확대된 복소평면 궤적')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(aa_output_dir, 'fano_outlier_check.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\n확대 시각화 저장 완료: {os.path.join(aa_output_dir, 'fano_outlier_check.png')}")
print("\n[다음 판단 기준]")
print("  - 점들이 매끈하게 이어진다면: 이상치가 아니라 실제로 딥 모양이")
print("    복잡한 것(예: 근처에 다른 공진 모드나 TLS가 있을 가능성)")
print("  - 몇몇 점이 정말 뚝뚝 끊겨 튀어 보인다면: 진짜 측정 이상치")
print("    (VNA 트리거 오류, 순간적 잡음 등) - 이 경우 전처리로 제거하거나")
print("    다른 전력 slice로 circle fit을 재시도하는 것이 좋음")
