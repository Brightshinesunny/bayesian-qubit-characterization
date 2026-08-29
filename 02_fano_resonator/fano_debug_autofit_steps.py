"""
fano_debug_autofit_steps.py
=================================================================
[문제 상황 정리] fano_check_outliers.py로 확인한 결과, 이전에
"이상치"로 의심했던 지점들은 실제로는 측정 잡음이 아니라 "딥이 아주
좁고 가파른" 정상적인 물리 신호였습니다(사용자가 직접 그래프를 보고
확인). 즉 데이터 자체는 정상인데 circle fit 결과(fr, Ql 등)가
크게 틀렸으므로, 문제는 데이터가 아니라 autofit() 파이프라인 내부의
"어느 계산 단계"에 있을 가능성이 높습니다.

이 스크립트는 fano_models.autofit()이 내부적으로 거치는 단계
(delay 처리 -> circle fit -> 원점 이동 -> fit_phase -> Qc/Qi 추출)를
한 번에 실행하지 않고, 한 단계씩 손으로 재현하며 각 단계의 중간
결과를 전부 출력합니다. 이렇게 하면 "몇 번째 계단에서 결과가
갑자기 이상해지는지"를 정확히 짚어낼 수 있습니다 - 이건 코드
디버깅의 표준적인 방법론("이분 탐색"과 비슷한 원리: 전체를 한 번에
보지 않고 중간 지점들을 순서대로 찍어보면서 문제 구간을 좁혀나감).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.getcwd())
import fano_loader
import fano_models


# =========================================================
# STEP 0. 분석자 설정값 (aa_ 접두사)
# =========================================================
aa_drive_folder = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module01/Fano/overcoupled/'
aa_filename = 'resonator_1_powersweep_overcoupled.npz'
aa_output_dir = './outputs/'
os.makedirs(aa_output_dir, exist_ok=True)
aa_slice_index = -1   # 문제가 재현됐던 그 slice(최고 전력) 그대로 사용

print("=" * 60)
print("분석자 설정값(aa_*) 요약:")
for k, v in list(globals().items()):
    if k.startswith("aa_"):
        print(f"  {k:24s} = {v}")
print("=" * 60)


# =========================================================
# STEP 1. 데이터 로드
# =========================================================
full_path = os.path.join(aa_drive_folder, aa_filename)
data = fano_loader.load_fano_npz(full_path)
f_data_hz = data['freq_hz']
s21_slice = data['s21'][aa_slice_index, :]
cable_delay = data['cable_delay']
print(f"\ncable_delay = {cable_delay} 초")


# =========================================================
# STEP 2. [1단계] delay 보정 적용 - autofit의 STEP B 시작 부분 재현
# =========================================================
# autofit 안에서: z_data = z_data_raw * exp(2j*pi*delay*f_data)
# 이렇게 delay로 인한 위상 회전을 미리 제거한 뒤에 circle fit을 함.
delay = cable_delay
z_delay_corrected = s21_slice * np.exp(2j * np.pi * delay * f_data_hz)

print("\n[1단계] delay 보정 후 데이터 (일부만 출력)")
print(f"  보정 전 첫 5개 위상: {np.round(np.angle(s21_slice[:5]), 4)}")
print(f"  보정 후 첫 5개 위상: {np.round(np.angle(z_delay_corrected[:5]), 4)}")


# =========================================================
# STEP 3. [2단계] circle fit - 원의 중심/반지름 계산
# =========================================================
xc, yc, r0 = fano_models.fit_circle_algebraic(z_delay_corrected)
print(f"\n[2단계] circle fit 결과")
print(f"  중심 (xc, yc) = ({xc:.6f}, {yc:.6f})")
print(f"  반지름 r0 = {r0:.6f}")
print(f"  -> 참고: 데이터 자체의 |S21| 범위가 [{np.abs(s21_slice).min():.5f}, "
      f"{np.abs(s21_slice).max():.5f}]이므로, r0가 이 스케일과 비슷한")
print(f"     크기(대략 0.005~0.02 수준)인지 확인 - 만약 r0가 훨씬 크거나")
print(f"     작다면 circle fit 자체가 이미 잘못됐다는 신호.")


# =========================================================
# STEP 4. [3단계] 원점으로 이동 후 fit_phase - fr, Ql 초기/최종 추정
# =========================================================
zc = complex(xc, yc)
z_centered = z_delay_corrected - zc

fr, Ql, theta, delay_remaining = fano_models.fit_phase(f_data_hz, z_centered)
print(f"\n[3단계] fit_phase 결과 (원점 이동 후 위상 피팅)")
print(f"  fr = {fr/1e9:.6f} GHz")
print(f"  Ql = {Ql:.2f}")
print(f"  theta = {theta:.4f} rad")
print(f"  delay_remaining = {delay_remaining:.4e} 초")
print(f"  -> 참고: fr이 데이터 스캔 범위[{f_data_hz.min()/1e9:.4f}, "
      f"{f_data_hz.max()/1e9:.4f}] GHz 안에 있는지,")
print(f"     그리고 육안으로 봤던 딥 위치(약 4.9956 GHz)와 가까운지 확인.")


# =========================================================
# STEP 5. [3-1단계] fit_phase 내부의 "초기 추정값"까지 더 파고들기
# =========================================================
# fit_phase()의 최종 결과가 이상하다면, 처음 출발점(초기 추정값)부터
# 이미 잘못됐을 가능성이 높음. fit_phase 내부 로직을 손으로 재현해서
# 초기 추정값(fr_guess, Ql_guess)까지 직접 확인.
phase = np.unwrap(np.angle(z_centered))
    # np.unwrap: 위상이 -pi에서 +pi로 "감기며" 생기는 인위적인 불연속
    # 점프를 제거하고, 실제 물리적으로 연속인 위상 곡선으로 펴주는 함수.

roll_off_full = phase.max() - phase.min()
roll_off = roll_off_full if roll_off_full <= 0.8*2*np.pi else 2*np.pi
print(f"\n[3-1단계] fit_phase 내부 진단")
print(f"  unwrap된 phase 전체 변화폭: {roll_off_full:.4f} rad "
      f"(2pi={2*np.pi:.4f})")
print(f"  -> 이 값이 2pi(완전히 한 바퀴)에 훨씬 못 미치면(예: 0.8*2pi=",
      f"{0.8*2*np.pi:.4f} 미만), circuit.py 원본이 '데이터가 원을 다")
print(f"     못 돌았다'는 경고를 내는 조건. 이 경우 fr_guess/Ql_guess")
print(f"     추정이 원래 부정확해지기 쉬움 - 원저자 코드도 이 상황을")
print(f"     '주파수 스캔 범위를 넓히라'는 경고로 명시적으로 안내함.")

pad_width = 5
phase_padded = np.pad(phase, pad_width, mode='edge')
kernel = np.ones(11) / 11
phase_smooth_padded = np.convolve(phase_padded, kernel, mode='same')
phase_smooth = phase_smooth_padded[pad_width:-pad_width]
phase_derivative = np.gradient(phase_smooth)
fr_guess_idx = np.argmax(np.abs(phase_derivative))
fr_guess = f_data_hz[fr_guess_idx]
Ql_guess = 2 * fr_guess / (f_data_hz[-1] - f_data_hz[0])

print(f"\n  fr_guess(초기 추정) = {fr_guess/1e9:.6f} GHz "
      f"(인덱스 {fr_guess_idx} / 전체 {len(f_data_hz)})")
print(f"  Ql_guess(초기 추정) = {Ql_guess:.2f}")
print(f"  -> fr_guess가 위 최종 fr({fr/1e9:.6f} GHz)과 비슷한 위치인지,")
print(f"     그리고 데이터의 실제 딥 위치와 가까운지가 핵심 확인 포인트.")


# =========================================================
# STEP 6. 시각화 - roll_off(위상 전체 회전폭)가 부족한지 직접 확인
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(f_data_hz/1e9, phase, '.-', ms=3)
axes[0].axhline(phase.max(), color='gray', ls='--', lw=1)
axes[0].axhline(phase.min(), color='gray', ls='--', lw=1)
axes[0].set_xlabel('Frequency (GHz)')
axes[0].set_ylabel('unwrapped phase (rad, 원점 이동 후)')
axes[0].set_title(f'위상 변화 (roll_off={roll_off_full:.3f} rad, '
                   f'2pi={2*np.pi:.3f})')

axes[1].plot(np.real(z_centered), np.imag(z_centered), '.-', ms=3)
axes[1].plot(0, 0, 'r+', ms=15, mew=2, label='원점(circle fit 중심)')
axes[1].set_xlabel('Re (원점 이동 후)')
axes[1].set_ylabel('Im (원점 이동 후)')
axes[1].set_aspect('equal')
axes[1].set_title('원점으로 이동된 궤적')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(aa_output_dir, 'fano_debug_steps.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\n디버깅 시각화 저장 완료: {os.path.join(aa_output_dir, 'fano_debug_steps.png')}")
