"""
Noisy Realistic Mock Data Generator
=================================================================
기존 mock 데이터(qubit_2d_flux_sweep_mock.h5)는 순수 가우시안 백색잡음만
섞여 있어서, 실제 실험 데이터보다 훨씬 "깨끗"합니다.

이 스크립트는 실제 냉동기/VNA 측정에서 흔히 나타나는 세 가지 지저분함을
일부러 추가한 버전을 만듭니다:

  1. 1/f 잡음 (플리커 노이즈, flicker noise)
     - 저주파(천천히 변하는) 성분이 강하고 고주파(빠르게 변하는) 성분이 약한 잡음.
     - 전자회로, 특히 HEMT 증폭기에서 실제로 흔히 나타나는 잡음 특성.
     - 백색잡음(모든 주파수에서 세기가 동일)과 대비되는 개념.

  2. 캘리브레이션 드리프트 (calibration drift)
     - 시간이 지나며(여기서는 flux slice가 진행되며) 배경 진폭/위상이
       서서히 변하는 현상. 실제로는 온도 변화, 증폭기 이득 변화 등이 원인.
     - flux를 스캔하는 동안 냉동기 온도가 미세하게 오르내리는 것과 유사한 효과.

  3. Outlier (이상치, 튀는 값)
     - 우주선(cosmic ray) 충돌, 준입자 파열(quasiparticle burst) 등으로
       가끔 정상 범위를 크게 벗어나는 튀는 측정값이 섞이는 현상.
     - 통계적으로는 "긴 꼬리(heavy tail)"를 가진 분포로 모델링됨.

이 데이터로 이전에 짠 MCMC 파이프라인(anticrossing_full_flux_sweep.py)을
그대로 돌려보면, 깨끗한 데이터에서는 안 보이던 문제들이 나타날 것입니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

# =========================================================
# STEP 0. 분석자 설정값 (aa_ 접두사)
# =========================================================
aa_drive_path = '/content/drive/MyDrive/eunjunglee/SuperQuantum/'
aa_h5_filename_noisy = 'qubit_2d_flux_sweep_mock_NOISY.h5'  # 기존 파일과 구분되는 새 이름

# --- 참값 (기존과 동일하게 유지: 비교를 위해) ---
aa_f_r0 = 5.000
aa_kappa = 0.030
aa_fq_max = 5.150
aa_EC = 0.250
aa_g = 0.040

# --- 측정 grid ---
aa_n_freq = 401
aa_n_flux = 101
aa_freq_range = (4.8, 5.2)
aa_flux_range = (-0.5, 0.5)

# --- 잡음 관련 설정 (오늘 새로 추가하는 "지저분함" 파라미터들) ---
aa_white_noise_level = 0.015
    # 기존과 동일한 기본 백색잡음 크기 (베이스라인)

aa_flicker_noise_level = 0.020
    # 1/f 잡음의 세기. 값이 클수록 저주파 흔들림이 더 두드러짐.
    # (백색잡음과 별도로 추가되는 성분)

aa_drift_amplitude = 0.08
    # 캘리브레이션 드리프트의 진폭(크기). flux slice가 진행되며
    # 배경 진폭이 최대 이 정도(± aa_drift_amplitude)까지 서서히 변함.
aa_drift_period_fraction = 0.6
    # 드리프트가 전체 101개 slice 중 몇 주기(cycle)를 도는지의 역수 개념.
    # 0.6이면 전체 스캔 동안 느린 sine 파형이 대략 0.6주기 정도 그려짐
    # (완전한 주기적 진동이 아니라 "서서히 한 방향으로 흘러갔다 돌아오는" 형태를 표현)

aa_outlier_probability = 0.01
    # 각 데이터 포인트가 outlier가 될 확률 (1% = 401*101개 포인트 중 약 400여개)
aa_outlier_scale = 0.4
    # outlier가 발생했을 때, 정상 잡음보다 얼마나 더 크게 튈지의 배수/크기

np.random.seed(123)   # 재현성을 위한 난수 시드 (기존 42와 다르게 설정해 구분)

os.makedirs(aa_drive_path, exist_ok=True)

print("=" * 60)
print("분석자 설정값(aa_*) 요약:")
for k, v in list(globals().items()):
    if k.startswith("aa_"):
        print(f"  {k:28s} = {v}")
print("=" * 60)

# =========================================================
# STEP 1. Grid 및 깨끗한 물리 신호 생성 (기존과 동일한 로직)
# =========================================================
f_grid = np.linspace(*aa_freq_range, aa_n_freq)
flux_grid = np.linspace(*aa_flux_range, aa_n_flux)

fq_flux = (aa_fq_max + aa_EC) * np.sqrt(np.abs(np.cos(np.pi * flux_grid))) - aa_EC

s21_2d_clean = np.zeros((aa_n_flux, aa_n_freq), dtype=complex)

for i, fq in enumerate(fq_flux):
    delta = fq - aa_f_r0
    hybrid_plus = 0.5 * (aa_f_r0 + fq + np.sqrt(delta**2 + 4 * aa_g**2))
    hybrid_minus = 0.5 * (aa_f_r0 + fq - np.sqrt(delta**2 + 4 * aa_g**2))

    sin2_theta = 0.5 * (1.0 - delta / np.sqrt(delta**2 + 4 * aa_g**2))
    cos2_theta = 1.0 - sin2_theta

    s21_mode1 = cos2_theta / (1.0 + 1j * (f_grid - hybrid_minus) / (aa_kappa / 2))
    s21_mode2 = sin2_theta / (1.0 + 1j * (f_grid - hybrid_plus) / (aa_kappa / 2))

    cable_delay = np.exp(-1j * 2 * np.pi * f_grid * 0.12)
    s21_2d_clean[i, :] = (1.0 - (s21_mode1 + s21_mode2)) * cable_delay

# =========================================================
# STEP 2. 세 가지 "지저분함" 순차적으로 추가
# =========================================================

# --- 2.1 기본 백색잡음 (기존과 동일) ---
noise_white_real = np.random.normal(0, aa_white_noise_level, s21_2d_clean.shape)
noise_white_imag = np.random.normal(0, aa_white_noise_level, s21_2d_clean.shape)

# --- 2.2 1/f 잡음 (flicker noise) 생성 ---
# 원리: 백색잡음을 FFT(고속푸리에변환)로 주파수 영역으로 옮긴 뒤,
# 각 주파수 성분의 크기를 1/sqrt(frequency)에 비례하도록 눌러주면
# (파워 스펙트럼이 1/f가 되도록), 저주파 성분이 강조된 "천천히 출렁이는" 잡음이 됨.
def generate_1f_noise(n_points, level):
    white = np.random.normal(0, 1, n_points)
    freqs = np.fft.rfftfreq(n_points)
        # np.fft.rfftfreq: 실수 신호의 FFT에 대응하는 주파수 축을 생성하는 함수
        # (rfft는 real FFT의 줄임말로, 실수 입력에 최적화된 푸리에변환)
    freqs[0] = freqs[1]   # 0번째(DC, 직류성분)는 나누기 에러 방지를 위해 근처 값으로 치환
    spectrum = np.fft.rfft(white) / np.sqrt(freqs)
        # 진폭을 1/sqrt(f)로 스케일링 -> 파워(진폭 제곱)로는 1/f가 됨
    colored = np.fft.irfft(spectrum, n=n_points)
        # irfft: 역푸리에변환으로 다시 시간(여기선 주파수 스캔축) 영역으로 되돌림
    colored = colored / np.std(colored) * level
        # 원하는 세기(level)에 맞게 정규화
    return colored

noise_flicker_real = np.array([generate_1f_noise(aa_n_freq, aa_flicker_noise_level)
                                for _ in range(aa_n_flux)])
noise_flicker_imag = np.array([generate_1f_noise(aa_n_freq, aa_flicker_noise_level)
                                for _ in range(aa_n_flux)])
    # 각 flux slice(행)마다 독립적으로 1/f 잡음을 새로 생성해서 쌓음
    # (리스트 컴프리헨션으로 101번 생성 후 np.array로 2D 배열로 결합)

# --- 2.3 캘리브레이션 드리프트 생성 ---
# flux slice 진행(=측정이 진행되는 "시간"에 해당)에 따라 배경 크기가
# 완만한 sine 곡선을 그리며 서서히 변한다고 가정.
slice_progress = np.arange(aa_n_flux) / aa_n_flux   # 0~1로 정규화된 "진행률"
drift_curve = aa_drift_amplitude * np.sin(2 * np.pi * aa_drift_period_fraction * slice_progress)
    # (n_flux,) 형태의 1D 배열: 각 slice마다 하나의 드리프트 값
drift_2d = drift_curve[:, np.newaxis]
    # np.newaxis: 배열에 새로운 축(차원)을 추가하는 인덱싱 트릭.
    # (101,) -> (101, 1)로 바꿔서, 아래에서 (101, 401) 배열과
    # 브로드캐스팅(broadcasting, 크기가 다른 배열끼리 자동으로 맞춰 연산)이 되게 함

# --- 2.4 Outlier 생성 ---
outlier_mask = np.random.random(s21_2d_clean.shape) < aa_outlier_probability
    # 각 포인트마다 0~1 균일분포 난수를 뽑아, 설정한 확률보다 작으면 outlier로 지정
n_outliers = np.sum(outlier_mask)
outlier_values_real = np.random.normal(0, aa_outlier_scale, s21_2d_clean.shape)
outlier_values_imag = np.random.normal(0, aa_outlier_scale, s21_2d_clean.shape)

print(f"\n생성된 outlier 개수: {n_outliers} / {s21_2d_clean.size} "
      f"({100*n_outliers/s21_2d_clean.size:.2f}%)")

# =========================================================
# STEP 3. 모든 성분을 합쳐 최종 "지저분한" 데이터 생성
# =========================================================
s21_2d_noisy = (
    s21_2d_clean
    + (1 + drift_2d) * (noise_white_real + 1j * noise_white_imag)
        # 드리프트가 백색잡음 크기 자체에도 영향을 주도록 곱셈으로 결합
        # (실제로 증폭기 이득이 흔들리면 잡음 크기 자체도 같이 흔들리는 경우가 많음)
    + (noise_flicker_real + 1j * noise_flicker_imag)
    + drift_2d   # 신호 자체의 평균 레벨도 서서히 이동 (offset drift)
)

# outlier를 마지막에 덮어씌우듯 추가
s21_2d_noisy = np.where(outlier_mask, s21_2d_noisy + outlier_values_real + 1j * outlier_values_imag, s21_2d_noisy)
    # np.where(조건, A, B): 조건이 True인 위치는 A, False인 위치는 B 값을 쓰는 함수.
    # 여기선 outlier_mask가 True인 지점에만 추가로 큰 값을 더함.

i_data_2d = np.real(s21_2d_noisy)
q_data_2d = np.imag(s21_2d_noisy)
amp_data_2d = np.abs(s21_2d_noisy)

# =========================================================
# STEP 4. HDF5로 저장
# =========================================================
full_path = os.path.join(aa_drive_path, aa_h5_filename_noisy)

with h5py.File(full_path, "w") as f:
    meta = f.create_group("metadata")
    meta.attrs["f_r0_GHz"] = aa_f_r0
    meta.attrs["g_GHz"] = aa_g
    meta.attrs["kappa_GHz"] = aa_kappa
    meta.attrs["note"] = "1/f noise + calibration drift + outliers injected"

    data = f.create_group("data")
    data.create_dataset("frequency_GHz", data=f_grid)
    data.create_dataset("flux_Phi0", data=flux_grid)
    data.create_dataset("I_voltage", data=i_data_2d)
    data.create_dataset("Q_voltage", data=q_data_2d)
    data.create_dataset("S21_amplitude", data=amp_data_2d)

print(f"\n✅ 지저분한(realistic) mock 데이터 저장 완료: '{full_path}'")

# =========================================================
# STEP 5. 시각화 (깨끗한 버전과 나란히 비교)
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

im0 = axes[0].pcolormesh(flux_grid, f_grid, np.abs(s21_2d_clean).T, cmap="viridis", shading="auto")
axes[0].set_title("Clean Mock Data (white noise only)")
axes[0].set_xlabel(r"Flux $\Phi/\Phi_0$")
axes[0].set_ylabel("Frequency (GHz)")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].pcolormesh(flux_grid, f_grid, amp_data_2d.T, cmap="viridis", shading="auto")
axes[1].set_title("Noisy Mock Data (1/f + drift + outliers)")
axes[1].set_xlabel(r"Flux $\Phi/\Phi_0$")
axes[1].set_ylabel("Frequency (GHz)")
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.savefig(os.path.join(aa_drive_path, 'outputs', 'clean_vs_noisy_comparison.png'),
            dpi=150, bbox_inches='tight')
plt.show()

# 단일 slice(Φ=0)에서 magnitude 비교 (깨끗한 것 vs 지저분한 것)
mid_idx = aa_n_flux // 2
fig2, ax2 = plt.subplots(figsize=(9, 4))
ax2.plot(f_grid, np.abs(s21_2d_clean[mid_idx]), label='Clean', alpha=0.7)
ax2.plot(f_grid, amp_data_2d[mid_idx], label='Noisy (1/f+drift+outlier)', alpha=0.7)
ax2.set_xlabel("Frequency (GHz)")
ax2.set_ylabel("|S21|")
ax2.set_title(f"Φ=0 slice: Clean vs Noisy 비교")
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(aa_drive_path, 'outputs', 'clean_vs_noisy_single_slice.png'),
            dpi=150, bbox_inches='tight')
plt.show()

print("\n시각화 저장 완료.")
