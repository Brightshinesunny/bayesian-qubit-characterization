"""
stress_test_mock_scenarios.py
=================================================================
지난번 models.py / mock_data.py에 추가한 "교과서 모델과 실측의 괴리"
4가지를 실제로 켜고 끄면서, 지금까지 만든 MCMC 파이프라인이 각
시나리오에서 얼마나 잘 버티는지(또는 어디서 깨지는지) 확인하는
스트레스 테스트 스크립트입니다.

4가지 시나리오:
  1. Fano 비대칭 lineshape (models.fano_lineshape_correction)
  2. TLS로 인한 가짜 추가 딥 (models.add_spurious_tls_dip)
     -> 가장 위험한 시나리오. 초기값 탐색 로직(딥 2개 찾기)이
        TLS 딥을 avoided-crossing 봉우리로 착각할 수 있음.
  3. Baseline 기울어짐 (mock_data.add_baseline_tilt)
     -> noise_sigma 추정이 왜곡될 수 있음 (adaptive 방식이 baseline이
        평평하다고 가정하기 때문).
  4. κ의 시간적 요동 (mock_data.generate_fluctuating_kappa)
     -> flux slice마다 "진짜 κ"가 달라지므로, warm-start 가정
        ("인접 slice는 파라미터가 비슷하다")이 흔들릴 수 있음.

각 시나리오를 개별적으로, 그리고 다 합쳐서 한 번에 테스트합니다.
빠른 확인을 위해 101개 slice 전체가 아니라 "avoided-crossing이 가장
뚜렷한 대표 slice(Φ=0) 하나"에 대해서만 MCMC를 돌립니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import models
import mock_data
import likelihood
import mcmc_pipeline
import diagnostics

# =========================================================
# STEP 0. 분석자 설정값 - 4가지 시나리오를 개별적으로 켜고 끄는 스위치
# =========================================================
aa_output_dir = './outputs/'
os.makedirs(aa_output_dir, exist_ok=True)

aa_f_r0, aa_g, aa_kappa = 5.000, 0.040, 0.030
aa_fq_max, aa_EC, aa_tau = 5.150, 0.250, 0.12
aa_n_freq = 401
aa_freq_range = (4.8, 5.2)
aa_case_flux_val = 0.0   # 대표로 볼 flux 지점 (avoided-crossing이 가장 뚜렷한 Φ=0)

aa_white_noise_level = 0.015   # 기본 백색잡음은 항상 켜둠 (완전히 잡음 없는 건 비현실적)

# --- 시나리오 스위치 (True/False로 하나씩 켜보면서 실험) ---
aa_enable_fano = False
aa_fano_q = 2.0   # Fano 비대칭 계수 (작을수록 더 비대칭)

aa_enable_tls = False
aa_tls_offset_from_fr = 0.03   # TLS 주파수를 fr에서 얼마나 떨어뜨릴지 (GHz)
                                 # -> 작을수록 진짜 avoided-crossing 봉우리와
                                 #    더 가까워져서 파이프라인이 헷갈리기 쉬움
aa_tls_coupling = 0.4          # TLS 딥의 깊이
aa_tls_linewidth = 0.006       # TLS 딥의 폭 (보통 진짜 결합보다 훨씬 좁음)

aa_enable_tilt = False
aa_tilt_slope = 0.15

aa_enable_all = False   # True로 두면 위 4개(백색잡음 제외) 시나리오를 전부 동시에 켬
                          # -> 실전에서 여러 비이상적 요소가 동시에 섞인 "진짜 지저분한" 상황

# --- 초기값 탐색/MCMC 설정 (기존과 동일한 값 재사용) ---
aa_smoothing_window = 5
aa_nwalkers = 32
aa_nsteps = 2000
aa_burn_in_discard = 200
aa_thin_by = 10
aa_init_scatter = 5e-3
aa_prior_bounds = {'f_r': (4.8, 5.2), 'g': (0.001, 0.1), 'kappa': (0.001, 0.1)}

if aa_enable_all:
    aa_enable_fano = aa_enable_tls = aa_enable_tilt = True

print("=" * 60)
print("시나리오 스위치 상태:")
for name in ['aa_enable_fano', 'aa_enable_tls', 'aa_enable_tilt', 'aa_enable_all']:
    print(f"  {name:20s} = {globals()[name]}")
print("=" * 60)

# =========================================================
# STEP 1. Clean 신호 생성 (기본 avoided-crossing 모델)
# =========================================================
f_grid = np.linspace(*aa_freq_range, aa_n_freq)

s21 = models.s21_anticrossing_model(
    f_grid, aa_case_flux_val, aa_f_r0, aa_g, aa_kappa, aa_fq_max, aa_EC, aa_tau
)

# =========================================================
# STEP 2. 시나리오별 왜곡 적용 (켜진 것만 순차적으로 반영)
# =========================================================
applied_effects = []

if aa_enable_fano:
    s21 = models.fano_lineshape_correction(s21, f_grid, aa_f_r0, aa_fano_q)
    applied_effects.append(f"Fano(q={aa_fano_q})")

if aa_enable_tls:
    f_tls = aa_f_r0 + aa_tls_offset_from_fr
    s21 = models.add_spurious_tls_dip(s21, f_grid, f_tls, aa_tls_coupling, aa_tls_linewidth)
    applied_effects.append(f"TLS(offset={aa_tls_offset_from_fr}, f_tls={f_tls:.4f})")

if aa_enable_tilt:
    s21_2d_temp = s21[np.newaxis, :]   # add_baseline_tilt는 2D를 기대하므로 임시로 차원 추가
    s21_2d_temp = mock_data.add_baseline_tilt(s21_2d_temp, f_grid, aa_tilt_slope)
    s21 = s21_2d_temp[0, :]
    applied_effects.append(f"Tilt(slope={aa_tilt_slope})")

# 마지막으로 백색잡음 추가 (항상 켜둠)
s21_noisy = mock_data.add_white_noise(s21, aa_white_noise_level, rng=np.random.default_rng(7))

print(f"\n적용된 시나리오: {applied_effects if applied_effects else ['(기본 백색잡음만)']}")

# =========================================================
# STEP 3. 초기값 추정 - "딥 2개 찾기" 로직이 얼마나 흔들리는지 직접 확인
# =========================================================
def initial_guess_from_dip_search(f_grid_local, s21_1d, smoothing_window):
    mag = np.abs(s21_1d)
    mag_smooth = diagnostics.moving_average(mag, smoothing_window)
    dip_candidates = [i for i in range(2, len(mag_smooth) - 2)
                       if mag_smooth[i] < mag_smooth[i-1] and mag_smooth[i] < mag_smooth[i+1]]
    # [진단용 추가] 찾아낸 모든 딥의 위치와 깊이를 출력해서, TLS 딥이
    # "진짜 avoided-crossing 봉우리"로 착각되고 있는지 눈으로 확인 가능하게 함.
    dip_info = [(f_grid_local[i], mag_smooth[i]) for i in dip_candidates]
    dip_info.sort(key=lambda x: x[1])   # 깊은 순으로 정렬
    print(f"  탐지된 딥 후보 (주파수, 깊이) - 깊은 순 상위 5개: "
          f"{[(round(f,4), round(d,4)) for f,d in dip_info[:5]]}")

    if len(dip_candidates) >= 2:
        dip_candidates.sort(key=lambda i: mag_smooth[i])
        two = sorted(dip_candidates[:2])
        fr_g = 0.5 * (f_grid_local[two[0]] + f_grid_local[two[1]])
        g_g = abs(f_grid_local[two[1]] - f_grid_local[two[0]]) / 2.0
        print(f"  -> 초기값으로 선택된 두 딥: {f_grid_local[two[0]]:.4f} GHz, "
              f"{f_grid_local[two[1]]:.4f} GHz  =>  fr_guess={fr_g:.4f}, g_guess={g_g:.4f}")
    else:
        fr_g, g_g = 5.0, 0.01
        print(f"  -> 딥이 2개 미만 발견되어 폴백 초기값 사용: fr={fr_g}, g={g_g}")
    return np.array([fr_g, g_g, 0.02])

initial_guess = initial_guess_from_dip_search(f_grid, s21_noisy, aa_smoothing_window)

# =========================================================
# STEP 4. MCMC 실행 (gaussian 우도, 기존 파이프라인 그대로)
# =========================================================
log_prior = likelihood.make_uniform_log_prior(aa_prior_bounds)
model_kwargs = {'flux_val': aa_case_flux_val, 'fq_max': aa_fq_max, 'EC': aa_EC, 'tau': aa_tau}
log_prob = likelihood.make_log_probability(
    models.s21_anticrossing_model, model_kwargs, log_prior, likelihood_type='gaussian'
)

noise_sigma_est = diagnostics.estimate_noise_sigma(s21_noisy, method='adaptive')
print(f"\n추정된 noise_sigma: {noise_sigma_est:.5f} "
      f"(참값 백색잡음 수준: {aa_white_noise_level})")
if aa_enable_tilt:
    print(f"  [주의] baseline tilt가 켜져 있으면, adaptive 방식이 baseline을")
    print(f"        '평평하다'고 가정하기 때문에 이 추정치가 왜곡될 수 있습니다.")

print("\nMCMC 실행 중...")
result = mcmc_pipeline.run_single_mcmc(
    log_prob, initial_guess, f_grid, s21_noisy, noise_sigma_est,
    nwalkers=aa_nwalkers, nsteps=aa_nsteps,
    init_scatter=aa_init_scatter, burn_in_discard=aa_burn_in_discard, thin_by=aa_thin_by,
    bounds=aa_prior_bounds, suppress_warnings=True
)

# =========================================================
# STEP 5. 결과 평가 - 참값과 비교해 이 시나리오가 얼마나 파이프라인을 흔들었는지
# =========================================================
true_theta = np.array([aa_f_r0, aa_g, aa_kappa])
labels = ['f_r', 'g', 'kappa']

print("\n" + "=" * 60)
print("결과: 참값 대비 복원 정확도")
print("=" * 60)
for i, name in enumerate(labels):
    err_pct = abs(result['median'][i] - true_theta[i]) / true_theta[i] * 100
    flag = "  ⚠️ 10% 이상 벗어남!" if err_pct > 10 else ""
    print(f"  {name:8s}: 추정={result['median'][i]:.5f}  참값={true_theta[i]:.5f}  "
          f"오차={err_pct:.1f}%{flag}")

print(f"\n  수렴 여부(converged): {result['converged']}")

# =========================================================
# STEP 6. 시각화 - 데이터 형태와 MCMC 최적 피팅을 함께 확인
# =========================================================
model_best = models.s21_anticrossing_model(
    f_grid, aa_case_flux_val, *result['median'], aa_fq_max, aa_EC, aa_tau
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(f_grid, np.abs(s21_noisy), '.', ms=3, alpha=0.5, label='왜곡된 데이터')
axes[0].plot(f_grid, np.abs(model_best), '-', color='red', label='MCMC 피팅(표준 모델)')
axes[0].axvline(initial_guess[0], color='gray', ls='--', lw=1, label='초기 fr 추정')
axes[0].set_xlabel('Frequency (GHz)')
axes[0].set_ylabel('|S21|')
axes[0].legend(fontsize=8)
axes[0].set_title(f'시나리오: {applied_effects if applied_effects else ["기본"]}')

axes[1].plot(f_grid, np.angle(s21_noisy), '.', ms=3, alpha=0.5, label='왜곡된 데이터')
axes[1].plot(f_grid, np.angle(model_best), '-', color='red', label='MCMC 피팅')
axes[1].set_xlabel('Frequency (GHz)')
axes[1].set_ylabel('phase (rad)')
axes[1].legend(fontsize=8)
axes[1].set_title('Phase')

plt.tight_layout()
scenario_tag = '_'.join(applied_effects).replace('(', '').replace(')', '').replace('=', '').replace(',', '').replace(' ', '_') or 'baseline'
plt.savefig(os.path.join(aa_output_dir, f'stress_test_{scenario_tag}.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\n결과 그래프 저장: {os.path.join(aa_output_dir, f'stress_test_{scenario_tag}.png')}")
