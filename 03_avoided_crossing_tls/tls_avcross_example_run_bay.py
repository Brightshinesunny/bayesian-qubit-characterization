"""
example_run.py
=================================================================
템플릿 6개 모듈(models, mock_data, likelihood, bayesian_toolkit,
mcmc_pipeline, diagnostics)을 엮어서, 오늘 했던 전체 분석 흐름을
재현하는 예시 스크립트입니다.

이 파일 자체가 "새 프로젝트를 시작할 때 복사해서 쓰는 시작점" 역할을
하도록 설계했습니다. 다른 물리계를 다루게 되면:
  1. models.py 에 새 forward model 함수 추가
  2. 아래 STEP 1~2에서 그 함수를 부르도록 교체
  3. 나머지(STEP 3 이후)는 거의 그대로 재사용

STEP 1~7: 101개 flux slice 전체 배치 피팅 (기존과 동일)
STEP 8(신규): bayesian_toolkit.py 사용 예시 - 대표로 중앙(Φ=0) slice
   하나를 깊게 파고들어, credible interval / HPD interval / R-hat /
   AIC-BIC 모델비교(gaussian vs robust)를 실제로 계산해봄.
   (101개 전부에 대해 이걸 다 하면 시간이 오래 걸리므로, "대표 slice
    하나로 방법론을 검증하고 필요하면 확장한다"는 실전 전략을 따름)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import tls_avcross_models as models
import tls_avcross_mock_data as mock_data
import tls_avcross_likelihood as likelihood
import tls_avcross_bayesian_toolkit as bt
import tls_avcross_mcmc_pipeline as mcmc_pipeline
import tls_avcross_diagnostics as diagnostics

# =========================================================
# STEP 0. 분석자 설정값 (aa_ 접두사)
# =========================================================
aa_output_dir = './outputs/'
os.makedirs(aa_output_dir, exist_ok=True)

# 참값 (실측 데이터라면 모르는 값들 - 여기서는 mock 검증용으로 사용)
aa_f_r0, aa_g, aa_kappa = 5.000, 0.040, 0.030
aa_fq_max, aa_EC, aa_tau = 5.150, 0.250, 0.12

aa_n_freq, aa_n_flux = 401, 101
aa_freq_range = (4.8, 5.2)
aa_flux_range = (-0.5, 0.5)

# 잡음 주입 강도 (0으로 두면 해당 잡음 꺼짐 -> "얼마나 지저분하게 할지" 조절)
aa_white_level = 0.015
aa_flicker_level = 0.0     # 0 = 꺼짐. 1/f 잡음 연습하려면 0.02 정도로.
aa_drift_amplitude = 0.0   # 0 = 꺼짐. 드리프트 연습하려면 0.08 정도로.
aa_outlier_probability = 0.0  # 0 = 꺼짐. outlier 연습하려면 0.01 정도로.
aa_outlier_scale = 0.4

# 우도 함수 선택: 'gaussian'(기존) 또는 'robust'(outlier에 강건)
aa_likelihood_type = 'gaussian'
aa_robust_nu = 4.0   # robust 선택시에만 사용 (Student's t 분포의 자유도)

# MCMC 설정
aa_nwalkers = 32
aa_nsteps_per_slice = 800
aa_burn_in_discard = 200
aa_thin_by = 10
aa_init_scatter = 5e-3
aa_prior_bounds = {'f_r': (4.8, 5.2), 'g': (0.001, 0.1), 'kappa': (0.001, 0.1)}
aa_unstable_threshold = {1: 0.02, 2: 0.02}   # 인덱스1(g), 인덱스2(kappa)의 오차 임계값

print("=" * 60)
print("분석자 설정값(aa_*) 요약:")
for k, v in list(globals().items()):
    if k.startswith("aa_"):
        print(f"  {k:28s} = {v}")
print("=" * 60)

# =========================================================
# STEP 1. Clean 신호 생성 (models.py 사용)
# =========================================================
f_grid = np.linspace(*aa_freq_range, aa_n_freq)
flux_grid = np.linspace(*aa_flux_range, aa_n_flux)

s21_clean_2d = np.zeros((aa_n_flux, aa_n_freq), dtype=complex)
for i, flux_val in enumerate(flux_grid):
    s21_clean_2d[i, :] = models.s21_anticrossing_model(
        f_grid, flux_val, aa_f_r0, aa_g, aa_kappa, aa_fq_max, aa_EC, aa_tau
    )

# =========================================================
# STEP 2. 잡음 주입 (mock_data.py 사용)
# =========================================================
s21_noisy_2d, outlier_mask = mock_data.make_realistic_2d_dataset(
    s21_clean_2d,
    white_level=aa_white_level,
    flicker_level=aa_flicker_level,
    drift_amplitude=aa_drift_amplitude,
    outlier_probability=aa_outlier_probability,
    outlier_scale=aa_outlier_scale,
    seed=123,
)
print(f"\n생성된 outlier 개수: {np.sum(outlier_mask)} / {outlier_mask.size}")

# =========================================================
# STEP 3. 우도 함수 준비 (likelihood.py 사용)
# =========================================================
log_prior = likelihood.make_uniform_log_prior(aa_prior_bounds)

def log_probability_factory(flux_val):
    """이 flux 값 전용 log_probability 함수를 만들어 반환"""
    model_kwargs = {'flux_val': flux_val, 'fq_max': aa_fq_max, 'EC': aa_EC, 'tau': aa_tau}
    return likelihood.make_log_probability(
        models.s21_anticrossing_model, model_kwargs, log_prior,
        likelihood_type=aa_likelihood_type,
        **({'nu': aa_robust_nu} if aa_likelihood_type == 'robust' else {})
    )

# =========================================================
# STEP 4. 초기값 추정 함수 (diagnostics.py의 moving_average 활용)
# =========================================================
def initial_guess_from_dip_search(f_grid_local, s21_1d):
    mag = np.abs(s21_1d)
    mag_smooth = diagnostics.moving_average(mag, 5)
    dip_candidates = [i for i in range(2, len(mag_smooth) - 2)
                       if mag_smooth[i] < mag_smooth[i-1] and mag_smooth[i] < mag_smooth[i+1]]
    if len(dip_candidates) >= 2:
        dip_candidates.sort(key=lambda i: mag_smooth[i])
        two = sorted(dip_candidates[:2])
        fr_g = 0.5 * (f_grid_local[two[0]] + f_grid_local[two[1]])
        g_g = abs(f_grid_local[two[1]] - f_grid_local[two[0]]) / 2.0
    else:
        fr_g, g_g = 5.0, 0.01
    return np.array([fr_g, g_g, 0.02])

# =========================================================
# STEP 5. 배치 MCMC 실행 (mcmc_pipeline.py 사용)
# =========================================================
def sigma_estimator(s21_1d):
    return diagnostics.estimate_noise_sigma(s21_1d, method='adaptive')

print(f"\n총 {aa_n_flux}개 slice에 대해 배치 MCMC 시작 "
      f"(likelihood={aa_likelihood_type})...\n")

results = mcmc_pipeline.run_batch_mcmc_warmstart(
    log_probability_factory=log_probability_factory,
    x_values=flux_grid,
    data_2d=s21_noisy_2d,
    sigma_estimator=sigma_estimator,
    initial_guess_func=initial_guess_from_dip_search,
    f_grid=f_grid,
    nwalkers=aa_nwalkers,
    nsteps_per_point=aa_nsteps_per_slice,
    init_scatter=aa_init_scatter,
    burn_in_discard=aa_burn_in_discard,
    thin_by=aa_thin_by,
    bounds=aa_prior_bounds,
    unstable_threshold=aa_unstable_threshold,
)

n_ok = np.sum(results['status'] == 'ok')
n_unstable = np.sum(results['status'] == 'unstable')
n_failed = np.sum(results['status'] == 'failed')
print(f"\n완료: ok={n_ok}, unstable={n_unstable}, failed={n_failed} (전체 {aa_n_flux})")

# =========================================================
# STEP 6. 진단: 안정 추정 가능 범위 탐지 (diagnostics.py 사용)
# =========================================================
fq_arr = models.qubit_frequency_vs_flux(results['x'], aa_fq_max, aa_EC)
abs_delta = np.abs(fq_arr - aa_f_r0)
g_err_total = results['err_lo'][:, 1] + results['err_hi'][:, 1]

threshold_crossing = diagnostics.find_stability_threshold_by_crossing(
    abs_delta, g_err_total, threshold_line=0.02
)
threshold_inflection = diagnostics.find_stability_threshold_by_inflection(
    abs_delta, g_err_total
)
print(f"\n[진단] threshold-crossing 방식: |Δ| ≈ {threshold_crossing:.4f} GHz")
print(f"[진단] 변곡점 자동탐지 방식   : |Δ| ≈ {threshold_inflection:.4f} GHz")

# =========================================================
# STEP 7. 결과 저장 및 시각화
# =========================================================
np.savez(os.path.join(aa_output_dir, 'template_run_results.npz'), **results)

fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
labels = [r'$f_r$ (GHz)', r'$g$ (GHz)', r'$\kappa$ (GHz)']
colors = {'ok': 'tab:blue', 'unstable': 'tab:orange', 'failed': 'tab:red'}
for status_name, color in colors.items():
    mask = results['status'] == status_name
    if not np.any(mask):
        continue
    for i in range(3):
        axes[i].errorbar(results['x'][mask], results['median'][mask, i],
                          yerr=[results['err_lo'][mask, i], results['err_hi'][mask, i]],
                          fmt='o', ms=3, color=color, label=status_name if i == 0 else None, alpha=0.7)
for i, true_val in enumerate([aa_f_r0, aa_g, aa_kappa]):
    axes[i].axhline(true_val, color='gray', ls='--', lw=1)
    axes[i].set_ylabel(labels[i])
axes[2].set_xlabel(r'Flux $\Phi/\Phi_0$')
axes[0].legend()
axes[0].set_title(f'Template Pipeline Result (likelihood={aa_likelihood_type})')
plt.tight_layout()
plt.savefig(os.path.join(aa_output_dir, 'template_run_result.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n결과 저장 완료:", aa_output_dir)

# =========================================================
# STEP 8. bayesian_toolkit.py 사용 예시 (대표 slice 심층 분석)
# =========================================================
# 101개 slice 전부에 아래 진단을 다 돌리면 시간이 오래 걸리므로,
# 물리적으로 가장 신호가 뚜렷한 중앙(Φ=0, avoided-crossing이 가장
# 선명한 지점) 하나를 골라 "이 방법론이 잘 작동하는지" 검증하는
# 대표 분석(case study)을 수행합니다. 여기서 검증된 방법론은 필요하면
# STEP 3~6의 배치 루프 안에 그대로 넣어 101개 전부로 확장할 수 있습니다.

aa_case_study_flux_idx = aa_n_flux // 2   # 중앙(Φ=0) slice 선택
aa_credible_level = 0.68                   # 68% (가우시안의 ±1시그마와 유사) 신뢰구간

print("\n" + "=" * 60)
print(f"STEP 8. 베이지안 도구 심층 분석 (flux idx={aa_case_study_flux_idx}, "
      f"Φ={flux_grid[aa_case_study_flux_idx]:.3f})")
print("=" * 60)

case_flux_val = flux_grid[aa_case_study_flux_idx]
case_data_1d = s21_noisy_2d[aa_case_study_flux_idx, :]
case_sigma = diagnostics.estimate_noise_sigma(case_data_1d, method='adaptive')
case_initial_guess = initial_guess_from_dip_search(f_grid, case_data_1d)

# --- 8.1 gaussian 우도로 단일 MCMC 실행 (원본 체인까지 보존) ---
log_prob_gaussian = log_probability_factory(case_flux_val)
    # 주의: 이 시점에서 log_probability_factory는 STEP 3에서 aa_likelihood_type
    # 값으로 이미 고정되어 만들어진 함수. gaussian/robust를 둘 다 비교하려면
    # 아래처럼 likelihood_type을 직접 지정해 별도로 만들어야 함.
model_kwargs_case = {'flux_val': case_flux_val, 'fq_max': aa_fq_max, 'EC': aa_EC, 'tau': aa_tau}

log_prob_gauss = likelihood.make_log_probability(
    models.s21_anticrossing_model, model_kwargs_case, log_prior,
    likelihood_type='gaussian'
)
log_prob_robust = likelihood.make_log_probability(
    models.s21_anticrossing_model, model_kwargs_case, log_prior,
    likelihood_type='robust', nu=aa_robust_nu
)

result_gauss = mcmc_pipeline.run_single_mcmc(
    log_prob_gauss, case_initial_guess, f_grid, case_data_1d, case_sigma,
    nwalkers=aa_nwalkers, nsteps=2000,   # 심층분석이라 배치보다 스텝을 넉넉히
    burn_in_discard=aa_burn_in_discard, thin_by=aa_thin_by,
    bounds=aa_prior_bounds, suppress_warnings=True
)
result_robust = mcmc_pipeline.run_single_mcmc(
    log_prob_robust, case_initial_guess, f_grid, case_data_1d, case_sigma,
    nwalkers=aa_nwalkers, nsteps=2000,
    burn_in_discard=aa_burn_in_discard, thin_by=aa_thin_by,
    bounds=aa_prior_bounds, suppress_warnings=True
)

# --- 8.2 Credible interval vs HPD interval 비교 ---
# g(결합강도) 파라미터(인덱스 1)의 posterior 샘플로 두 종류의 구간을 비교.
g_samples = result_gauss['flat_samples'][:, 1]

ci_lo, ci_med, ci_hi = bt.credible_interval(g_samples, level=aa_credible_level)
hpd_lo, hpd_hi = bt.highest_posterior_density_interval(g_samples, level=aa_credible_level)

print(f"\n[8.2] g의 posterior 구간 비교 (level={aa_credible_level}):")
print(f"  등꼬리 신뢰구간(credible interval): [{ci_lo:.5f}, {ci_hi:.5f}]  (중앙값 {ci_med:.5f})")
print(f"  HPD 구간(highest posterior density): [{hpd_lo:.5f}, {hpd_hi:.5f}]")
print(f"  -> posterior가 거의 대칭(가우시안형)이면 두 구간이 비슷하게 나오고,")
print(f"     비대칭이거나 다봉(multi-modal)이면 HPD가 더 좁고 정확한 구간을 줌.")

# --- 8.3 Gelman-Rubin R-hat으로 수렴 여부 정량 진단 ---
# raw_chain의 shape: (걸음수, 워커수, 파라미터수).
# gelman_rubin_rhat은 (체인개수, 걸음수) 형태를 기대하므로, 각 워커를
# "하나의 독립 체인"으로 보고 축을 바꿔줌(transpose).
raw_chain_g = result_gauss['raw_chain'][:, :, 1].T   # (걸음수,워커수)->(워커수,걸음수)
    # [:, :, 1] : (걸음수, 워커수, 파라미터수) 배열에서 g(인덱스1)만 선택
    # .T (transpose): 축의 순서를 뒤바꿔 (워커수, 걸음수) 형태로 맞춤
rhat_g = bt.gelman_rubin_rhat(raw_chain_g)
print(f"\n[8.3] g 파라미터의 R-hat(수렴 진단): {rhat_g:.4f}")
print(f"  -> 1.01 미만이면 '잘 수렴했다'고 흔히 판단하는 경험적 기준.")

if result_gauss['autocorr_time'] is not None:
    ess = bt.effective_sample_size_approx(
        result_gauss['autocorr_time'][1], result_gauss['flat_samples'].shape[0]
    )
    print(f"  유효 샘플 크기(ESS, g 기준) ≈ {ess:.0f} "
          f"(수백 이상이면 posterior 요약이 안정적이라고 봄)")

# --- 8.4 AIC/BIC로 gaussian vs robust 우도 비교 ---
# 두 모델(gaussian 우도, robust 우도)이 "같은 데이터"를 얼마나 잘
# 설명하는지 비교. 값이 작을수록 더 선호되는 모델.
n_data_points = 2 * len(f_grid)   # 실수부+허수부를 각각 하나의 데이터점으로 취급
n_params = 3   # f_r, g, kappa

if result_gauss['max_log_prob'] is not None and result_robust['max_log_prob'] is not None:
    aic_gauss = bt.aic(result_gauss['max_log_prob'], n_params)
    aic_robust = bt.aic(result_robust['max_log_prob'], n_params)
    bic_gauss = bt.bic(result_gauss['max_log_prob'], n_params, n_data_points)
    bic_robust = bt.bic(result_robust['max_log_prob'], n_params, n_data_points)

    print(f"\n[8.4] 모델 비교 (gaussian vs robust 우도, 같은 데이터에 대해):")
    print(f"  AIC: gaussian={aic_gauss:.2f}  robust={aic_robust:.2f}  "
          f"(작을수록 선호 -> {'gaussian' if aic_gauss < aic_robust else 'robust'} 우세)")
    print(f"  BIC: gaussian={bic_gauss:.2f}  robust={bic_robust:.2f}  "
          f"(작을수록 선호 -> {'gaussian' if bic_gauss < bic_robust else 'robust'} 우세)")
    print(f"  참고: 지금 데이터는 outlier가 없는(aa_outlier_probability=0) 상태이므로")
    print(f"        gaussian이 근소하게 우세하거나 비슷하게 나오는 것이 정상입니다.")
    print(f"        aa_outlier_probability를 0.01 이상으로 켜고 다시 돌려보면")
    print(f"        robust 쪽이 더 선호되는 방향으로 바뀌는지 확인할 수 있습니다.")

print("\nSTEP 8 완료.")

