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
STEP 8: bayesian_toolkit.py 사용 예시 - 대표로 중앙(Φ=0) slice
   하나를 깊게 파고들어, credible interval / HPD interval / R-hat /
   AIC-BIC 모델비교(gaussian vs robust)를 실제로 계산해봄.
STEP 8.5: corner plot 시각화 (STEP 8의 posterior 샘플 재사용)
STEP 9: fisher_matrix.py 사용 예시 - 피셔 행렬 vs MCMC posterior 비교
   (101개 전부에 대해 STEP 8~9를 다 하면 시간이 오래 걸리므로, "대표
    slice 하나로 방법론을 검증하고 필요하면 확장한다"는 실전 전략을 따름)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import models
import mock_data
import likelihood
import bayesian_toolkit as bt
import fisher_matrix as fm
import mcmc_pipeline
import diagnostics

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

# 우도 함수 선택: 리스트로 여러 개를 넣으면 자동으로 전부 순회하며 배치
# MCMC를 돌리고, 2개 이상이면 실행 끝에 정확도(RMS 오차) 비교까지 자동
# 수행합니다. 하나만 보고 싶으면 리스트에 하나만 남기면 됨.
aa_likelihood_types_to_run = ['gaussian', 'robust']
aa_robust_nu = 4.0   # robust 선택시에만 사용 (Student's t 분포의 자유도)

# 심층분석(STEP 8 이후)에는 위 리스트 중 어떤 결과를 대표로 쓸지 지정
aa_case_study_likelihood = 'gaussian'

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
# STEP 1. Clean 신호 생성 (models.py 사용) - 우도 종류와 무관하게 공유
# =========================================================
f_grid = np.linspace(*aa_freq_range, aa_n_freq)
flux_grid = np.linspace(*aa_flux_range, aa_n_flux)

s21_clean_2d = np.zeros((aa_n_flux, aa_n_freq), dtype=complex)
for i, flux_val in enumerate(flux_grid):
    s21_clean_2d[i, :] = models.s21_anticrossing_model(
        f_grid, flux_val, aa_f_r0, aa_g, aa_kappa, aa_fq_max, aa_EC, aa_tau
    )

# =========================================================
# STEP 2. 잡음 주입 (mock_data.py 사용) - 우도 종류와 무관하게 공유
# =========================================================
# 같은 seed를 쓰므로, gaussian/robust 실행 모두 "정확히 같은 지저분한
# 데이터"를 보게 됩니다 - 이래야 두 우도 방식의 비교가 공정해짐
# (데이터 자체가 다르면 어느 쪽이 더 정확한지는 데이터 차이 때문일 수도
#  있으므로, 비교 실험에서는 항상 조건을 통제해야 함).
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
# STEP 3~7을 함수로 캡슐화: 우도 종류 하나를 받아 배치 MCMC 전체를
# 실행하고, 결과 저장 + 시각화까지 마친 뒤 results 딕셔너리를 반환.
# 이렇게 함수로 묶어두면, 아래에서 여러 우도 종류에 대해 "똑같은 절차"를
# 반복 호출하기만 하면 되므로 코드 중복 없이 gaussian/robust를 자동으로
# 둘 다 돌릴 수 있습니다.
# =========================================================
def run_full_batch_pipeline(likelihood_type):
    """
    지정한 likelihood_type으로 101개 flux slice 전체에 대해 배치 MCMC를
    실행하고, 결과를 저장 및 시각화한 뒤 results 딕셔너리를 반환.
    """
    # --- STEP 3. 우도 함수 준비 (likelihood.py 사용) ---
    log_prior_local = likelihood.make_uniform_log_prior(aa_prior_bounds)

    def log_probability_factory(flux_val):
        """이 flux 값 전용 log_probability 함수를 만들어 반환"""
        model_kwargs = {'flux_val': flux_val, 'fq_max': aa_fq_max, 'EC': aa_EC, 'tau': aa_tau}
        return likelihood.make_log_probability(
            models.s21_anticrossing_model, model_kwargs, log_prior_local,
            likelihood_type=likelihood_type,
            **({'nu': aa_robust_nu} if likelihood_type == 'robust' else {})
        )

    # --- STEP 4. 초기값 추정 함수 (diagnostics.py의 moving_average 활용) ---
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

    # --- STEP 5. 배치 MCMC 실행 (mcmc_pipeline.py 사용) ---
    def sigma_estimator(s21_1d):
        return diagnostics.estimate_noise_sigma(s21_1d, method='adaptive')

    print(f"\n총 {aa_n_flux}개 slice에 대해 배치 MCMC 시작 "
          f"(likelihood={likelihood_type})...\n")

    results_local = mcmc_pipeline.run_batch_mcmc_warmstart(
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

    n_ok = np.sum(results_local['status'] == 'ok')
    n_unstable = np.sum(results_local['status'] == 'unstable')
    n_failed = np.sum(results_local['status'] == 'failed')
    print(f"\n완료: ok={n_ok}, unstable={n_unstable}, failed={n_failed} (전체 {aa_n_flux})")

    # --- STEP 6. 진단: 안정 추정 가능 범위 탐지 (diagnostics.py 사용) ---
    fq_arr = models.qubit_frequency_vs_flux(results_local['x'], aa_fq_max, aa_EC)
    abs_delta = np.abs(fq_arr - aa_f_r0)
    g_err_total = results_local['err_lo'][:, 1] + results_local['err_hi'][:, 1]

    threshold_crossing = diagnostics.find_stability_threshold_by_crossing(
        abs_delta, g_err_total, threshold_line=0.02
    )
    threshold_inflection = diagnostics.find_stability_threshold_by_inflection(
        abs_delta, g_err_total
    )
    print(f"\n[진단] threshold-crossing 방식: |Δ| ≈ {threshold_crossing:.4f} GHz")
    print(f"[진단] 변곡점 자동탐지 방식   : |Δ| ≈ {threshold_inflection:.4f} GHz")

    # --- STEP 7. 결과 저장 및 시각화 ---
    result_filename = f'results_{likelihood_type}_outlier{aa_outlier_probability}.npz'
    np.savez(
        os.path.join(aa_output_dir, result_filename),
        true_fr=aa_f_r0, true_g=aa_g, true_kappa=aa_kappa,
        **results_local
    )

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    labels_local = [r'$f_r$ (GHz)', r'$g$ (GHz)', r'$\kappa$ (GHz)']
    colors = {'ok': 'tab:blue', 'unstable': 'tab:orange', 'failed': 'tab:red'}
    for status_name, color in colors.items():
        mask = results_local['status'] == status_name
        if not np.any(mask):
            continue
        for i in range(3):
            axes[i].errorbar(results_local['x'][mask], results_local['median'][mask, i],
                              yerr=[results_local['err_lo'][mask, i], results_local['err_hi'][mask, i]],
                              fmt='o', ms=3, color=color, label=status_name if i == 0 else None, alpha=0.7)
    for i, true_val in enumerate([aa_f_r0, aa_g, aa_kappa]):
        axes[i].axhline(true_val, color='gray', ls='--', lw=1)
        axes[i].set_ylabel(labels_local[i])
    axes[2].set_xlabel(r'Flux $\Phi/\Phi_0$')
    axes[0].legend()
    axes[0].set_title(f'Template Pipeline Result (likelihood={likelihood_type})')
    plt.tight_layout()
    plt.savefig(os.path.join(aa_output_dir, f'template_run_result_{likelihood_type}.png'),
                dpi=150, bbox_inches='tight')
        # [수정] 파일명에 likelihood_type을 넣어, gaussian/robust 그림도
        # 서로 덮어쓰지 않고 둘 다 남도록 함.
    plt.show()

    print(f"\n결과 저장 완료: {os.path.join(aa_output_dir, result_filename)}")
    return results_local


# =========================================================
# STEP 3~7 실행: aa_likelihood_types_to_run에 있는 모든 우도 종류에 대해
# 위 함수를 반복 호출. 결과는 {likelihood_type: results} 딕셔너리에 모음.
# =========================================================
all_results = {}
for lt in aa_likelihood_types_to_run:
    print("\n" + "#" * 60)
    print(f"# Likelihood = {lt} 로 전체 파이프라인 실행")
    print("#" * 60)
    all_results[lt] = run_full_batch_pipeline(lt)

# =========================================================
# STEP 7.5. Gaussian vs Robust 정확도(참값 대비 오차) 자동 비교
# =========================================================
# aa_likelihood_types_to_run에 'gaussian'과 'robust'가 둘 다 있을 때만
# 비교를 수행. (compare_gaussian_vs_robust.py와 동일한 로직을 여기
# 인라인으로 재사용 - 별도 스크립트를 다시 실행할 필요 없이 한 번의
# example_run.py 실행으로 비교까지 끝나도록 통합.)
if 'gaussian' in all_results and 'robust' in all_results:
    print("\n" + "=" * 60)
    print("STEP 7.5. Gaussian vs Robust 정확도 비교")
    print("=" * 60)

    true_values = np.array([aa_f_r0, aa_g, aa_kappa])
    param_labels_compare = [r'$f_r$', r'$g$', r'$\kappa$']

    def compute_accuracy_metrics(results_dict, true_vals):
        median = results_dict['median']
        errors = median - true_vals[np.newaxis, :]
        bias = np.mean(errors, axis=0)
        rms_error = np.sqrt(np.mean(errors ** 2, axis=0))
        return bias, rms_error, errors

    gauss_bias, gauss_rms, gauss_errors = compute_accuracy_metrics(all_results['gaussian'], true_values)
    robust_bias, robust_rms, robust_errors = compute_accuracy_metrics(all_results['robust'], true_values)

    print(f"\n{'파라미터':<10} {'Gaussian bias':>15} {'Robust bias':>15} "
          f"{'Gaussian RMS':>15} {'Robust RMS':>15}")
    print("-" * 70)
    for i, label in enumerate(param_labels_compare):
        print(f"{label:<10} {gauss_bias[i]:>15.6f} {robust_bias[i]:>15.6f} "
              f"{gauss_rms[i]:>15.6f} {robust_rms[i]:>15.6f}")

    print()
    for i, label in enumerate(param_labels_compare):
        winner = 'robust' if robust_rms[i] < gauss_rms[i] else 'gaussian'
        improvement = abs(gauss_rms[i] - robust_rms[i]) / gauss_rms[i] * 100
        print(f"{label}: RMS 오차 기준 {winner}가 더 정확함 (차이 {improvement:.1f}%)")

    fig_cmp, axes_cmp = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    flux_compare = all_results['gaussian']['x']
    for i, label in enumerate(param_labels_compare):
        axes_cmp[i].axhline(0, color='black', lw=0.5)
        axes_cmp[i].plot(flux_compare, gauss_errors[:, i], 'o', ms=3, alpha=0.6,
                          color='tab:red', label='gaussian error')
        axes_cmp[i].plot(flux_compare, robust_errors[:, i], 'o', ms=3, alpha=0.6,
                          color='tab:green', label='robust error')
        axes_cmp[i].set_ylabel(f'{label} 오차\n(추정값-참값)')
    axes_cmp[0].legend()
    axes_cmp[0].set_title(f'Gaussian vs Robust: 참값 대비 오차 비교 '
                           f'(outlier_probability={aa_outlier_probability})')
    axes_cmp[2].set_xlabel(r'Flux $\Phi/\Phi_0$')
    plt.tight_layout()
    plt.savefig(os.path.join(aa_output_dir, 'gaussian_vs_robust_accuracy.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n비교 그래프 저장 완료: "
          f"{os.path.join(aa_output_dir, 'gaussian_vs_robust_accuracy.png')}")
else:
    print("\n[안내] gaussian과 robust를 둘 다 실행하지 않아 정확도 비교를 건너뜁니다.")
    print("       비교하려면 aa_likelihood_types_to_run = ['gaussian', 'robust']로 설정하세요.")

# 이후 STEP 8(심층분석)에서 쓸 대표 결과를 선택
results = all_results[aa_case_study_likelihood]

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
      f"Φ={flux_grid[aa_case_study_flux_idx]:.3f}, 대표 우도={aa_case_study_likelihood})")
print("=" * 60)

case_flux_val = flux_grid[aa_case_study_flux_idx]
case_data_1d = s21_noisy_2d[aa_case_study_flux_idx, :]
case_sigma = diagnostics.estimate_noise_sigma(case_data_1d, method='adaptive')

# [수정] initial_guess_from_dip_search가 run_full_batch_pipeline() 함수
# 안의 지역함수였으므로, STEP 8에서 다시 쓰려면 여기서 동일한 로직으로
# 재정의해야 함 (함수 안에서 정의된 이름은 함수 밖에서 보이지 않는 것이
# 파이썬의 스코프(scope, 변수가 유효한 범위) 규칙).
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

case_initial_guess = initial_guess_from_dip_search(f_grid, case_data_1d)

# --- 8.1 gaussian/robust 우도로 각각 단일 MCMC 실행 (원본 체인까지 보존) ---
# [수정] log_prior도 마찬가지로 run_full_batch_pipeline() 안의 지역변수였으므로
# 여기서 다시 만듦 (STEP 3의 aa_prior_bounds는 전역이라 그대로 재사용 가능).
log_prior = likelihood.make_uniform_log_prior(aa_prior_bounds)
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

# =========================================================
# STEP 8.5. Corner plot 시각화 (누락되어 있던 부분 - 추가)
# =========================================================
# STEP 8에서 이미 확보된 result_gauss['flat_samples']를 이용해
# fr-g-kappa 세 파라미터 간의 posterior 분포와 상관관계(대각선 방향
# 퍼짐 등)를 한눈에 보여주는 2D corner plot을 그립니다.
# (개별 스크립트로 작업할 때는 이 시각화를 했었는데, 템플릿으로
#  리팩토링하는 과정에서 텍스트 통계치만 남고 그림이 빠졌던 부분)
try:
    import corner
        # corner: MCMC posterior 샘플을 "모서리 그림(corner plot)" 형태로
        # 시각화하는 전용 라이브러리. 대각선에는 각 파라미터의 1D
        # 히스토그램(주변분포, marginal distribution), 비대각선에는
        # 두 파라미터씩 짝지은 2D 등고선(결합분포, joint distribution)을
        # 그려줌 - 파라미터 간 상관관계(축퇴)를 한눈에 파악하기 좋음.

    fig_corner = corner.corner(
        result_gauss['flat_samples'],
        labels=[r'$f_r$ (GHz)', r'$g$ (GHz)', r'$\kappa$ (GHz)'],
        truths=[aa_f_r0, aa_g, aa_kappa],
            # truths: 참값 위치에 빨간 선/점을 표시 (mock 데이터라 참값을
            # 알고 있으므로 표시 가능 - 실측 데이터라면 이 인자는 생략)
        truth_color='red',
        show_titles=True,
            # show_titles: 각 대각선 패널 위에 "중앙값 +상한/-하한" 형태의
            # 요약값을 자동으로 표시
        title_kwargs={'fontsize': 11},
    )
    fig_corner.suptitle(
        f'Posterior Corner Plot (flux idx={aa_case_study_flux_idx}, likelihood=gaussian)',
        y=1.02, fontsize=13
    )
    plt.savefig(os.path.join(aa_output_dir, 'corner_plot_case_study.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nCorner plot 저장 완료: "
          f"{os.path.join(aa_output_dir, 'corner_plot_case_study.png')}")
except ImportError:
    # corner 패키지가 설치 안 된 환경을 대비한 안전장치.
    # (Colab에는 기본적으로 없을 수 있어 !pip install corner 필요할 수 있음)
    print("\n[안내] 'corner' 패키지가 설치되어 있지 않아 corner plot을 건너뜁니다.")
    print("       Colab/터미널에서 다음을 실행 후 다시 시도하세요: pip install corner")

# =========================================================
# STEP 9. 피셔 행렬 vs MCMC posterior 비교 (fisher_matrix.py 사용)
# =========================================================
# 중력파 분석에서 하시던 것과 동일한 비교: 같은 데이터, 같은 물리
# 모델에 대해 (1) 피셔 행렬로 빠르게 근사한 오차와 (2) MCMC로 직접
# 샘플링한 posterior의 오차가 서로 얼마나 일치하는지 확인합니다.
# 일치할수록 "이 근방에서 log-likelihood가 2차함수(가우시안)에
# 가깝다"는 뜻이고, 어긋날수록 "posterior가 비선형/비대칭"이라는
# 신호입니다 (그럴 땐 MCMC 결과를 신뢰하는 것이 안전).

print("\n" + "=" * 60)
print("STEP 9. 피셔 행렬 vs MCMC posterior 비교")
print("=" * 60)

# MCMC 결과의 median을 "기준점"으로 사용 (피셔 행렬은 특정 지점에서의
# 곡률만 보므로, 그 지점을 어디로 잡을지가 중요함 - 참값을 모르는 실전
# 상황을 가정해 MCMC median을 씀).
theta_best_gauss = result_gauss['median']

sigma_fisher, corr_fisher = fm.fisher_parameter_uncertainties(
    models.s21_anticrossing_model, theta_best_gauss, f_grid, case_sigma, model_kwargs_case
)

# MCMC posterior의 오차폭(비대칭 오차의 평균으로 단순화해 피셔의
# "대칭적인 1-sigma"와 같은 기준으로 비교)
sigma_mcmc = (result_gauss['err_lo'] + result_gauss['err_hi']) / 2

param_labels = ['f_r', 'g', 'kappa']
print(f"\n{'파라미터':<10} {'MCMC 1σ':>12} {'피셔 1σ':>12} {'비율(MCMC/피셔)':>16}")
print("-" * 55)
for i, name in enumerate(param_labels):
    ratio = sigma_mcmc[i] / sigma_fisher[i]
    print(f"{name:<10} {sigma_mcmc[i]:>12.6f} {sigma_fisher[i]:>12.6f} {ratio:>16.3f}")

print(f"\n[해석 가이드]")
print(f"  비율이 1에 가까우면 -> 이 지점에서 posterior가 거의 가우시안(대칭)형")
print(f"  비율이 1보다 많이 크면 -> posterior가 피셔 예측보다 넓음")
print(f"    (비선형 축퇴, 다봉 구조, 또는 outlier로 인한 꼬리 확장 등이 원인일 수 있음)")
print(f"  비율이 1보다 많이 작으면 -> posterior가 피셔 예측보다 좁음")
print(f"    (prior가 posterior 폭을 실제로 좁혀준 경우 등에서 나타날 수 있음)")

print(f"\n[상관관계 비교] 피셔 행렬 기반 fr-g 상관계수: {corr_fisher[0,1]:.3f}")
print(f"  -> corner plot(있다면)에서 본 fr-g의 대각선 방향 상관관계와")
print(f"     부호/크기가 비슷한지 육안으로도 비교해볼 수 있습니다.")

print("\nSTEP 9 완료.")

