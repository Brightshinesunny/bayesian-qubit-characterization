"""
fano_single_resonator_corner_v2.py
=================================================================
[v2 - 라이브러리 사용 버전] 이전 버전(fano_single_resonator_corner.py)은
급하게 즉석에서 짠 코드라, prior/우도/공분산 계산을 전부 이 파일
안에서 직접 구현했습니다. 이번엔 그 부분들을 전부 정식 라이브러리
함수로 교체합니다:

  이전(즉석 코드)                    이번(라이브러리)
  --------------------------------------------------------------
  scipy.optimize.curve_fit bounds   fano_likelihood.make_uniform_log_prior
  (직접 계산한 chi2)                 fano_likelihood.make_log_probability_generic
                                     (오늘 새로 추가한 범용 버전)
  fano_fisher_matrix 함수들          동일(원래도 라이브러리 사용 중이었음)
  (직접 계산한 16%/84% 백분위수)     fano_bayesian_toolkit.credible_interval

[emcee 관련 중요한 안내]
이 스크립트가 실행되는 환경(샌드박스)에는 emcee가 설치되어 있지
않고, 네트워크가 막혀 있어 설치도 불가능합니다. 그래서 진짜
MCMC(fano_mcmc_pipeline.run_single_mcmc)는 여기서 못 돌립니다.
대신 이전과 같은 방식(curve_fit으로 MAP 추정 + 피셔행렬로 공분산
+ 공분산에서 몬테카를로 샘플링)을 쓰되, log_probability 함수
자체는 이번에 fano_likelihood.make_log_probability_generic으로
만들어서, "이 함수를 그대로 Colab의 fano_mcmc_pipeline.run_single_mcmc
에 넘기면 진짜 MCMC도 바로 돌아간다"는 걸 보장합니다 - 즉 이
스크립트는 "emcee 없이도 돌아가는 근사 버전"이면서 동시에
"emcee가 있는 환경에서 진짜 MCMC로 그대로 업그레이드 가능한 코드"
입니다.
"""

import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import json

sys.path.insert(0, os.getcwd())
import fano_models
import fano_likelihood as likelihood
import fano_fisher_matrix as fm
import fano_bayesian_toolkit as bt


# =========================================================
# STEP 0. 분석자 설정값 (aa_ 접두사)
# =========================================================
aa_upload_dir = '/mnt/user-data/uploads/'
aa_resonator_choice = 'resonator_7_powersweep_overcoupled.npz'
aa_power_slice_index = 0
aa_n_ports = 1.0
aa_isolation_db = 15
aa_step_fraction = 1e-6
aa_phi_candidates = [-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5]
aa_mc_sample_size = 30000
aa_credible_level = 0.68
    # [조정 가능] fano_bayesian_toolkit.credible_interval에 넘길 신뢰수준.
    # 0.68=1-sigma에 해당(정규분포 기준), 0.95=2-sigma에 해당하는
    # 더 넓은(더 보수적인) 구간을 보고 싶으면 이 값을 0.95로 바꾸면 됨.

aa_output_dir = '/home/claude/qubit_analysis_template/outputs/'
os.makedirs(aa_output_dir, exist_ok=True)

print("=" * 60)
print("분석자 설정값(aa_*) 요약:")
for k, v in list(globals().items()):
    if k.startswith("aa_") and not isinstance(v, list):
        print(f"  {k:24s} = {v}")
print("=" * 60)


# =========================================================
# STEP 1. 데이터 로드
# =========================================================
def load_fano_npz(filepath):
    raw = np.load(filepath, allow_pickle=True)
    amplitude, phase = raw['amplitude'], raw['phase']
    freq_hz, power_dbm = raw['frequency'], raw['power']
    s21 = amplitude * np.exp(1j * phase)
    settings_dict = json.loads(raw['settings'][0])
    cable_delay = settings_dict.get('vna', {}).get('port1_edel', None)
    return {'power_dbm': power_dbm, 'freq_hz': freq_hz, 's21': s21, 'cable_delay': cable_delay}


full_path = os.path.join(aa_upload_dir, aa_resonator_choice)
data = load_fano_npz(full_path)
f_grid = data['freq_hz']
s21 = data['s21'][aa_power_slice_index, :]
delay = data['cable_delay']
power_val = data['power_dbm'][aa_power_slice_index]
mag = np.abs(s21)

print(f"\n분석 대상: {aa_resonator_choice}, power={power_val:.1f}dBm")


# =========================================================
# STEP 2. Circle fit (참고용)
# =========================================================
cf_result = fano_models.autofit(f_grid, s21, n_ports=aa_n_ports, fixed_delay=delay, isolation=aa_isolation_db)
qi_max_str = 'inf' if np.isinf(cf_result['Qi_max']) else f"{cf_result['Qi_max']:.1f}"
print(f"\n[참고] circle fit 결과:")
print(f"  fr={cf_result['fr']/1e9:.6f}GHz, Ql={cf_result['Ql']:.1f}, Qc={cf_result['Qc']:.1f}, "
      f"Qi={cf_result['Qi']:.1f}")
print(f"  Qi_min={cf_result['Qi_min']:.1f}, Qi_max={qi_max_str}")


# =========================================================
# STEP 3. [라이브러리 사용] prior + log_probability 정의
# =========================================================
param_names = ['fr', 'Ql', 'Qc', 'phi', 'a', 'alpha']
param_labels = [r'$f_r$', r'$Q_l$', r'$Q_c$', r'$\phi$', r'$a$', r'$\alpha$']

prior_bounds = {
    'fr':  (f_grid.min(), f_grid.max()),
    'Ql':  (1.0, 1e8),
    'Qc':  (1.0, 1e8),
    'phi': (-np.pi, np.pi),
    'a':   (1e-8, 1.0),
    'alpha': (-np.pi, np.pi),
}
log_prior = likelihood.make_uniform_log_prior(prior_bounds)
    # [라이브러리 사용] fano_likelihood.make_uniform_log_prior - 이전
    # 버전에서 scipy curve_fit의 bounds=(lower,upper) 배열로 직접
    # 넣었던 것을, 여기서는 딕셔너리 기반의 표준 prior 함수로 표현.

mask = mag > np.percentile(mag, 80)
noise_sigma = np.std(np.real(s21[mask]))
fixed_kwargs = {'delay': delay, 'n_ports': aa_n_ports}

log_probability = likelihood.make_log_probability_generic(
    fano_models.Sij, fixed_kwargs, param_names, log_prior, likelihood_type='gaussian'
)
    # [라이브러리 사용, 신규 추가한 범용 버전] 이 함수의 시그니처는
    # log_probability(theta, f_grid, data_1d, sigma)로, emcee가 있는
    # 환경(Colab)에서는 fano_mcmc_pipeline.run_single_mcmc(log_probability,
    # ...)에 그대로 넘겨서 진짜 MCMC를 돌릴 수 있습니다. 여기서는
    # emcee 없이, 이 함수를 "MAP을 찾는 목적함수"로만 재사용합니다
    # (아래 STEP 4에서 -log_probability를 최소화 = log_probability를
    # 최대화 = posterior의 최댓값(MAP)을 찾는 것과 동일).

print(f"\nnoise_sigma = {noise_sigma:.6f}")


# =========================================================
# STEP 4. [라이브러리 활용] log_probability를 목적함수로 삼아 MAP 탐색
# (multi-start, curve_fit 대신 scipy.optimize.minimize로 log_probability를
#  직접 최대화 - 이러면 prior까지 명시적으로 반영된 진짜 "MAP" 탐색이 됨)
# =========================================================
fr_guess = f_grid[np.argmin(mag)]
half_level = (mag.max() + mag.min()) / 2
above_half = np.where(mag > half_level)[0]
fwhm_guess = f_grid[above_half[-1]] - f_grid[above_half[0]] if len(above_half) > 1 else 1e6
Ql_guess = fr_guess / max(fwhm_guess, 1e3)

print(f"\nmulti-start 탐색 실행 중 (phi 후보 {len(aa_phi_candidates)}개)...")
best_theta, best_neg_log_prob = None, np.inf
for phi0 in aa_phi_candidates:
    theta0 = np.array([fr_guess, Ql_guess, Ql_guess * 1.1, phi0, mag.max() - mag.min(), phi0])

    def neg_log_prob(theta):
        # log_probability(theta, f_grid, data_1d, sigma)를 그대로
        # 재사용 - "부호만 뒤집어서" scipy.optimize.minimize(최솟값을
        # 찾는 함수)로 posterior의 최댓값(MAP)을 찾는 표준적인 트릭.
        val = log_probability(theta, f_grid, s21, noise_sigma)
        return -val if np.isfinite(val) else 1e10

    res = minimize(neg_log_prob, theta0, method='Nelder-Mead',
                    options={'maxiter': 20000, 'xatol': 1e-6, 'fatol': 1e-6})
    print(f"  phi0={phi0:+.1f} -> -log_prob={res.fun:.2f} "
          f"({'수렴' if res.success else '미수렴'})")
    if res.fun < best_neg_log_prob:
        best_neg_log_prob = res.fun
        best_theta = res.x

popt = best_theta
print(f"\n최종 채택 (-log_prob={best_neg_log_prob:.2f}):")
for i, name in enumerate(param_names):
    print(f"  {name}: {popt[i]:.6g}")


# =========================================================
# STEP 5. [라이브러리 사용] 피셔행렬 -> 공분산 -> 몬테카를로 샘플
# =========================================================
F = fm.fisher_information_matrix(fano_models.Sij, popt, param_names, f_grid,
                                    noise_sigma, fixed_kwargs, step_fraction=aa_step_fraction)
cov = fm.covariance_from_fisher(F)
eigvals = np.linalg.eigvalsh(cov)
print(f"\n공분산 행렬 최소 고유값: {eigvals.min():.3e} "
      f"({'정상(양수)' if eigvals.min() > 0 else '⚠️ 비정상(음수)'})")

mc_samples = np.random.default_rng(0).multivariate_normal(popt, cov, size=aa_mc_sample_size)

# =========================================================
# STEP 6. [라이브러리 사용] credible_interval로 신뢰구간 계산
# =========================================================
print(f"\n[파라미터별 {aa_credible_level*100:.0f}% 신뢰구간 - "
      f"fano_bayesian_toolkit.credible_interval 사용]")
for i, name in enumerate(param_names):
    lo, med, hi = bt.credible_interval(mc_samples[:, i], level=aa_credible_level)
        # [라이브러리 사용] fano_bayesian_toolkit.credible_interval은
        # (하한, 중앙값, 상한) 3개 값을 튜플로 반환함(문서화된 그대로).
        # 이전 버전에서 np.percentile(samples,16), np.percentile(
        # samples,84)를 따로 계산했던 것을, 이 함수 하나로 통일.
    print(f"  {name}: {med:.4g}  [{lo:.4g}, {hi:.4g}]")

Qi_mc = 1.0 / (1.0/mc_samples[:, 1] - 1.0/mc_samples[:, 2])
valid_qi = (Qi_mc > 0) & (Qi_mc < 1e7)
Qi_lo, Qi_med, Qi_hi = bt.credible_interval(Qi_mc[valid_qi], level=aa_credible_level)
print(f"\nQi (베이지안/피셔, 라이브러리 신뢰구간): "
      f"중앙값={Qi_med:.1f}, [{Qi_lo:.1f}, {Qi_hi:.1f}]")


# =========================================================
# STEP 7. [라이브러리 사용] AIC/BIC로 모델 적합도 요약
# =========================================================
log_lik_max = log_probability(popt, f_grid, s21, noise_sigma) - log_prior(popt)
    # log_probability = log_prior + log_likelihood 이므로, log_prior를
    # 빼면 순수한 log_likelihood만 남음 (AIC/BIC 공식은 우도만 필요).
aic_val = bt.aic(log_lik_max, n_params=len(param_names))
bic_val = bt.bic(log_lik_max, n_params=len(param_names), n_data=len(f_grid)*2)
    # [라이브러리 사용] fano_bayesian_toolkit.aic / bic: 정보량 기준
    # (파라미터가 많아질수록 페널티를 주는 모델 비교 지표). 오늘은
    # 모델이 하나뿐이라 비교 대상은 없지만, 나중에 "phi를 포함한
    # 모델 vs phi=0으로 고정한 단순 모델"을 이 지표로 비교할 수 있음
    # (첫날 avoided-crossing 예제에서 이미 이 용도로 썼던 것과 동일).
print(f"\nAIC={aic_val:.2f}, BIC={bic_val:.2f} (모델 비교용 지표, fano_bayesian_toolkit 사용)")


# =========================================================
# STEP 8. Corner plot
# =========================================================
def simple_corner_plot(samples, labels, truths=None, bins=40):
    n = samples.shape[1]
    fig, axes = plt.subplots(n, n, figsize=(2.2*n, 2.2*n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis('off')
                continue
            if i == j:
                ax.hist(samples[:, i], bins=bins, color='tab:green', alpha=0.7)
                if truths is not None:
                    ax.axvline(truths[i], color='red', ls='--', lw=1)
            else:
                ax.plot(samples[:, j], samples[:, i], '.', ms=1, alpha=0.05, color='tab:green')
                if truths is not None:
                    ax.plot(truths[j], truths[i], 'r+', ms=10, mew=2)
            if i == n-1:
                ax.set_xlabel(labels[j], fontsize=9)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(labels[i], fontsize=9)
            else:
                ax.set_yticklabels([])
    plt.tight_layout()
    return fig


fig = simple_corner_plot(mc_samples, param_labels, truths=popt)
fig.suptitle(f'{aa_resonator_choice} (라이브러리 버전 v2)\n'
             f'fano_likelihood + fano_fisher_matrix + fano_bayesian_toolkit 사용',
             y=1.01, fontsize=11)
plt.savefig(os.path.join(aa_output_dir, 'fano_single_resonator_corner_v2.png'), dpi=140, bbox_inches='tight')
print(f"\nCorner plot 저장 완료: {os.path.join(aa_output_dir, 'fano_single_resonator_corner_v2.png')}")
