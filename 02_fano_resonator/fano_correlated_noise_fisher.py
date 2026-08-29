"""
fano_correlated_noise_fisher.py
=================================================================
[목적] STEP B에서 확인한 대로, 9개 resonator 전부 잔차가 강한
자기상관(lag-1 상관계수 0.83~0.998)을 보였습니다. 지금까지 쓴
피셔행렬 공식은 "각 데이터 포인트의 잡음이 서로 완전히 독립"이라고
가정하는데(대각행렬 형태의 잡음 공분산), 이 가정이 깨져 있으므로
공식 자체를 확장해야 합니다.

[용어 정리]
  일반화된 최소제곱(GLS, Generalized Least Squares): 잡음이 서로
    독립이 아닐 때(상관되어 있을 때) 쓰는, 최소제곱법의 확장판.
    기존 최소제곱(OLS)은 잡음 공분산이 "대각행렬 x sigma^2"라고
    가정하지만, GLS는 잡음 공분산 행렬 전체(비대각 성분 포함)를
    반영해서 파라미터와 그 오차를 다시 계산함.
  자기회귀(AR, AutoRegressive) 모델: "지금 값이 바로 이전 값에
    얼마나 의존하는지"로 상관된 잡음을 표현하는 가장 간단한 모델.
    AR(1) 모델: noise[i] = rho * noise[i-1] + white_noise[i]
    여기서 rho(로우, 상관계수)가 1에 가까울수록 "매우 천천히 변하는"
    잡음이 되고, 0이면 완전한 백색잡음이 됨. 오늘 잔차의 lag-1
    자기상관(0.83~0.998)이 바로 이 rho에 대한 직접적인 추정치입니다.
  화이트닝(whitening): 상관된 데이터를 수학적 변환을 통해 다시
    서로 독립인(백색잡음 같은) 형태로 바꾸는 표준적인 기법.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import json
from scipy.optimize import minimize

sys.path.insert(0, os.getcwd())
import fano_models
import fano_likelihood as likelihood
import fano_bayesian_toolkit as bt


# =========================================================
# STEP 0. 분석자 설정값 (aa_ 접두사)
# =========================================================
aa_upload_dir = '/mnt/user-data/uploads/'
aa_resonator_choice = 'resonator_7_powersweep_overcoupled.npz'
aa_power_slice_index = 0
aa_n_ports = 1.0
aa_phi_candidates = [-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5]
aa_mc_sample_size = 20000
aa_output_dir = '/home/claude/qubit_analysis_template/outputs/'
os.makedirs(aa_output_dir, exist_ok=True)

print("=" * 60)
print("분석자 설정값(aa_*) 요약:")
for k, v in list(globals().items()):
    if k.startswith("aa_") and not isinstance(v, list):
        print(f"  {k:24s} = {v}")
print("=" * 60)


# =========================================================
# STEP 1. 데이터 로드 및 (기존 방식) MAP 탐색 - 잔차/rho 추정용
# =========================================================
def load_fano_npz(filepath):
    raw = np.load(filepath, allow_pickle=True)
    amplitude, phase = raw['amplitude'], raw['phase']
    freq_hz, power_dbm = raw['frequency'], raw['power']
    s21 = amplitude * np.exp(1j * phase)
    settings_dict = json.loads(raw['settings'][0])
    cable_delay = settings_dict.get('vna', {}).get('port1_edel', None)
    return {'power_dbm': power_dbm, 'freq_hz': freq_hz, 's21': s21, 'cable_delay': cable_delay}


data = load_fano_npz(os.path.join(aa_upload_dir, aa_resonator_choice))
f_grid = data['freq_hz']
s21 = data['s21'][aa_power_slice_index, :]
delay = data['cable_delay']
mag = np.abs(s21)

param_names = ['fr', 'Ql', 'Qc', 'phi', 'a', 'alpha']
prior_bounds = {
    'fr': (f_grid.min(), f_grid.max()), 'Ql': (1.0, 1e8), 'Qc': (1.0, 1e8),
    'phi': (-np.pi, np.pi), 'a': (1e-8, 1.0), 'alpha': (-np.pi, np.pi),
}
log_prior = likelihood.make_uniform_log_prior(prior_bounds)
fixed_kwargs = {'delay': delay, 'n_ports': aa_n_ports}
log_probability = likelihood.make_log_probability_generic(
    fano_models.Sij, fixed_kwargs, param_names, log_prior, likelihood_type='gaussian'
)

fr_guess = f_grid[np.argmin(mag)]
half_level = (mag.max()+mag.min())/2
above_half = np.where(mag > half_level)[0]
fwhm_guess = f_grid[above_half[-1]]-f_grid[above_half[0]] if len(above_half)>1 else 1e6
Ql_guess = fr_guess/max(fwhm_guess, 1e3)

baseline_mask = mag > np.percentile(mag, 80)
sigma_baseline = np.std(np.real(s21[baseline_mask]))

best_theta, best_neg_lp = None, np.inf
for phi0 in aa_phi_candidates:
    theta0 = np.array([fr_guess, Ql_guess, Ql_guess*1.1, phi0, mag.max()-mag.min(), phi0])
    def neg_log_prob(theta):
        val = log_probability(theta, f_grid, s21, sigma_baseline)
        return -val if np.isfinite(val) else 1e10
    res = minimize(neg_log_prob, theta0, method='Nelder-Mead',
                    options={'maxiter': 20000, 'xatol': 1e-6, 'fatol': 1e-6})
    if res.fun < best_neg_lp:
        best_neg_lp = res.fun
        best_theta = res.x

popt = best_theta
model_best = fano_models.Sij(f_grid, *popt, delay, n_ports=aa_n_ports)
residual_real = np.real(s21 - model_best)

xc = residual_real - np.mean(residual_real)
rho = np.sum(xc[:-1]*xc[1:]) / np.sum(xc**2)
mad = np.median(np.abs(residual_real - np.median(residual_real)))
sigma_robust = mad * 1.4826

print(f"\nMAP 파라미터 (baseline sigma 기준):")
for i, name in enumerate(param_names):
    print(f"  {name}: {popt[i]:.6g}")
print(f"\n추정된 AR(1) 상관계수 rho = {rho:.4f}")
print(f"강건한(MAD) sigma = {sigma_robust:.6f}")


# =========================================================
# STEP 2. 화이트닝 방식으로 상관 잡음을 반영한 피셔행렬 계산
# =========================================================
def whiten(x, rho):
    """AR(1) 화이트닝 필터를 배열 x에 적용."""
    x_w = np.empty_like(x)
    x_w[0] = x[0] * np.sqrt(max(1 - rho**2, 1e-12))
    x_w[1:] = x[1:] - rho * x[:-1]
    return x_w


def numerical_derivative_local(model_func, theta, param_index, param_names,
                                  f_grid, fixed_kwargs, step_fraction=1e-6):
    theta_plus = np.array(theta, dtype=float).copy()
    theta_minus = np.array(theta, dtype=float).copy()
    h = step_fraction * max(abs(theta[param_index]), 1e-8)
    theta_plus[param_index] += h
    theta_minus[param_index] -= h
    kwargs_plus = dict(zip(param_names, theta_plus))
    kwargs_minus = dict(zip(param_names, theta_minus))
    model_plus = model_func(f_grid, **kwargs_plus, **fixed_kwargs)
    model_minus = model_func(f_grid, **kwargs_minus, **fixed_kwargs)
    return (model_plus - model_minus) / (2 * h)


def fisher_matrix_correlated(model_func, theta_best, param_names, f_grid,
                                sigma, rho, fixed_kwargs, step_fraction=1e-6):
    """AR(1) 상관 구조를 반영한 피셔행렬 (화이트닝 후 내적)."""
    ndim = len(theta_best)
    derivs = [
        numerical_derivative_local(model_func, theta_best, i, param_names,
                                      f_grid, fixed_kwargs, step_fraction)
        for i in range(ndim)
    ]
    derivs_whitened = []
    for d in derivs:
        d_real_w = whiten(np.real(d), rho) / sigma
        d_imag_w = whiten(np.imag(d), rho) / sigma
        derivs_whitened.append(d_real_w + 1j*d_imag_w)

    F = np.zeros((ndim, ndim))
    for i in range(ndim):
        for j in range(ndim):
            F[i, j] = (np.sum(np.real(derivs_whitened[i]) * np.real(derivs_whitened[j]))
                       + np.sum(np.imag(derivs_whitened[i]) * np.imag(derivs_whitened[j])))
    return F


F_independent = fisher_matrix_correlated(fano_models.Sij, popt, param_names, f_grid,
                                            sigma_robust, rho=0.0, fixed_kwargs=fixed_kwargs)
F_correlated = fisher_matrix_correlated(fano_models.Sij, popt, param_names, f_grid,
                                           sigma_robust, rho=rho, fixed_kwargs=fixed_kwargs)

cov_independent = np.linalg.inv(F_independent)
cov_correlated = np.linalg.inv(F_correlated)

print(f"\n[독립 가정 vs 상관 반영 - 파라미터별 1-sigma 오차 비교]")
sigma_indep = np.sqrt(np.abs(np.diag(cov_independent)))
sigma_corr = np.sqrt(np.abs(np.diag(cov_correlated)))
for i, name in enumerate(param_names):
    ratio = sigma_corr[i] / sigma_indep[i] if sigma_indep[i] > 0 else np.nan
    print(f"  {name}: 독립가정={sigma_indep[i]:.5g}, 상관반영={sigma_corr[i]:.5g}, "
          f"비율={ratio:.2f}배")


# =========================================================
# STEP 3. Qi 신뢰구간 비교
# =========================================================
def qi_credible_interval(popt, cov):
    mc = np.random.default_rng(0).multivariate_normal(popt, cov, size=aa_mc_sample_size)
    Qi_mc = 1.0/(1.0/mc[:,1] - 1.0/mc[:,2])
    valid = (Qi_mc>0) & (Qi_mc<1e7)
    return bt.credible_interval(Qi_mc[valid], level=0.68), mc

(Qi_lo_i, Qi_med_i, Qi_hi_i), mc_indep = qi_credible_interval(popt, cov_independent)
(Qi_lo_c, Qi_med_c, Qi_hi_c), mc_corr = qi_credible_interval(popt, cov_correlated)

print(f"\n[Qi 신뢰구간 최종 비교]")
print(f"  독립 가정(잘못된 가정): Qi={Qi_med_i:.1f} [{Qi_lo_i:.1f},{Qi_hi_i:.1f}] "
      f"(폭={Qi_hi_i-Qi_lo_i:.1f})")
print(f"  상관 반영(GLS, 더 정직함): Qi={Qi_med_c:.1f} [{Qi_lo_c:.1f},{Qi_hi_c:.1f}] "
      f"(폭={Qi_hi_c-Qi_lo_c:.1f})")
wider = (Qi_hi_c-Qi_lo_c) > (Qi_hi_i-Qi_lo_i)
print(f"  -> 상관을 반영하면 신뢰구간이 {'넓어짐' if wider else '좁아짐'} "
      f"(상관을 무시하면 '같은 정보를 여러 번 센' 것과 비슷해서, 보통 실제보다")
print(f"     신뢰구간이 부당하게 좁게(과신하게) 나오는 경향이 있습니다)")


# =========================================================
# STEP 4. 시각화
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(mc_indep[:,1], bins=60, alpha=0.5, color='tab:orange', density=True,
             label='독립 가정 (과신 위험)')
axes[0].hist(mc_corr[:,1], bins=60, alpha=0.5, color='tab:blue', density=True,
             label='상관 반영 (GLS)')
axes[0].set_xlabel('Ql')
axes[0].set_ylabel('밀도')
axes[0].set_title('Ql posterior: 독립 가정 vs 상관 반영')
axes[0].legend(fontsize=8)

qi_indep = 1.0/(1.0/mc_indep[:,1]-1.0/mc_indep[:,2])
qi_corr = 1.0/(1.0/mc_corr[:,1]-1.0/mc_corr[:,2])
valid_i = (qi_indep>0)&(qi_indep<1e6)
valid_c = (qi_corr>0)&(qi_corr<1e6)
axes[1].hist(qi_indep[valid_i], bins=60, alpha=0.5, color='tab:orange', density=True,
             label='독립 가정')
axes[1].hist(qi_corr[valid_c], bins=60, alpha=0.5, color='tab:blue', density=True,
             label='상관 반영 (GLS)')
axes[1].set_xlabel('Qi')
axes[1].set_ylabel('밀도')
axes[1].set_title('Qi posterior: 독립 가정 vs 상관 반영')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(aa_output_dir, 'fano_correlated_noise_comparison.png'), dpi=140, bbox_inches='tight')
print(f"\n비교 그래프 저장 완료: "
      f"{os.path.join(aa_output_dir, 'fano_correlated_noise_comparison.png')}")
