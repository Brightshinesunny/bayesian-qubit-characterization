"""
fano_correlated_noise.py
=================================================================
[STEP A] 지금까지(STEP B까지) "잔차가 자기상관을 갖는다"는 걸
진단만 하고, 실제 계산에서는 여전히 "잡음이 서로 독립"이라고
가정한 채(대각 공분산) MAD로 스케일만 조정했습니다. 이번엔 그
자기상관 자체를 공분산 행렬에 정식으로 반영합니다.

[수학적 배경 - 일반화 최소제곱(GLS)]
지금까지 쓴 카이제곱(chi-square)은:
    chi2 = sum_i (residual_i / sigma)^2  =  residual^T * (1/sigma^2 * I) * residual
"1/sigma^2 * I"는 "모든 점이 독립이고 크기가 같다"는 가장 단순한
가중치 행렬(대각 행렬)입니다. 잡음이 서로 얽혀있을 때는 이 대신
"공분산 행렬의 역행렬(정밀도 행렬) Sigma^-1"을 통째로 써야 합니다:
    chi2_GLS = residual^T * Sigma^-1 * residual
이렇게 확장한 최소제곱법을 일반화 최소제곱(GLS, Generalized Least
Squares)이라 부릅니다. Sigma가 대각 행렬(독립)이면 GLS는 정확히
지금까지 쓰던 일반 최소제곱과 같아집니다 - 즉 오늘 확장은 기존
방법을 "포함하는" 더 넓은 개념입니다.

[AR(1) 잡음 모델과 그 역행렬의 닫힌 형태 공식]
잔차가 AR(1)(1차 자기회귀) 과정을 따른다고 가정: 이웃한 두 점의
상관계수가 rho(lag-1 자기상관, STEP B에서 이미 측정함), k칸
떨어진 두 점의 상관계수는 rho^k로 지수적으로 감소.
    Sigma_ij = sigma^2 * rho^|i-j|
이 행렬을 그냥 역행렬 계산(numpy.linalg.inv)하면 1001x1001 크기라
느리고(O(n^3)), 수치적으로도 불안정해질 수 있습니다. 다행히 AR(1)의
역행렬(정밀도 행렬)은 "삼중대각(tridiagonal, 대각선 근처만 값이
있고 나머지는 0)" 행렬이라는 닫힌 형태 공식이 있어서, O(n) 만에
정확하게 계산할 수 있습니다(이 스크립트를 만들기 전 작은 예제로
numpy 직접 역행렬과 정확히 일치함을 미리 검증함 - 최대 오차 3e-10,
부동소수점 반올림 오차 수준).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import json
from scipy.optimize import minimize
from scipy.sparse import diags

sys.path.insert(0, os.getcwd())
import fano_models
import fano_likelihood as likelihood
import fano_bayesian_toolkit as bt


# =========================================================
# STEP 0. 분석자 설정값 (aa_ 접두사)
# =========================================================
aa_upload_dir = '/mnt/user-data/uploads/'
aa_resonator_choices = [
    'resonator_7_powersweep_overcoupled.npz',
    'resonator_4_powersweep_overcoupled.npz',
]
aa_power_slice_index = 0
aa_n_ports = 1.0
aa_phi_candidates = [-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5]
aa_mc_sample_size = 20000
aa_step_fraction = 1e-6
aa_output_dir = '/home/claude/qubit_analysis_template/outputs/'
os.makedirs(aa_output_dir, exist_ok=True)

print("=" * 60)
print("분석자 설정값(aa_*) 요약:")
for k, v in list(globals().items()):
    if k.startswith("aa_") and not isinstance(v, list):
        print(f"  {k:24s} = {v}")
print("=" * 60)


def load_fano_npz(filepath):
    raw = np.load(filepath, allow_pickle=True)
    amplitude, phase = raw['amplitude'], raw['phase']
    freq_hz, power_dbm = raw['frequency'], raw['power']
    s21 = amplitude * np.exp(1j * phase)
    settings_dict = json.loads(raw['settings'][0])
    cable_delay = settings_dict.get('vna', {}).get('port1_edel', None)
    return {'power_dbm': power_dbm, 'freq_hz': freq_hz, 's21': s21, 'cable_delay': cable_delay}


def lag1_autocorrelation(x):
    xc = x - np.mean(x)
    denom = np.sum(xc**2)
    return np.sum(xc[:-1]*xc[1:]) / denom if denom > 0 else 0.0


def ar1_precision_sparse(n, sigma, rho):
    factor = 1.0 / (sigma**2 * (1 - rho**2))
    main_diag = np.full(n, 1 + rho**2)
    main_diag[0] = 1.0
    main_diag[-1] = 1.0
    off_diag = np.full(n-1, -rho)
    P = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format='csc')
    return factor * P


def gls_chi2(residual_real, residual_imag, sigma, rho):
    n = len(residual_real)
    P = ar1_precision_sparse(n, sigma, rho)
    chi2_real = residual_real @ (P @ residual_real)
    chi2_imag = residual_imag @ (P @ residual_imag)
    return chi2_real + chi2_imag


param_names = ['fr', 'Ql', 'Qc', 'phi', 'a', 'alpha']

for resonator_choice in aa_resonator_choices:
    print(f"\n{'='*70}\n분석 대상: {resonator_choice}\n{'='*70}")

    data = load_fano_npz(os.path.join(aa_upload_dir, resonator_choice))
    f_grid = data['freq_hz']
    s21 = data['s21'][aa_power_slice_index, :]
    delay = data['cable_delay']
    mag = np.abs(s21)

    prior_bounds = {
        'fr': (f_grid.min(), f_grid.max()), 'Ql': (1.0, 1e8), 'Qc': (1.0, 1e8),
        'phi': (-np.pi, np.pi), 'a': (1e-8, 1.0), 'alpha': (-np.pi, np.pi),
    }
    log_prior = likelihood.make_uniform_log_prior(prior_bounds)
    fixed_kwargs = {'delay': delay, 'n_ports': aa_n_ports}

    fr_guess = f_grid[np.argmin(mag)]
    half_level = (mag.max()+mag.min())/2
    above_half = np.where(mag > half_level)[0]
    fwhm_guess = f_grid[above_half[-1]]-f_grid[above_half[0]] if len(above_half)>1 else 1e6
    Ql_guess = fr_guess/max(fwhm_guess, 1e3)

    baseline_mask = mag > np.percentile(mag, 80)
    sigma_baseline = np.std(np.real(s21[baseline_mask]))

    log_probability = likelihood.make_log_probability_generic(
        fano_models.Sij, fixed_kwargs, param_names, log_prior, likelihood_type='gaussian'
    )

    def find_map(sigma, phi_candidates):
        best_theta, best_neg_lp = None, np.inf
        for phi0 in phi_candidates:
            theta0 = np.array([fr_guess, Ql_guess, Ql_guess*1.1, phi0, mag.max()-mag.min(), phi0])
            def neg_log_prob(theta):
                val = log_probability(theta, f_grid, s21, sigma)
                return -val if np.isfinite(val) else 1e10
            res = minimize(neg_log_prob, theta0, method='Nelder-Mead',
                            options={'maxiter': 20000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_neg_lp:
                best_neg_lp = res.fun
                best_theta = res.x
        return best_theta

    popt1 = find_map(sigma_baseline, aa_phi_candidates)
    model1 = fano_models.Sij(f_grid, *popt1, delay, n_ports=aa_n_ports)
    residual_real = np.real(s21 - model1)
    residual_imag = np.imag(s21 - model1)

    mad = np.median(np.abs(residual_real - np.median(residual_real)))
    sigma_robust = mad * 1.4826
    rho = lag1_autocorrelation(residual_real)
    print(f"\n1차(baseline) 피팅의 잔차 진단: sigma_robust={sigma_robust:.6f}, rho(lag1)={rho:.4f}")

    popt2 = find_map(sigma_robust, aa_phi_candidates)

    def neg_log_posterior_gls(theta, sigma, rho):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return 1e10
        model = fano_models.Sij(f_grid, *theta, delay, n_ports=aa_n_ports)
        res_r = np.real(s21 - model)
        res_i = np.imag(s21 - model)
        chi2 = gls_chi2(res_r, res_i, sigma, rho)
        return 0.5*chi2 - lp

    best_theta3, best_val3 = None, np.inf
    for phi0 in aa_phi_candidates:
        theta0 = np.array([fr_guess, Ql_guess, Ql_guess*1.1, phi0, mag.max()-mag.min(), phi0])
        res = minimize(neg_log_posterior_gls, theta0, args=(sigma_robust, rho),
                        method='Nelder-Mead', options={'maxiter': 20000, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < best_val3:
            best_val3 = res.fun
            best_theta3 = res.x
    popt3 = best_theta3

    print(f"\n[파라미터 비교: 1차(독립,baseline) vs 2차(독립,MAD) vs 3차(AR(1) 상관반영)]")
    for i, name in enumerate(param_names):
        print(f"  {name:6s}: {popt1[i]:>14.6g}  |  {popt2[i]:>14.6g}  |  {popt3[i]:>14.6g}")

    def gls_fisher_matrix(popt, sigma, rho):
        n = len(popt)
        P = ar1_precision_sparse(len(f_grid), sigma, rho)
        derivs = []
        for i in range(n):
            theta_p, theta_m = popt.copy(), popt.copy()
            h = 1e-6 * max(abs(popt[i]), 1e-8)
            theta_p[i] += h; theta_m[i] -= h
            m_p = fano_models.Sij(f_grid, *theta_p, delay, n_ports=aa_n_ports)
            m_m = fano_models.Sij(f_grid, *theta_m, delay, n_ports=aa_n_ports)
            derivs.append((m_p - m_m) / (2*h))
        F = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                real_term = np.real(derivs[i]) @ (P @ np.real(derivs[j]))
                imag_term = np.imag(derivs[i]) @ (P @ np.imag(derivs[j]))
                F[i, j] = real_term + imag_term
        return F

    F3 = gls_fisher_matrix(popt3, sigma_robust, rho)
    try:
        cov3 = np.linalg.inv(F3)
        eigvals3 = np.linalg.eigvalsh(cov3)
        cov3_ok = eigvals3.min() > 0
    except np.linalg.LinAlgError:
        cov3_ok = False

    if cov3_ok:
        mc3 = np.random.default_rng(0).multivariate_normal(popt3, cov3, size=aa_mc_sample_size)
        Qi_mc3 = 1.0/(1.0/mc3[:,1] - 1.0/mc3[:,2])
        valid3 = (Qi_mc3>0) & (Qi_mc3<1e7)
        Qi_lo3, Qi_med3, Qi_hi3 = bt.credible_interval(Qi_mc3[valid3], level=0.68)
        print(f"\n3차(AR(1) 상관반영) Qi: {Qi_med3:.1f} [{Qi_lo3:.1f}, {Qi_hi3:.1f}] "
              f"(폭={Qi_hi3-Qi_lo3:.1f})")
    else:
        print(f"\n3차(AR(1) 상관반영) 공분산 계산 실패 - 최소 고유값이 음수")
