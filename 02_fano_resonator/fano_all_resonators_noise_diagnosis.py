"""
fano_all_resonators_noise_diagnosis.py
=================================================================
[목적] fano_noise_diagnosis.py에서 resonator_7 하나로 확인한 절차
(MAP 탐색 -> 잔차 진단 -> MAD 기반 강건한 sigma로 재탐색 -> Qi 신뢰
구간 비교)를, 9개 resonator 전부에 자동으로 반복 적용합니다.

[이 스크립트의 위치 - "0단계" 표준화]
"피팅 전에 baseline만 보고 잡음을 대충 추정하는 것"이 아니라,
"일단 피팅해보고, 그 잔차를 진단해서 잡음을 다시 추정하고, 필요하면
재피팅하는" 이 절차 자체를 모든 데이터 분석의 표준 0단계로 삼으려는
것이 목적입니다. resonator_7에서 우연히 나온 특이 현상인지, 아니면
이 데이터셋 전체(어쩌면 이 VNA/실험 세팅 전체)에 공통적으로 나타나는
현상인지 확인하는 것이 이번 실행의 핵심 질문입니다.
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
import fano_fisher_matrix as fm
import fano_bayesian_toolkit as bt


# =========================================================
# STEP 0. 분석자 설정값 (aa_ 접두사)
# =========================================================
aa_upload_dir = '/mnt/user-data/uploads/'
aa_resonator_files = [
    'resonator_1_powersweep_overcoupled.npz', 'resonator_2_powersweep_overcoupled.npz',
    'resonator_3_powersweep_overcoupled.npz', 'resonator_4_powersweep_overcoupled.npz',
    'resonator_5_powersweep_overcoupled.npz', 'resonator_6_powersweep_overcoupled.npz',
    'resonator_7_powersweep_overcoupled.npz', 'resonator_8_powersweep_overcoupled.npz',
    'resonator_9_powersweep_overcoupled.npz',
]
aa_power_slice_index = 0
aa_n_ports = 1.0
aa_step_fraction = 1e-6
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


def find_map(f_grid, s21, sigma, param_names, log_probability, fr_guess, Ql_guess, mag, phi_candidates):
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


# =========================================================
# STEP 1. 9개 resonator 전부 반복 처리
# =========================================================
param_names = ['fr', 'Ql', 'Qc', 'phi', 'a', 'alpha']
results_summary = []

for filename in aa_resonator_files:
    full_path = os.path.join(aa_upload_dir, filename)
    if not os.path.exists(full_path):
        print(f"[건너뜀] 파일 없음: {filename}")
        continue

    data = load_fano_npz(full_path)
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
    log_probability = likelihood.make_log_probability_generic(
        fano_models.Sij, fixed_kwargs, param_names, log_prior, likelihood_type='gaussian'
    )

    fr_guess = f_grid[np.argmin(mag)]
    half_level = (mag.max()+mag.min())/2
    above_half = np.where(mag > half_level)[0]
    fwhm_guess = f_grid[above_half[-1]]-f_grid[above_half[0]] if len(above_half)>1 else 1e6
    Ql_guess = fr_guess/max(fwhm_guess, 1e3)

    # --- 1차: baseline sigma로 MAP 탐색 ---
    baseline_mask = mag > np.percentile(mag, 80)
    sigma_baseline = np.std(np.real(s21[baseline_mask]))
    popt1 = find_map(f_grid, s21, sigma_baseline, param_names, log_probability,
                       fr_guess, Ql_guess, mag, aa_phi_candidates)

    # --- 잔차 진단 ---
    model1 = fano_models.Sij(f_grid, *popt1, delay, n_ports=aa_n_ports)
    residual_real = np.real(s21 - model1)
    mad = np.median(np.abs(residual_real - np.median(residual_real)))
    sigma_robust = mad * 1.4826
    ac1 = lag1_autocorrelation(residual_real)

    # --- 2차: 강건한 sigma로 MAP 재탐색 ---
    popt2 = find_map(f_grid, s21, sigma_robust, param_names, log_probability,
                       fr_guess, Ql_guess, mag, aa_phi_candidates)

    def qi_credible(popt, sigma):
        F = fm.fisher_information_matrix(fano_models.Sij, popt, param_names, f_grid,
                                            sigma, fixed_kwargs, step_fraction=aa_step_fraction)
        cov = fm.covariance_from_fisher(F)
        mc = np.random.default_rng(0).multivariate_normal(popt, cov, size=aa_mc_sample_size)
        Qi_mc = 1.0/(1.0/mc[:,1] - 1.0/mc[:,2])
        valid = (Qi_mc>0) & (Qi_mc<1e7)
        if np.sum(valid) < 10:
            return np.nan, np.nan, np.nan
        return bt.credible_interval(Qi_mc[valid], level=0.68)

    Qi_lo1, Qi_med1, Qi_hi1 = qi_credible(popt1, sigma_baseline)
    Qi_lo2, Qi_med2, Qi_hi2 = qi_credible(popt2, sigma_robust)

    width1 = Qi_hi1 - Qi_lo1
    width2 = Qi_hi2 - Qi_lo2
    narrowing_factor = width1/width2 if width2 > 0 else np.nan

    results_summary.append({
        'resonator': filename.split('_powersweep')[0],
        'sigma_baseline': sigma_baseline, 'sigma_robust': sigma_robust,
        'lag1_autocorr': ac1,
        'Qi_med1': Qi_med1, 'width1': width1,
        'Qi_med2': Qi_med2, 'width2': width2,
        'narrowing_factor': narrowing_factor,
    })

    print(f"{filename}: sigma(baseline/robust)={sigma_baseline:.5f}/{sigma_robust:.5f}, "
          f"lag1_autocorr={ac1:.3f}, Qi폭 {width1:.0f}->{width2:.0f} "
          f"({narrowing_factor:.2f}배 좁아짐)")


# =========================================================
# STEP 2. 요약표 및 시각화
# =========================================================
print("\n" + "=" * 100)
print(f"{'Resonator':<14} {'lag1 자기상관':>14} {'sigma(base/robust)':>22} "
      f"{'Qi폭(1차->2차)':>18} {'좁아진배수':>10}")
print("=" * 100)
for r in results_summary:
    sigma_str = f"{r['sigma_baseline']:.4f}/{r['sigma_robust']:.4f}"
    width_str = f"{r['width1']:.0f}->{r['width2']:.0f}"
    print(f"{r['resonator']:<14} {r['lag1_autocorr']:>14.3f} {sigma_str:>22} "
          f"{width_str:>18} {r['narrowing_factor']:>10.2f}")

n_strong_autocorr = sum(1 for r in results_summary if abs(r['lag1_autocorr']) > 0.3)
mean_narrowing = np.nanmean([r['narrowing_factor'] for r in results_summary])
print("-" * 100)
print(f"\nlag-1 자기상관이 0.3을 넘는(색깔있는 잡음 의심) resonator: "
      f"{n_strong_autocorr}/{len(results_summary)}")
print(f"평균 신뢰구간 좁아짐 배수: {mean_narrowing:.2f}배")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
names = [r['resonator'].replace('resonator_', 'R') for r in results_summary]

axes[0].bar(names, [r['lag1_autocorr'] for r in results_summary], color='tab:purple')
axes[0].axhline(0.3, color='red', ls='--', lw=1, label='의심 기준선(0.3)')
axes[0].axhline(-0.3, color='red', ls='--', lw=1)
axes[0].set_ylabel('lag-1 자기상관')
axes[0].set_title('9개 resonator의 잔차 자기상관\n(모두 색깔있는 잡음을 갖는지 확인)')
axes[0].legend(fontsize=8)

axes[1].bar(names, [r['narrowing_factor'] for r in results_summary], color='tab:green')
axes[1].axhline(1.0, color='gray', ls='--', lw=1, label='변화 없음(1배)')
axes[1].set_ylabel('신뢰구간 좁아짐 배수')
axes[1].set_title('강건한 sigma 적용 후 Qi 신뢰구간이 얼마나 좁아졌는지')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(aa_output_dir, 'fano_all_resonators_noise_diagnosis.png'), dpi=140, bbox_inches='tight')
print(f"\n비교 그래프 저장 완료: "
      f"{os.path.join(aa_output_dir, 'fano_all_resonators_noise_diagnosis.png')}")
