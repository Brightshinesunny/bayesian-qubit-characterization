"""
fano_all_resonators_v3_final.py
=================================================================
[v3 - 최종 통합 버전] 오늘 전체 흐름의 결론을 하나로 합칩니다:

  STEP B(잡음 진단)  : baseline sigma -> 잔차 진단 -> MAD 강건 sigma
  STEP A(상관 잡음)  : lag-1 자기상관을 AR(1) 공분산으로 반영(GLS)
  [신규] circle fit을 초기값으로 우선 활용 - R4에서 발견한 것처럼,
         FWHM 기반 초기값 추정이 가끔 완전히 실패해서 최적화가
         prior 경계(Ql=1 또는 Qc=1)로 도망가는 문제가 있었음.
         circle fit(대수적, 반복 없음)은 이런 국소최적점 문제 자체가
         없으므로, "가장 믿을 만한 초기값 후보 하나"로 항상 같이
         시도하고, 기존 phi 스캔 후보들과 비교해 -log_prob가 가장
         낮은 것을 채택.

이렇게 3가지 개선을 모두 합친 것이 오늘 도달한 "이런 종류의 실측
데이터를 만났을 때 표준적으로 거쳐야 할 절차"입니다.
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
aa_resonator_files = [
    'resonator_1_powersweep_overcoupled.npz', 'resonator_2_powersweep_overcoupled.npz',
    'resonator_3_powersweep_overcoupled.npz', 'resonator_4_powersweep_overcoupled.npz',
    'resonator_5_powersweep_overcoupled.npz', 'resonator_6_powersweep_overcoupled.npz',
    'resonator_7_powersweep_overcoupled.npz', 'resonator_8_powersweep_overcoupled.npz',
    'resonator_9_powersweep_overcoupled.npz',
]
aa_power_slice_index = 0
aa_n_ports = 1.0
aa_isolation_db = 15
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


def ar1_precision_sparse(n, sigma, rho):
    """
    AR(1) 공분산 행렬의 역행렬(정밀도 행렬)을 닫힌 형태 공식으로
    구성 (fano_correlated_noise.py에서 작은 예제로 numpy 직접 역행렬과
    최대 오차 3e-10 수준으로 일치함을 이미 검증한 공식).
    """
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
    return residual_real @ (P @ residual_real) + residual_imag @ (P @ residual_imag)


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

    # --- [신규] circle fit으로 안전한 초기값 후보 확보 ---
    try:
        cf_result = fano_models.autofit(f_grid, s21, n_ports=aa_n_ports,
                                          fixed_delay=delay, isolation=aa_isolation_db)
        circle_fit_theta = np.array([cf_result['fr'], cf_result['Ql'], cf_result['Qc'],
                                       cf_result['phi'], cf_result['a'], cf_result['alpha']])
    except Exception:
        circle_fit_theta = None

    candidates_theta0 = [np.array([fr_guess, Ql_guess, Ql_guess*1.1, phi0, mag.max()-mag.min(), phi0])
                          for phi0 in aa_phi_candidates]
    if circle_fit_theta is not None:
        candidates_theta0.append(circle_fit_theta)
            # [신규] circle fit 결과도 초기값 후보 목록에 추가 - R4
            # 사례에서 FWHM 기반 초기값 전부가 실패했을 때 이 후보
            # 하나가 정상적인 결과로 이끌어준 것을 확인했음.

    # --- 1차: baseline sigma로 MAP (모든 후보 시도) ---
    baseline_mask = mag > np.percentile(mag, 80)
    sigma_baseline = np.std(np.real(s21[baseline_mask]))

    def find_map(sigma, theta0_list):
        best_theta, best_neg_lp = None, np.inf
        for theta0 in theta0_list:
            def neg_log_prob(theta):
                val = log_probability(theta, f_grid, s21, sigma)
                return -val if np.isfinite(val) else 1e10
            res = minimize(neg_log_prob, theta0, method='Nelder-Mead',
                            options={'maxiter': 20000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_neg_lp:
                best_neg_lp = res.fun
                best_theta = res.x
        return best_theta

    popt1 = find_map(sigma_baseline, candidates_theta0)

    # --- 잔차 진단 ---
    model1 = fano_models.Sij(f_grid, *popt1, delay, n_ports=aa_n_ports)
    residual_real = np.real(s21 - model1)
    mad = np.median(np.abs(residual_real - np.median(residual_real)))
    sigma_robust = mad * 1.4826
    rho = lag1_autocorrelation(residual_real)

    # --- 3차: AR(1) 상관잡음까지 반영한 최종 MAP ---
    def neg_log_posterior_gls(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return 1e10
        model = fano_models.Sij(f_grid, *theta, delay, n_ports=aa_n_ports)
        res_r = np.real(s21 - model)
        res_i = np.imag(s21 - model)
        chi2 = gls_chi2(res_r, res_i, sigma_robust, rho)
        return 0.5*chi2 - lp

    best_theta3, best_val3 = None, np.inf
    all_candidates_v2 = candidates_theta0 + [popt1]
        # popt1(1차 결과)도 3차 최적화의 출발점 후보로 추가 - 이미
        # 좋은 답 근처에서 다시 정밀화하는 효과.
    for theta0 in all_candidates_v2:
        res = minimize(neg_log_posterior_gls, theta0, method='Nelder-Mead',
                        options={'maxiter': 20000, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < best_val3:
            best_val3 = res.fun
            best_theta3 = res.x
    popt3 = best_theta3

    # --- AR(1) 반영 피셔행렬로 Qi 신뢰구간 계산 ---
    def gls_fisher_matrix(popt):
        n = len(popt)
        P = ar1_precision_sparse(len(f_grid), sigma_robust, rho)
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
                F[i,j] = (np.real(derivs[i])@(P@np.real(derivs[j]))
                          + np.imag(derivs[i])@(P@np.imag(derivs[j])))
        return F

    F3 = gls_fisher_matrix(popt3)
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
        if np.sum(valid3) >= 10:
            Qi_lo3, Qi_med3, Qi_hi3 = bt.credible_interval(Qi_mc3[valid3], level=0.68)
        else:
            Qi_lo3, Qi_med3, Qi_hi3 = np.nan, np.nan, np.nan
    else:
        Qi_lo3, Qi_med3, Qi_hi3 = np.nan, np.nan, np.nan

    width3 = Qi_hi3 - Qi_lo3 if cov3_ok else np.nan
    Ql_at_boundary = (popt3[1] <= 1.01) or (popt3[2] <= 1.01)

    results_summary.append({
        'resonator': filename.split('_powersweep')[0],
        'circle_fit_ok': circle_fit_theta is not None,
        'rho': rho, 'sigma_robust': sigma_robust,
        'Ql': popt3[1], 'Qc': popt3[2],
        'boundary_flag': Ql_at_boundary,
        'Qi_med': Qi_med3, 'Qi_width': width3,
    })

    flag_str = "⚠️ 경계 문제" if Ql_at_boundary else "정상"
    print(f"{filename}: Ql={popt3[1]:.1f}, Qc={popt3[2]:.1f} [{flag_str}], "
          f"Qi={Qi_med3:.1f}, 폭={width3:.1f}")


# =========================================================
# STEP 2. 최종 요약표
# =========================================================
print("\n" + "=" * 90)
print(f"{'Resonator':<14} {'Ql':>10} {'Qc':>10} {'경계문제':>10} {'Qi(AR1반영)':>14} {'Qi폭':>12}")
print("=" * 90)
for r in results_summary:
    flag = "예" if r['boundary_flag'] else "아니오"
    print(f"{r['resonator']:<14} {r['Ql']:>10.1f} {r['Qc']:>10.1f} {flag:>10} "
          f"{r['Qi_med']:>14.1f} {r['Qi_width']:>12.1f}")

n_boundary = sum(1 for r in results_summary if r['boundary_flag'])
print("-" * 90)
print(f"\ncircle fit 초기값 추가 후에도 경계 문제가 남은 resonator: {n_boundary}/{len(results_summary)}")
print("(0이면 오늘 발견한 초기값 문제가 완전히 해결된 것)")

fig, ax = plt.subplots(figsize=(10, 5))
names = [r['resonator'].replace('resonator_', 'R') for r in results_summary]
colors = ['tab:red' if r['boundary_flag'] else 'tab:blue' for r in results_summary]
medians = [r['Qi_med'] for r in results_summary]
widths = [r['Qi_width'] for r in results_summary]
ax.errorbar(range(len(names)), medians, yerr=[np.array(widths)/2], fmt='o', ecolor='gray', capsize=4, ms=0)
for i, (m, c) in enumerate(zip(medians, colors)):
    ax.plot(i, m, 'o', color=c, ms=10)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names)
ax.set_yscale('log')
ax.set_ylabel('Qi (AR(1) 상관잡음 반영, 중앙값 ± 68%)')
ax.set_title('v3 최종 결과: circle fit 초기값 + MAD sigma + AR(1) 상관잡음 반영\n'
              '(빨간 점 = 여전히 prior 경계 문제가 있는 경우)')
plt.tight_layout()
plt.savefig(os.path.join(aa_output_dir, 'fano_all_resonators_v3_final.png'), dpi=140, bbox_inches='tight')
print(f"\n최종 비교 그래프 저장 완료: "
      f"{os.path.join(aa_output_dir, 'fano_all_resonators_v3_final.png')}")
