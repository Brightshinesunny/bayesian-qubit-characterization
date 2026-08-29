"""
fano_noise_model_selection.py
=================================================================
[사용자 제안 반영 + 개선] "이상치 5% 제거 -> 남은 잔차로 rho 측정
-> rho에 따라 AR(1) 또는 1차 차분"이라는 제안을 실제로 검증했습니다.

[검증 중 발견한 것 - 결정 기준 개선]
lag-1 rho만 보고 판단하면(예: rho>0.9면 차분) 놓치는 경우가 있었습니다.
resonator_7의 lag-1 rho는 0.84로 0.9를 못 넘지만, 실제로는 lag=29
까지도 rho가 0.75로 거의 안 줄어드는 "거의 완벽한 랜덤워크(단위근)"
였습니다 - AR(1)이었다면 lag=29에서 0.84^29≈0.006까지 떨어져야
하는데 전혀 그렇지 않았습니다. 그래서 이 스크립트는 "lag-1 rho"
대신 "여러 lag에 걸친 자기상관이 AR(1) 예측과 얼마나 다른지"를
직접 비교해서 모델을 선택합니다 - 훨씬 엄밀한 기준입니다.

[결정 절차]
  1. baseline sigma + circle fit 초기값으로 1차 MAP 피팅
  2. |잔차| 상위 aa_outlier_trim_pct% 를 이상치로 제거
  3. 남은 잔차로 lag-1 rho 추정
  4. AR(1) 모델이 lag=L(예: 20)에서 예측하는 자기상관(rho^L)과
     실제 관측된 lag=L 자기상관을 비교
  5. 실제 값이 AR(1) 예측보다 훨씬 크면(장거리 지속성이 강하면)
     -> 1차 차분 사용 / 아니면 -> AR(1) 사용
  6. 어느 쪽을 선택했든, 최종 방식 적용 후 잔차의 자기상관이 실제로
     충분히 작아졌는지(백색잡음에 가까워졌는지) 다시 검증해서
     "선택이 실제로 맞았는지"까지 확인
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
aa_outlier_trim_pct = 5.0
    # [사용자 제안] 잔차 절댓값 기준 상위 5%를 이상치로 제거.
aa_check_lag = 20
    # [신규] "장거리 지속성"을 확인할 기준 lag. AR(1) 모델의 예측값과
    # 실제 관측값을 이 lag에서 비교해 모델을 선택.
aa_long_range_ratio_threshold = 5.0
    # [신규] 실제 관측된 lag=aa_check_lag 자기상관이, AR(1) 모델이
    # 예측하는 값보다 이 배수 이상 크면 "AR(1)로는 설명 안 되는 장거리
    # 지속성이 있다"고 판단해 1차 차분을 선택.
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


def autocorr_at_lag(x, lag):
    xc = x - np.mean(x)
    denom = np.sum(xc**2)
    return np.sum(xc[:-lag]*xc[lag:]) / denom if denom > 0 else 0.0


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
    return residual_real @ (P @ residual_real) + residual_imag @ (P @ residual_imag)


param_names = ['fr', 'Ql', 'Qc', 'phi', 'a', 'alpha']
results_summary = []

for filename in aa_resonator_files:
    full_path = os.path.join(aa_upload_dir, filename)
    if not os.path.exists(full_path):
        print(f"[건너뜀] 파일 없음: {filename}")
        continue

    print(f"\n{'='*70}\n{filename}\n{'='*70}")

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

    try:
        cf_result = fano_models.autofit(f_grid, s21, n_ports=aa_n_ports,
                                          fixed_delay=delay, isolation=aa_isolation_db)
        circle_fit_theta = np.array([cf_result['fr'], cf_result['Ql'], cf_result['Qc'],
                                       cf_result['phi'], cf_result['a'], cf_result['alpha']])
    except Exception:
        circle_fit_theta = None

    candidates = [np.array([fr_guess, Ql_guess, Ql_guess*1.1, phi0, mag.max()-mag.min(), phi0])
                  for phi0 in aa_phi_candidates]
    if circle_fit_theta is not None:
        candidates.append(circle_fit_theta)

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

    popt1 = find_map(sigma_baseline, candidates)
    model1 = fano_models.Sij(f_grid, *popt1, delay, n_ports=aa_n_ports)
    residual_real = np.real(s21 - model1)

    abs_res = np.abs(residual_real)
    thresh = np.percentile(abs_res, 100 - aa_outlier_trim_pct)
    keep_mask = abs_res <= thresh
    residual_trimmed = residual_real[keep_mask]
    n_outliers = np.sum(~keep_mask)

    rho = autocorr_at_lag(residual_trimmed, 1)
    observed_long_range = autocorr_at_lag(residual_trimmed, aa_check_lag)
    ar1_predicted_long_range = rho ** aa_check_lag
    long_range_ratio = (abs(observed_long_range) / max(abs(ar1_predicted_long_range), 1e-10))

    use_differencing = long_range_ratio > aa_long_range_ratio_threshold
    method_name = "1차차분" if use_differencing else "AR(1)"
    print(f"이상치 {n_outliers}개 제거, rho(lag1)={rho:.4f}")
    print(f"  lag={aa_check_lag}: 실제={observed_long_range:.4f}, AR(1)예측={ar1_predicted_long_range:.6f}, "
          f"비율={long_range_ratio:.1f}배")
    print(f"  -> 선택된 방식: {method_name}")

    if use_differencing:
        s21_diff = np.diff(s21)
        sigma_diff_est = np.median(np.abs(np.diff(residual_real) - np.median(np.diff(residual_real)))) * 1.4826

        def neg_log_posterior_diff(theta):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return 1e10
            model = fano_models.Sij(f_grid, *theta, delay, n_ports=aa_n_ports)
            model_diff = np.diff(model)
            res_diff = s21_diff - model_diff
            chi2 = np.sum((np.real(res_diff)/sigma_diff_est)**2 + (np.imag(res_diff)/sigma_diff_est)**2)
            return 0.5*chi2 - lp

        best_theta_final, best_val = None, np.inf
        for theta0 in candidates + [popt1]:
            res = minimize(neg_log_posterior_diff, theta0, method='Nelder-Mead',
                            options={'maxiter': 20000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_theta_final = res.x
        popt_final = best_theta_final

        model_final = fano_models.Sij(f_grid, *popt_final, delay, n_ports=aa_n_ports)
        final_residual_diff = np.real(s21_diff - np.diff(model_final))
        validation_rho = autocorr_at_lag(final_residual_diff, 1)
        print(f"  [검증] 차분 후 최종 잔차의 lag-1 자기상관: {validation_rho:.4f}")

        n = len(popt_final)
        derivs = []
        for i in range(n):
            tp, tm = popt_final.copy(), popt_final.copy()
            h = 1e-6*max(abs(popt_final[i]), 1e-8)
            tp[i]+=h; tm[i]-=h
            mp = np.diff(fano_models.Sij(f_grid, *tp, delay, n_ports=aa_n_ports))
            mm = np.diff(fano_models.Sij(f_grid, *tm, delay, n_ports=aa_n_ports))
            derivs.append((mp-mm)/(2*h))
        F = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                F[i,j] = np.sum(np.real(derivs[i])*np.real(derivs[j])/sigma_diff_est**2
                                  + np.imag(derivs[i])*np.imag(derivs[j])/sigma_diff_est**2)
    else:
        def neg_log_posterior_gls(theta):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return 1e10
            model = fano_models.Sij(f_grid, *theta, delay, n_ports=aa_n_ports)
            res_r = np.real(s21 - model)
            res_i = np.imag(s21 - model)
            chi2 = gls_chi2(res_r, res_i, sigma_baseline, rho)
            return 0.5*chi2 - lp

        best_theta_final, best_val = None, np.inf
        for theta0 in candidates + [popt1]:
            res = minimize(neg_log_posterior_gls, theta0, method='Nelder-Mead',
                            options={'maxiter': 20000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_theta_final = res.x
        popt_final = best_theta_final

        n = len(popt_final)
        P = ar1_precision_sparse(len(f_grid), sigma_baseline, rho)
        derivs = []
        for i in range(n):
            tp, tm = popt_final.copy(), popt_final.copy()
            h = 1e-6*max(abs(popt_final[i]), 1e-8)
            tp[i]+=h; tm[i]-=h
            mp = fano_models.Sij(f_grid, *tp, delay, n_ports=aa_n_ports)
            mm = fano_models.Sij(f_grid, *tm, delay, n_ports=aa_n_ports)
            derivs.append((mp-mm)/(2*h))
        F = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                F[i,j] = (np.real(derivs[i])@(P@np.real(derivs[j]))
                          + np.imag(derivs[i])@(P@np.imag(derivs[j])))

    try:
        cov_final = np.linalg.inv(F)
        eigvals = np.linalg.eigvalsh(cov_final)
        cov_ok = eigvals.min() > 0
    except np.linalg.LinAlgError:
        cov_ok = False

    if cov_ok:
        mc = np.random.default_rng(0).multivariate_normal(popt_final, cov_final, size=aa_mc_sample_size)
        Qi_mc = 1.0/(1.0/mc[:,1] - 1.0/mc[:,2])
        valid = (Qi_mc>0) & (Qi_mc<1e7)
        if np.sum(valid) >= 10:
            Qi_lo, Qi_med, Qi_hi = bt.credible_interval(Qi_mc[valid], level=0.68)
        else:
            Qi_lo, Qi_med, Qi_hi = np.nan, np.nan, np.nan
    else:
        Qi_lo, Qi_med, Qi_hi = np.nan, np.nan, np.nan

    boundary_flag = (popt_final[1] <= 1.01) or (popt_final[2] <= 1.01)
    width = Qi_hi - Qi_lo if cov_ok else np.nan

    results_summary.append({
        'resonator': filename.split('_powersweep')[0], 'method': method_name,
        'Ql': popt_final[1], 'Qc': popt_final[2], 'boundary_flag': boundary_flag,
        'Qi_med': Qi_med, 'Qi_width': width,
    })
    print(f"최종: Ql={popt_final[1]:.1f}, Qc={popt_final[2]:.1f}, Qi={Qi_med:.1f}, 폭={width:.1f}")


print("\n" + "=" * 100)
print(f"{'Resonator':<14} {'방식':>8} {'Ql':>10} {'Qc':>10} {'경계':>6} {'Qi':>14} {'폭':>12}")
print("=" * 100)
for r in results_summary:
    flag = "예" if r['boundary_flag'] else "-"
    print(f"{r['resonator']:<14} {r['method']:>8} {r['Ql']:>10.1f} {r['Qc']:>10.1f} "
          f"{flag:>6} {r['Qi_med']:>14.1f} {r['Qi_width']:>12.1f}")
