"""
validation_robots.py
=================================================================
[종합판] 오늘 하루 배운 검증 원칙 전부를 하나의 "로봇 부대"로 정리.
"바둑판에 통계 로봇을 풀어놓고 자동 탐색시킨다"는 아이디어의 완성판.

포함된 로봇들:
  1. check_prior_sanity        - prior 범위 자체가 답을 배제하는지
  2. run_recovery_robots        - 무작위 참값으로 파라미터 복원 반복검증
  3. check_multistart_consistency - 여러 시작점이 같은 답으로 모이는지
  4. check_parameter_correlation  - 파라미터끼리 심하게 얽혔는지(V0 문제)
  5. check_residual_autocorrelation - 잔차가 넓은 lag 범위까지 독립인지
  6. cross_check_gaussian_vs_robust - 이상치 오염 여부
  7. compute_quality_score      - 신호 vs 잡음 구분(대비/배경 비율)
  8. run_full_diagnostic_suite  - 위 전부를 순서대로 실행하는 종합진단
"""

import numpy as np
from scipy.optimize import curve_fit


# =========================================================
# 로봇 1. Prior 범위 자동 검사기
# =========================================================
def check_prior_sanity(prior_bounds, reference_candidates=None):
    """
    prior 범위 자체에 문제가 없는지 자동 점검.
    오늘 사례: gamma_stark prior가 (-0.01,0.01)이라 실제 답(0.109)을
    원천 배제했던 버그 - 이걸 코드가 먼저 잡아내는 게 목적.
    """
    warnings = []
    for name, (lo, hi) in prior_bounds.items():
        if lo >= hi:
            warnings.append(f"[치명적] '{name}': 하한({lo})이 상한({hi})보다 크거나 같음")
            continue
        if reference_candidates and name in reference_candidates:
            for cand in reference_candidates[name]:
                if not (lo < cand < hi):
                    warnings.append(
                        f"[경고] '{name}': 후보값 {cand}가 prior 범위 [{lo},{hi}] 밖 - "
                        f"이 값은 최적화가 원리적으로 절대 못 찾음"
                    )
    return warnings


# =========================================================
# 로봇 2. 자동 파라미터 복원 검증
# =========================================================
def run_recovery_robots(model_func, param_names, prior_bounds, x_grid,
                           n_trials=30, noise_level=0.05, tolerance=0.1,
                           fixed_kwargs=None, seed=0):
    """무작위 참값 n_trials개로 합성 데이터를 만들고, 다시 피팅해서
    참값을 복원하는지 반복 확인. 오늘 손으로 30번 넘게 한 검증의 자동화."""
    if fixed_kwargs is None:
        fixed_kwargs = {}
    rng = np.random.default_rng(seed)

    def model_wrapper(x_flat, *theta):
        params = dict(zip(param_names, theta))
        return model_func(x_grid, **params, **fixed_kwargs).flatten()

    results = []
    for trial in range(n_trials):
        true_params = {name: rng.uniform(lo, hi) for name, (lo, hi) in prior_bounds.items()}
        true_theta = [true_params[name] for name in param_names]

        data_true = model_func(x_grid, **true_params, **fixed_kwargs)
        noise = rng.normal(0, noise_level * (np.abs(data_true).max() + 1e-10), data_true.shape)
        data_noisy = data_true + noise

        try:
            popt, _ = curve_fit(model_wrapper, x_grid, data_noisy.flatten(),
                                  p0=true_theta, maxfev=10000)
            recovered = dict(zip(param_names, popt))
        except Exception as e:
            results.append({'trial': trial, 'success': False, 'error': str(e)})
            continue

        param_results = {}
        for name in param_names:
            tv, rv = true_params[name], recovered[name]
            rel_err = abs(rv - tv) / (abs(tv) + 1e-10)
            param_results[name] = {'true': tv, 'recovered': rv, 'rel_err': rel_err,
                                     'pass': rel_err < tolerance}
        overall_pass = all(p['pass'] for p in param_results.values())
        results.append({'trial': trial, 'success': True, 'overall_pass': overall_pass,
                         'params': param_results})

    successful = [r for r in results if r['success']]
    overall_pass_rate = np.mean([r['overall_pass'] for r in successful]) if successful else 0.0
    per_param_pass_rate = {}
    for name in param_names:
        passes = [r['params'][name]['pass'] for r in successful if name in r['params']]
        per_param_pass_rate[name] = np.mean(passes) if passes else 0.0

    return {'pass_rate': overall_pass_rate, 'per_param_pass_rate': per_param_pass_rate,
            'n_trials': n_trials, 'all_results': results}


# =========================================================
# 로봇 3. multi-start 일관성 검사 (국소최적점/다중봉우리 탐지)
# =========================================================
def check_multistart_consistency(model_func, param_names, x_grid, data,
                                     p0_candidates, fixed_kwargs=None,
                                     bounds=None, chi2_similarity_thresh=0.2):
    """
    같은 데이터를 여러 초기값에서 각각 피팅해서, 결과가 한 곳으로
    모이는지(=신뢰할 만한 유일한 답) 아니면 흩어지는지(=국소최적점
    또는 다중봉우리 posterior 의심) 확인.

    [오늘 배운 핵심] chi2들이 서로 비슷한데(chi2_similarity_thresh
    이내) 파라미터 값이 서로 다르면, 이건 점추정으로 해결 못 하는
    다중봉우리 상황 - 진짜 MCMC가 필요하다는 신호.
    """
    if fixed_kwargs is None:
        fixed_kwargs = {}

    def model_wrapper(x_flat, *theta):
        params = dict(zip(param_names, theta))
        return model_func(x_grid, **params, **fixed_kwargs).flatten()

    trials = []
    for p0 in p0_candidates:
        try:
            if bounds is not None:
                popt, _ = curve_fit(model_wrapper, x_grid, data.flatten(), p0=p0,
                                      bounds=bounds, maxfev=20000)
            else:
                popt, _ = curve_fit(model_wrapper, x_grid, data.flatten(), p0=p0, maxfev=20000)
            pred = model_wrapper(x_grid, *popt).reshape(data.shape)
            chi2 = np.sum((data - pred) ** 2)
            trials.append({'p0': p0, 'popt': popt, 'chi2': chi2})
        except Exception:
            continue

    if len(trials) < 2:
        return {'consistent': None, 'warning': 'multi-start 시도 대부분 실패 - 판단 불가',
                'trials': trials}

    chi2_vals = np.array([t['chi2'] for t in trials])
    min_chi2 = chi2_vals.min()
    near_best = [t for t, c in zip(trials, chi2_vals)
                  if (c - min_chi2) / max(min_chi2, 1e-10) < chi2_similarity_thresh]

    if len(near_best) < 2:
        return {'consistent': True, 'best_trial': trials[np.argmin(chi2_vals)], 'trials': trials}

    popts_near_best = np.array([t['popt'] for t in near_best])
    param_spread = np.std(popts_near_best, axis=0) / (np.abs(np.mean(popts_near_best, axis=0)) + 1e-10)
    is_consistent = np.all(param_spread < tolerance_for_spread(1.0))

    return {
        'consistent': bool(np.all(param_spread < 0.3)),
        'n_near_best_chi2': len(near_best),
        'param_spread_relative': dict(zip(param_names, param_spread)),
        'best_trial': trials[np.argmin(chi2_vals)],
        'trials': trials,
    }


def tolerance_for_spread(x):
    """가독성을 위한 헬퍼 - 실제 임계값은 check_multistart_consistency
    안의 0.3 기준을 직접 사용."""
    return x


# =========================================================
# 로봇 4. 파라미터 상관관계 검사 (V0 재중심화 필요 여부 탐지)
# =========================================================
def check_parameter_correlation(pcov, param_names, threshold=0.8):
    """
    공분산 행렬에서 상관계수를 계산해, 심하게 얽힌 파라미터 쌍이
    있는지 확인. 오늘 V0 재중심화 전 f_TLS0-gamma_stark 상관계수가
    1.0이었던 것과 같은 문제를 자동으로 잡아냄.
    """
    std = np.sqrt(np.abs(np.diag(pcov)))
    corr = pcov / np.outer(std, std)
    n = len(param_names)
    flagged_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if abs(corr[i, j]) > threshold:
                flagged_pairs.append((param_names[i], param_names[j], corr[i, j]))
    return {'correlation_matrix': corr, 'flagged_pairs': flagged_pairs,
            'suggestion': "재중심화(V0 같은 기준점 이동)를 고려하세요" if flagged_pairs else None}


# =========================================================
# 로봇 5. 잔차 자기상관 검사 (넓은 lag 범위)
# =========================================================
def check_residual_autocorrelation(residual, lags=None):
    """
    잔차가 정말 "독립"인지, 넓은 lag 범위에 걸쳐 확인. 오늘 배운
    핵심 교훈: lag=29까지만 보면 "거의 안 줄어드는 것처럼" 보여서
    잘못 판단할 뻔했음 - lag를 데이터 길이의 상당 부분까지 넓게
    봐야 진짜 감쇠 구조가 드러남.
    """
    n = len(residual)
    if lags is None:
        lags = sorted(set([1, 5, 10, 20, n//10, n//5, n//3, n//2]))
        lags = [l for l in lags if 0 < l < n]

    xc = residual - np.mean(residual)
    denom = np.sum(xc**2)

    autocorrs = {}
    for lag in lags:
        num = np.sum(xc[:-lag] * xc[lag:])
        autocorrs[lag] = num / denom if denom > 0 else 0.0

    long_range_lag = max(lags)
    long_range_value = autocorrs[long_range_lag]
    short_range_value = autocorrs[min(lags)]

    warning = None
    if abs(long_range_value) > 0.3:
        warning = (f"lag={long_range_lag}(데이터 상당 부분)에서도 자기상관이 "
                   f"{long_range_value:.3f}로 남아있음 - AR(1) 같은 단거리 모델로는 "
                   f"부족할 수 있음(오늘 배운 교훈)")

    return {'autocorrelations': autocorrs, 'short_range': short_range_value,
            'long_range': long_range_value, 'warning': warning}


# =========================================================
# 로봇 6. Gaussian vs Robust 교차검증
# =========================================================
def cross_check_gaussian_vs_robust(model_func, param_names, x_grid, data,
                                       p0, fixed_kwargs=None, bounds=None,
                                       nu=4.0, disagreement_threshold=0.2):
    """
    같은 데이터를 Gaussian과 Robust(Student-t) 우도로 각각 피팅해서
    비교. 둘이 거의 같으면 이상치 오염이 없다는 뜻, 크게 다르면
    이상치가 결과를 왜곡시키고 있다는 신호.

    [오늘 검증된 방식 그대로 사용] scipy curve_fit의 soft_l1 loss는
    처음 시도했을 때 이상치 감지력이 약해서(합성 데이터로 확인:
    심한 이상치 7개를 넣어도 threshold를 못 넘음) 하루 종일 실제로
    검증된 Student-t 로그우도 + scipy.optimize.minimize 방식으로 교체.
    """
    if fixed_kwargs is None:
        fixed_kwargs = {}

    def model_wrapper(theta):
        params = dict(zip(param_names, theta))
        return model_func(x_grid, **params, **fixed_kwargs).flatten()

    sigma_est = np.std(data.flatten() - model_wrapper(p0)) or 1.0

    def neg_log_gaussian(theta):
        residual = data.flatten() - model_wrapper(theta)
        return 0.5 * np.sum((residual/sigma_est)**2)

    def neg_log_robust(theta):
        residual = data.flatten() - model_wrapper(theta)
        return 0.5*(nu+1) * np.sum(np.log1p((residual/sigma_est)**2/nu))

    from scipy.optimize import minimize
    res_g = minimize(neg_log_gaussian, p0, method='Nelder-Mead',
                       options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8})
    res_r = minimize(neg_log_robust, res_g.x, method='Nelder-Mead',
                       options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8})
    popt_gauss, popt_robust = res_g.x, res_r.x

    rel_diffs = {}
    for i, name in enumerate(param_names):
        rel_diffs[name] = abs(popt_gauss[i] - popt_robust[i]) / (abs(popt_gauss[i]) + 1e-10)

    max_disagreement = max(rel_diffs.values())
    warning = None
    if max_disagreement > disagreement_threshold:
        worst_param = max(rel_diffs, key=rel_diffs.get)
        warning = (f"Gaussian과 Robust가 '{worst_param}'에서 "
                   f"{max_disagreement*100:.1f}% 차이 - 이상치 오염 의심")

    return {'popt_gaussian': dict(zip(param_names, popt_gauss)),
            'popt_robust': dict(zip(param_names, popt_robust)),
            'relative_disagreement': rel_diffs, 'warning': warning}


# =========================================================
# 로봇 7. 품질점수(SNR) - 신호 vs 잡음 구분
# =========================================================
def compute_quality_score(signal_values_at_feature, background_values):
    """
    "찾은 게 진짜 신호인지 잡음인지" 판단하는 절대적 기준. 오늘 배운
    교훈: 길이/단조성 같은 상대적 기준만으로는 순수 잡음도 통과함
    (직접 확인함) - 배경 대비 얼마나 두드러지는지(대비/배경표준편차)
    라는 절대적 기준이 반드시 필요.
    """
    depth = np.mean(np.abs(signal_values_at_feature))
    background_std = np.std(background_values)
    return depth / (background_std + 1e-12)


# =========================================================
# 로봇 8. 종합 진단 - 위 전부를 순서대로 실행
# =========================================================
def run_full_diagnostic_suite(model_func, param_names, prior_bounds, x_grid, data,
                                 fixed_kwargs=None, reference_candidates=None,
                                 p0_candidates=None, bounds_for_fit=None,
                                 n_recovery_trials=20, verbose=True):
    """
    오늘 확립한 8단계 표준 작업순서를 그대로 자동 실행하는 종합진단.
      1. prior 범위 검사
      2. (합성 데이터로) 파라미터 복원 검증
      3. multi-start 일관성 검사
      4. 파라미터 상관관계 검사
      5. Gaussian vs Robust 교차검증
      6. 잔차 자기상관 검사
    각 단계 결과를 모아 최종 "신뢰 판정"을 내림.
    """
    report = {}

    if verbose:
        print("="*60)
        print("[종합 진단 시작]")
        print("="*60)

    # 1. prior 검사
    prior_warnings = check_prior_sanity(prior_bounds, reference_candidates)
    report['prior_warnings'] = prior_warnings
    if verbose:
        print(f"\n[1/6] Prior 검사: {'문제없음' if not prior_warnings else f'{len(prior_warnings)}개 경고'}")
        for w in prior_warnings:
            print(f"  {w}")

    # 2. 파라미터 복원 검증(합성 데이터)
    recovery = run_recovery_robots(model_func, param_names, prior_bounds, x_grid,
                                      n_trials=n_recovery_trials, fixed_kwargs=fixed_kwargs)
    report['recovery'] = recovery
    if verbose:
        print(f"\n[2/6] 파라미터 복원 검증: 통과율 {recovery['pass_rate']*100:.1f}%")

    # 3. multi-start (p0_candidates 제공된 경우만)
    if p0_candidates is not None:
        multistart = check_multistart_consistency(
            model_func, param_names, x_grid, data, p0_candidates,
            fixed_kwargs=fixed_kwargs, bounds=bounds_for_fit
        )
        report['multistart'] = multistart
        if verbose:
            print(f"\n[3/6] multi-start 일관성: {'일관됨' if multistart.get('consistent') else '불일치(다중봉우리 의심)'}")
    else:
        report['multistart'] = None
        if verbose:
            print("\n[3/6] multi-start: p0_candidates 미제공 - 건너뜀")

    # 4. 파라미터 상관관계 (실제 데이터 피팅해서)
    def model_wrapper(x_flat, *theta):
        params = dict(zip(param_names, theta))
        return model_func(x_grid, **params, **(fixed_kwargs or {})).flatten()
    try:
        p0_default = report['multistart']['best_trial']['popt'] if report['multistart'] else \
                      [np.mean(prior_bounds[n]) for n in param_names]
        popt_real, pcov_real = curve_fit(model_wrapper, x_grid, data.flatten(), p0=p0_default, maxfev=20000)
        corr_check = check_parameter_correlation(pcov_real, param_names)
        report['correlation'] = corr_check
        if verbose:
            print(f"\n[4/6] 파라미터 상관관계: {len(corr_check['flagged_pairs'])}개 쌍 요주의")
            for name1, name2, c in corr_check['flagged_pairs']:
                print(f"  {name1}-{name2}: 상관계수={c:.3f}")

        # 5. Gaussian vs Robust
        gr_check = cross_check_gaussian_vs_robust(model_func, param_names, x_grid, data,
                                                      p0_default, fixed_kwargs=fixed_kwargs,
                                                      bounds=bounds_for_fit)
        report['gaussian_vs_robust'] = gr_check
        if verbose:
            print(f"\n[5/6] Gaussian vs Robust: {'일치' if not gr_check['warning'] else gr_check['warning']}")

        # 6. 잔차 자기상관
        residual = data.flatten() - model_wrapper(x_grid, *popt_real)
        autocorr_check = check_residual_autocorrelation(residual)
        report['autocorrelation'] = autocorr_check
        if verbose:
            print(f"\n[6/6] 잔차 자기상관: {'문제없음' if not autocorr_check['warning'] else autocorr_check['warning']}")
    except Exception as e:
        if verbose:
            print(f"\n[4-6단계 실패] {e}")
        report['correlation'] = None
        report['gaussian_vs_robust'] = None
        report['autocorrelation'] = None

    # 최종 판정
    issues = len(prior_warnings)
    issues += 1 if recovery['pass_rate'] < 0.7 else 0
    issues += 1 if report.get('multistart') and not report['multistart'].get('consistent') else 0
    issues += 1 if report.get('correlation') and report['correlation']['flagged_pairs'] else 0
    issues += 1 if report.get('gaussian_vs_robust') and report['gaussian_vs_robust']['warning'] else 0
    issues += 1 if report.get('autocorrelation') and report['autocorrelation']['warning'] else 0

    verdict = "신뢰 가능" if issues == 0 else ("주의 필요" if issues <= 2 else "신뢰 어려움 - 재검토 필요")
    report['verdict'] = verdict
    report['n_issues'] = issues

    if verbose:
        print("\n" + "="*60)
        print(f"[최종 판정] {verdict} (발견된 이슈: {issues}개)")
        print("="*60)

    return report


# =========================================================
# [용어 정리]
# =========================================================
"""
prior sanity check    : 사전분포 범위가 진짜 답을 배제하지 않는지 확인
parameter recovery    : 합성 데이터로 모델+피팅 파이프라인 자체를 검증
multi-start consistency: 여러 초기값이 같은 답으로 수렴하는지(다중봉우리 탐지)
parameter correlation : 파라미터끼리 얽혀있는지(재중심화 필요성 판단)
residual autocorrelation: 잔차가 독립인지, 넓은 lag까지 확인
Gaussian vs Robust     : 이상치 오염 여부 교차검증
quality score(SNR)     : 신호와 잡음을 구분하는 절대적 기준
"""
