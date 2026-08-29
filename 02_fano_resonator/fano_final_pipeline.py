"""
fano_final_pipeline.py
=================================================================
[오늘 하루의 최종 통합 파이프라인] 실측 Fano 공진기 데이터에서
Qi(내부 품질계수)를 안정적으로 추정하기 위한, 오늘 하루 시행착오
끝에 도달한 표준 절차입니다.

[전체 순서 - 왜 이 순서인가]
  1. circle fit(대수적 방법)으로 "안전한 초기값" 확보
     - 이유: FWHM 기반 추정만으로는 가끔(예: resonator_4) 완전히
       실패해서 최적화가 물리적으로 불가능한 지점(Ql 또는 Qc가
       0에 가까운 prior 경계)으로 도망가는 문제가 있었음. circle
       fit은 반복 없는 대수적 방법이라 이런 국소최적점 함정이 없어,
       "믿을 수 있는 출발점 하나"를 항상 보장해줌.
  2. 1차 피팅(baseline sigma) -> 잔차 진단
     - 잔차의 |값| 상위 5%를 이상치로 제거
     - 이유: 이상치를 먼저 안 걷어내면, 이후 잡음 구조 진단(자기상관
       등)이 이상치에 의해 왜곡될 수 있음. 표준적인 통계 실무 순서는
       "이상치 처리 -> 그 다음에 남은 잡음의 구조(독립/상관) 판단".
  3. 이상치 제거 후 남은 잔차로 MAD(중앙값 절대편차) 기반 강건한
     sigma 추정
     - 단순 표준편차보다 이상치에 덜 흔들리는 잡음 크기 추정치.
  4. Gaussian 우도와 Robust(Student-t) 우도로 각각 독립적으로 재피팅
     - 이유: 서로 완전히 다른 두 우도 함수가 거의 같은 답에
       수렴하면, 그 답이 우연이 아니라 진짜 최적점이라는 강한
       증거가 됨(오늘 R7에서 실제로 확인: 두 방식이 소수점 몇
       자리까지 일치).
  5. 최종 채택된 파라미터에서, 피셔행렬 기반 공분산으로 Qi의
     68% 신뢰구간을 계산 (emcee 없이도 MCMC posterior와 개념적으로
     동일한 정보를 주는 방법 - Laplace 근사)

[오늘 발견한, 아직 안 풀린 한계 - 정직하게 남겨둠]
  - 잔차에 강한 자기상관(색깔있는 잡음)이 있다는 게 확인됐지만
    (lag-1 rho 0.7~0.99, 9/9 resonator 전부), 이걸 AR(1)로
    반영하려던 시도는 모델 오적합(장거리 지속성을 못 잡음) 문제가,
    1차 차분으로 반영하려던 시도는 물리 신호 정보 손실 문제가,
    다항식 배경 추가로 반영하려던 시도는 새로운 축퇴(배경과 물리
    파라미터가 뒤엉킴) 문제가 각각 있었음. 그래서 이 최종
    파이프라인은 "잡음이 독립"이라는 단순 가정으로 되돌아갔고,
    이는 파라미터 중심값(point estimate)은 안정적으로 주지만,
    신뢰구간(폭)은 실제보다 좁게 나올 수 있다는 걸 의미함 -
    자기상관을 정확히 반영하는 것은 다음 과제로 남겨둠.

[emcee 관련 안내]
이 스크립트가 실행되는 환경에는 emcee가 없어 진짜 MCMC는 못
돌립니다. 대신 curve_fit/scipy.optimize.minimize로 MAP(posterior
최댓값)을 찾고, 피셔행렬로 그 주변의 곡률(공분산)을 구해 몬테카를로
샘플링하는 방식(Laplace 근사)을 씁니다. log_probability 함수는
fano_likelihood.make_log_probability_generic으로 만들어져 있어서,
emcee가 있는 환경(Colab)에서는 fano_mcmc_pipeline.run_single_mcmc에
그대로 넘겨 진짜 MCMC로 업그레이드할 수 있습니다.
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
    # [조정] 실제 .npz 파일들이 있는 폴더 경로로 바꿔주세요.
aa_resonator_files = [
    'resonator_1_powersweep_overcoupled.npz', 'resonator_2_powersweep_overcoupled.npz',
    'resonator_3_powersweep_overcoupled.npz', 'resonator_4_powersweep_overcoupled.npz',
    'resonator_5_powersweep_overcoupled.npz', 'resonator_6_powersweep_overcoupled.npz',
    'resonator_7_powersweep_overcoupled.npz', 'resonator_8_powersweep_overcoupled.npz',
    'resonator_9_powersweep_overcoupled.npz',
]
aa_power_slice_index = 0
    # [조정] 어느 전력 slice를 볼지. 0=최저 전력(원저자 예제와 일관).
aa_n_ports = 1.0
    # [조정] reflection_port(1.0) vs notch_port(2.0). 오늘 데이터는
    # 원저자 예제 노트북 확인 결과 reflection(1.0)이 맞았음.
aa_isolation_db = 15
    # [조정] circle fit의 Fano 불확실성 범위 계산에 쓸 가정값.
aa_phi_candidates = [-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5]
    # [조정] multi-start 최적화에 쓸 phi 초기값 후보들. phi가 주기적
    # 파라미터라 국소최적점에 빠지기 쉬우므로, 여러 후보를 넉넉히
    # 시도하고 chi2가 가장 낮은 것을 채택.
aa_outlier_trim_pct = 5.0
    # [조정] 잔차 절댓값 기준 상위 몇 %를 이상치로 제거할지. 너무
    # 크게 잡으면(예: 20%) 진짜 신호 정보까지 잃을 위험, 너무 작게
    # 잡으면(예: 1%) 이상치 제거 효과가 미미함 - 5%가 오늘 검증된
    # 무난한 절충점.
aa_robust_nu = 4.0
    # [조정] Robust(Student-t) 우도의 자유도. 작을수록(예: 2) 꼬리가
    # 두꺼워져 이상치에 더 관대해지고, 클수록(예: 30) 가우시안에
    # 가까워짐. 4.0은 흔히 쓰이는 중간값.
aa_mc_sample_size = 30000
    # [조정] 피셔행렬 공분산에서 뽑을 몬테카를로 표본 개수.
aa_step_fraction = 1e-6
    # [조정] 피셔행렬 수치미분 스텝 크기. fr처럼 절대 스케일이 큰
    # 파라미터(~1e9)의 미분이 부정확해지지 않도록 충분히 작게 설정
    # (오늘 1e-2~1e-4는 부정확, 1e-6부터 안정적으로 수렴 확인됨).
aa_output_dir = './outputs/'
os.makedirs(aa_output_dir, exist_ok=True)

print("=" * 60)
print("분석자 설정값(aa_*) 요약:")
for k, v in list(globals().items()):
    if k.startswith("aa_") and not isinstance(v, list):
        print(f"  {k:24s} = {v}")
print("=" * 60)


# =========================================================
# STEP 1. 데이터 로더
# =========================================================
def load_fano_npz(filepath):
    """
    resonator_*_powersweep_overcoupled.npz 파일 하나를 읽어서
    복소수 S21과 케이블 지연(cable_delay)을 반환.
    amplitude*exp(i*phase) 형태로 저장된 진폭/위상을 복소수로 재구성.
    """
    raw = np.load(filepath, allow_pickle=True)
        # allow_pickle=True: settings 항목이 순수 숫자 배열이 아니라
        # JSON 문자열(파이썬 객체)을 담고 있어서 필요.
    amplitude, phase = raw['amplitude'], raw['phase']
    freq_hz, power_dbm = raw['frequency'], raw['power']
    s21 = amplitude * np.exp(1j * phase)
    settings_dict = json.loads(raw['settings'][0])
    cable_delay = settings_dict.get('vna', {}).get('port1_edel', None)
    return {'power_dbm': power_dbm, 'freq_hz': freq_hz, 's21': s21, 'cable_delay': cable_delay}


# =========================================================
# STEP 2. 하나의 resonator를 처리하는 함수 (전체 파이프라인)
# =========================================================
param_names = ['fr', 'Ql', 'Qc', 'phi', 'a', 'alpha']


def process_one_resonator(filename):
    """
    파일 하나에 오늘의 전체 절차(circle fit 안전망 -> 이상치 제거 ->
    Gaussian/Robust 교차검증 -> 피셔행렬 신뢰구간)를 적용하고,
    결과 딕셔너리를 반환.
    """
    full_path = os.path.join(aa_upload_dir, filename)
    data = load_fano_npz(full_path)
    f_grid = data['freq_hz']
    s21 = data['s21'][aa_power_slice_index, :]
    delay = data['cable_delay']
    mag = np.abs(s21)

    # --- prior 및 log_probability 준비 ([라이브러리 사용]) ---
    prior_bounds = {
        'fr': (f_grid.min(), f_grid.max()), 'Ql': (1.0, 1e8), 'Qc': (1.0, 1e8),
        'phi': (-np.pi, np.pi), 'a': (1e-8, 1.0), 'alpha': (-np.pi, np.pi),
    }
    log_prior = likelihood.make_uniform_log_prior(prior_bounds)
        # [라이브러리] fano_likelihood.make_uniform_log_prior:
        # {파라미터이름: (하한,상한)} 형태의 딕셔너리를 받아, 그 범위
        # 안에서는 균등(uniform)한 사전확률을, 범위 밖에서는 -inf를
        # 반환하는 함수를 만들어줌.
    fixed_kwargs = {'delay': delay, 'n_ports': aa_n_ports}
    log_probability = likelihood.make_log_probability_generic(
        fano_models.Sij, fixed_kwargs, param_names, log_prior, likelihood_type='gaussian'
    )
        # [라이브러리, 오늘 신규 추가] fano_likelihood.make_log_probability_generic:
        # 시그니처가 log_probability(theta, f_grid, data_1d, sigma)로,
        # emcee가 있는 환경에서는 fano_mcmc_pipeline.run_single_mcmc에
        # 그대로 넘겨 진짜 MCMC를 돌릴 수 있음.

    # --- 데이터 기반 초기값 추정 ---
    fr_guess = f_grid[np.argmin(mag)]
        # 이 모델은 reflection(딥 형태)이므로, |S21|이 최소인 지점이
        # 공진 주파수(fr)에 대한 직접적인 추정치가 됨.
    half_level = (mag.max() + mag.min()) / 2
    above_half = np.where(mag > half_level)[0]
    fwhm_guess = f_grid[above_half[-1]] - f_grid[above_half[0]] if len(above_half) > 1 else 1e6
    Ql_guess = fr_guess / max(fwhm_guess, 1e3)
        # Ql = fr/kappa 관계를 이용해 반치폭(FWHM)으로부터 역산.

    # --- [핵심 1] circle fit으로 "안전한" 초기값 후보 확보 ---
    try:
        cf_result = fano_models.autofit(f_grid, s21, n_ports=aa_n_ports,
                                          fixed_delay=delay, isolation=aa_isolation_db)
        circle_fit_theta = np.array([cf_result['fr'], cf_result['Ql'], cf_result['Qc'],
                                       cf_result['phi'], cf_result['a'], cf_result['alpha']])
    except Exception:
        circle_fit_theta = None
        cf_result = None

    candidates = [np.array([fr_guess, Ql_guess, Ql_guess*1.1, phi0, mag.max()-mag.min(), phi0])
                  for phi0 in aa_phi_candidates]
    if circle_fit_theta is not None:
        candidates.append(circle_fit_theta)
            # circle fit 결과도 초기값 후보 목록에 추가. 국소최적점에
            # 갇히기 쉬운 FWHM 기반 후보들과 달리, circle fit은
            # 대수적(반복 없는) 방법이라 항상 물리적으로 타당한
            # 지점을 준다는 게 오늘 resonator_4 사례로 확인됨.

    def find_map(sigma, theta0_list, log_prob_func):
        """multi-start Nelder-Mead: 여러 초기값 후보 중 -log_prob가
        가장 낮은(=posterior가 가장 높은) 지점을 채택."""
        best_theta, best_neg_lp = None, np.inf
        for theta0 in theta0_list:
            def neg_log_prob(theta):
                val = log_prob_func(theta, sigma)
                return -val if np.isfinite(val) else 1e10
            res = minimize(neg_log_prob, theta0, method='Nelder-Mead',
                            options={'maxiter': 20000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_neg_lp:
                best_neg_lp = res.fun
                best_theta = res.x
        return best_theta

    # --- 1차 피팅 (baseline sigma) - 잔차 진단을 위한 예비 단계 ---
    baseline_mask = mag > np.percentile(mag, 80)
        # reflection 모델은 공진에서 진폭이 "작아지는" 딥 형태이므로,
        # 진폭이 큰 상위 20% 구간을 "공진에서 먼 baseline"으로 봄.
    sigma_baseline = np.std(np.real(s21[baseline_mask]))

    popt0 = find_map(sigma_baseline, candidates,
                       lambda th, s: log_probability(th, f_grid, s21, s))
    model0 = fano_models.Sij(f_grid, *popt0, delay, n_ports=aa_n_ports)
    residual0 = np.real(s21 - model0)

    # --- [핵심 2] 이상치 상위 5% 제거 ---
    abs_res = np.abs(residual0)
    threshold = np.percentile(abs_res, 100 - aa_outlier_trim_pct)
    keep_mask = abs_res <= threshold
    n_outliers = np.sum(~keep_mask)

    f_trim = f_grid[keep_mask]
    s21_trim = s21[keep_mask]
    residual_trimmed = residual0[keep_mask]

    # --- [핵심 3] MAD 기반 강건한 sigma ---
    mad = np.median(np.abs(residual_trimmed - np.median(residual_trimmed)))
    sigma_mad = mad * 1.4826
        # 1.4826: MAD를 표준편차와 같은 스케일로 맞춰주는 표준적인
        # 보정 상수(정규분포 가정 하에 유도된 관례적 계수).

    # --- [핵심 4] Gaussian / Robust 우도로 각각 독립 재피팅 ---
    def log_prob_gaussian_trimmed(theta, sigma):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        model = fano_models.Sij(f_trim, *theta, delay, n_ports=aa_n_ports)
        res_r = np.real(s21_trim - model)
        res_i = np.imag(s21_trim - model)
        chi2 = np.sum((res_r/sigma)**2 + (res_i/sigma)**2)
        return lp - 0.5*chi2

    def log_prob_robust_trimmed(theta, sigma, nu=aa_robust_nu):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        model = fano_models.Sij(f_trim, *theta, delay, n_ports=aa_n_ports)
        res_r = np.real(s21_trim - model)
        res_i = np.imag(s21_trim - model)
        t_term = lambda r: np.log1p((r/sigma)**2/nu)
            # Student-t 분포의 로그우도 핵심 부분. 가우시안(-0.5*r^2/sigma^2)과
            # 달리, r이 커져도(이상치) log1p(...)가 완만하게 증가해서
            # "이상치의 영향력을 자동으로 깎아내는" 효과를 냄.
        return lp - 0.5*(nu+1)*np.sum(t_term(res_r)+t_term(res_i))

    popt_gaussian = find_map(sigma_mad, candidates, log_prob_gaussian_trimmed)
    popt_robust = find_map(sigma_mad, candidates, log_prob_robust_trimmed)

    # 두 방식이 얼마나 일치하는지 확인 (일치할수록 신뢰도 높음)
    param_agreement_pct = np.abs(popt_gaussian - popt_robust) / (np.abs(popt_gaussian) + 1e-10) * 100

    # --- [핵심 5] 피셔행렬로 Qi 신뢰구간 계산 (Gaussian 결과 기준) ---
    def qi_credible(popt, sigma, f_data):
        F = fm.fisher_information_matrix(fano_models.Sij, popt, param_names, f_data,
                                            sigma, fixed_kwargs, step_fraction=aa_step_fraction)
        cov = fm.covariance_from_fisher(F)
            # [라이브러리] fano_fisher_matrix.py: 수치미분으로 피셔행렬을
            # 계산하고, 그 역행렬(공분산)을 반환.
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals.min() <= 0:
            return None, None, None, False
        mc = np.random.default_rng(0).multivariate_normal(popt, cov, size=aa_mc_sample_size)
            # 공분산 행렬을 가진 다변량 정규분포에서 표본을 뽑는 것이,
            # "posterior가 정확히 가우시안"이라는 가정 하에 emcee 없이도
            # MCMC 표본과 동일한 정보를 주는 방법(Laplace 근사).
        Qi_mc = 1.0/(1.0/mc[:,1] - 1.0/mc[:,2])
        valid = (Qi_mc > 0) & (Qi_mc < 1e7)
        if np.sum(valid) < 10:
            return None, None, None, False
        lo, med, hi = bt.credible_interval(Qi_mc[valid], level=0.68)
            # [라이브러리] fano_bayesian_toolkit.credible_interval:
            # (하한, 중앙값, 상한) 튜플을 반환하는 68% 등꼬리 신뢰구간.
        return lo, med, hi, True

    Qi_lo, Qi_med, Qi_hi, cov_ok = qi_credible(popt_gaussian, sigma_mad, f_trim)
    boundary_flag = (popt_gaussian[1] <= 1.01) or (popt_gaussian[2] <= 1.01)

    return {
        'resonator': filename.split('_powersweep')[0],
        'circle_fit_Qi': cf_result['Qi'] if cf_result else np.nan,
        'popt_gaussian': popt_gaussian, 'popt_robust': popt_robust,
        'max_param_disagreement_pct': param_agreement_pct.max(),
        'n_outliers_removed': n_outliers,
        'Qi_median': Qi_med, 'Qi_lo': Qi_lo, 'Qi_hi': Qi_hi,
        'cov_ok': cov_ok, 'boundary_flag': boundary_flag,
    }


# =========================================================
# STEP 3. 9개 resonator 전부 반복 처리
# =========================================================
results_summary = []
for filename in aa_resonator_files:
    full_path = os.path.join(aa_upload_dir, filename)
    if not os.path.exists(full_path):
        print(f"[건너뜀] 파일 없음: {filename}")
        continue
    print(f"\n처리 중: {filename}")
    try:
        result = process_one_resonator(filename)
        results_summary.append(result)
        flag = "⚠️" if result['boundary_flag'] else "정상"
        print(f"  Qi={result['Qi_median']:.1f} [{result['Qi_lo']:.1f},{result['Qi_hi']:.1f}], "
              f"Gaussian/Robust 최대 불일치={result['max_param_disagreement_pct']:.2f}%, [{flag}]")
    except Exception as e:
        print(f"  실패: {e}")


# =========================================================
# STEP 4. 최종 요약표 및 시각화
# =========================================================
print("\n" + "=" * 100)
print(f"{'Resonator':<14} {'circle fit Qi':>14} {'최종 Qi':>12} {'[68% 신뢰구간]':>24} {'G/R 불일치%':>12}")
print("=" * 100)
for r in results_summary:
    ci_str = f"[{r['Qi_lo']:.0f}, {r['Qi_hi']:.0f}]" if r['cov_ok'] else "계산 실패"
    print(f"{r['resonator']:<14} {r['circle_fit_Qi']:>14.1f} {r['Qi_median']:>12.1f} "
          f"{ci_str:>24} {r['max_param_disagreement_pct']:>12.2f}")

fig, ax = plt.subplots(figsize=(10, 5))
names = [r['resonator'].replace('resonator_', 'R') for r in results_summary]
medians = [r['Qi_median'] for r in results_summary]
lo = [r['Qi_median']-r['Qi_lo'] for r in results_summary]
hi = [r['Qi_hi']-r['Qi_median'] for r in results_summary]
colors = ['tab:red' if r['boundary_flag'] else 'tab:blue' for r in results_summary]
ax.errorbar(range(len(names)), medians, yerr=[lo, hi], fmt='o', ecolor='gray', capsize=4, ms=0)
for i, (m, c) in enumerate(zip(medians, colors)):
    ax.plot(i, m, 'o', color=c, ms=10)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names)
ax.set_yscale('log')
ax.set_ylabel('Qi (68% 신뢰구간)')
ax.set_title('최종 파이프라인: circle fit 초기값 + 5% 이상치 제거 + Gaussian/Robust 교차검증 + 피셔행렬')
plt.tight_layout()
plt.savefig(os.path.join(aa_output_dir, 'fano_final_pipeline_result.png'), dpi=140, bbox_inches='tight')
print(f"\n최종 결과 그래프 저장: {os.path.join(aa_output_dir, 'fano_final_pipeline_result.png')}")

print("\n[알려진 한계 - 정직하게 남겨둠]")
print("  잔차에 강한 자기상관(색깔있는 잡음)이 있다는 게 확인됐지만, 이 파이프라인은")
print("  '잡음이 독립'이라는 단순 가정을 씁니다. 따라서 위 신뢰구간(폭)은 실제보다")
print("  좁게 나올 가능성이 있습니다 - 파라미터 중심값은 안정적이지만, 폭 자체의")
print("  정확한 보정은 다음 과제로 남아있습니다.")
