"""
fano_final_estimation_with_corner.py
=================================================================
[오늘의 최종 파이프라인] Fano 공진기 실측 데이터에서 Qi(내부
품질계수)를 추정하기 위해, 오늘 하루 여러 시행착오 끝에 확정된
"잡음 처리 -> 매개변수 추정(피셔행렬+베이지안)" 전체 절차입니다.

=================================================================
[오늘 확정된 잡음 처리 순서 - 반드시 이 순서를 지켜야 하는 이유]
=================================================================

  STEP 1. circle fit(대수적 방법)으로 (a,alpha) 배경값과 안전한
          초기값 후보를 확보
          - 이유: 나중에 (a,alpha)를 고정할 때, "이 스캔 범위 안에서
            직접 평균낸 값"보다 "circle fit의 기하학적으로 도출된
            값"이 훨씬 정확하다는 게 오늘 실험으로 확인됨(아래 STEP 6
            참고).

  STEP 2. baseline sigma(진폭 상위 20% 구간의 표준편차)로 1차
          피팅(multi-start: phi 여러 후보 + circle fit 후보)
          - 잔차를 얻기 위한 예비 단계.

  STEP 3. 잔차의 자기상관을 "넓은 lag 범위"(1~900)로 진단
          - [오늘의 핵심 교훈] lag=29까지만 보면 상관이 "거의 안
            줄어드는 것처럼" 보여서 AR(1)이 안 맞다고(또는
            compound symmetry가 맞다고) 성급히 판단할 뻔했습니다.
            실제로는 lag=100~300 사이에서 서서히 감쇠하는, 긴
            길이척도(약 113포인트, 가우시안 커널로 추정)를 가진
            구조였습니다. "짧은 범위만 보고 잡음 모델을 고르면
            안 된다"는 게 오늘의 가장 중요한 교훈 중 하나입니다.

  STEP 4. |잔차| 상위 5%를 이상치로 제거
          - 이후 잡음 구조(자기상관 등) 진단이 이상치에 의해
            왜곡되지 않도록, 먼저 걸러냄.

  STEP 5. 이상치 제거 후 남은 잔차로 MAD(중앙값 절대편차) 기반
          강건한 sigma 추정
          - 단순 표준편차보다 이상치에 덜 흔들리는 잡음 크기.

  STEP 6. [핵심 발견 - 긴 거리 상관의 진짜 원인] 여러 시도 끝에,
          "장거리 자기상관은 진짜 상관 잡음이 아니라, 모델에 빠진
          배경(background) 성분 때문"이라는 게 확인됨:

            시도 A) 자유로운 복소수 오프셋 c를 모델에 추가
                    -> Sij가 이미 원거리에서 a*exp(i*alpha)로
                       수렴하므로, c가 (a,alpha)와 축퇴를 일으켜
                       Qc가 물리적으로 불가능한 값(≈1)으로 붕괴.

            시도 B) baseline(진폭 상위 20%) 구간의 단순 평균으로
                    (a,alpha)를 고정
                    -> 이 스캔 범위(전체 4.8 linewidth)가 공진 폭에
                       비해 충분히 넓지 않아서, "baseline"이라 부른
                       구간에도 여전히 공진 딥의 꼬리가 섞여 있었음.
                       그 결과 alpha 추정이 circle fit의 alpha와
                       0.53 라디안이나 어긋났고, phi가 잘못된
                       국소최적점(±pi 근처)으로 밀려남.

            시도 C(채택) circle fit이 이미 구한 (a,alpha)를 그대로
                    고정값으로 사용
                    -> circle fit은 원의 기하학적 성질(전체 데이터를
                       다 활용)로 (a,alpha)를 구하므로, 우리가 임의로
                       고른 부분집합의 평균보다 훨씬 정확함. 이 값을
                       고정하고 (fr,Ql,Qc,phi) 4개만 phi 다중
                       초기값(13개, -3~3 범위)으로 재피팅하니:
                         - phi가 여러 초기값에서 일관되게 -0.037로
                           수렴(전역 최적점 확정)
                         - 긴 거리(lag>=200) 자기상관이 거의 완전히
                           사라짐(0.09 -> -0.05 수준)

  STEP 7. 남은 짧은 거리(lag 1~50) 상관을 구간별(공진중심/어깨)로
          분해해 진단
          - 이 스캔 범위(4.8 linewidth) 안에는 "완전한 원거리"
            구간이 존재하지 않는다는 구조적 한계를 발견(3 linewidth
            이상 떨어진 점이 0개).
          - 상관이 공진 중심(rho=0.60)보다 어깨(rho=0.76)에서 더
            강해서, 순수한 "공진 모양을 못 맞춘 모델 오차"보다는
            "국소적으로 퍼진 잡음"에 더 가까운 패턴으로 잠정 판단.

  STEP 8 [오늘 확정, 이 스크립트가 구현하는 부분]. 위 절차(STEP 6의
          채택안)로 얻은 최종 파라미터에, 피셔행렬 기반 공분산 +
          몬테카를로 샘플링으로 신뢰구간과 corner plot을 만듦.

  [다음 세션 과제 - 오늘 미해결로 남긴 것]
  짧은 거리(lag 1~50) 상관을 AR(1)로 처리하는 것은, 어제/오늘
  겪었듯 자칫 Qc를 다시 불안정하게 만들 위험이 있어 신중하게
  접근해야 함 - 오늘은 시도하지 않고 다음 과제로 남김.
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
aa_resonator_file = 'resonator_7_powersweep_overcoupled.npz'
    # [조정] 어떤 resonator를 볼지. 다른 파일로 바꾸면 그 파일에
    # 대해 전체 파이프라인이 그대로 적용됨.
aa_power_slice_index = 0
aa_n_ports = 1.0
aa_isolation_db = 15
aa_phi_candidates = np.linspace(-3.0, 3.0, 13)
    # [조정] STEP 6에서 phi가 잘못된 국소최적점(±pi 근처)으로 새는
    # 문제를 막기 위해, 넉넉하게 13개 후보로 촘촘히 스캔.
aa_outlier_trim_pct = 5.0
aa_mc_sample_size = 30000
aa_step_fraction = 1e-6
aa_credible_level = 0.68
aa_output_dir = './outputs/'
os.makedirs(aa_output_dir, exist_ok=True)

print("=" * 60)
print("분석자 설정값(aa_*) 요약:")
for k, v in list(globals().items()):
    if k.startswith("aa_") and not isinstance(v, np.ndarray):
        print(f"  {k:24s} = {v}")
print("=" * 60)


# =========================================================
# STEP 1. 데이터 로더
# =========================================================
def load_fano_npz(filepath):
    """
    resonator_*_powersweep_overcoupled.npz 파일 하나를 읽어서
    복소수 S21과 케이블 지연(cable_delay)을 반환.
    """
    raw = np.load(filepath, allow_pickle=True)
    amplitude, phase = raw['amplitude'], raw['phase']
    freq_hz, power_dbm = raw['frequency'], raw['power']
    s21 = amplitude * np.exp(1j * phase)
    settings_dict = json.loads(raw['settings'][0])
    cable_delay = settings_dict.get('vna', {}).get('port1_edel', None)
    return {'power_dbm': power_dbm, 'freq_hz': freq_hz, 's21': s21, 'cable_delay': cable_delay}


def autocorr_at_lag(x, lag):
    """자기상관: 신호를 lag칸 밀어서 자기 자신과 비교. 백색잡음이면 0에
    가깝고, 색깔있는(상관된) 잡음이면 0에서 벗어남."""
    xc = x - np.mean(x)
    denom = np.sum(xc**2)
    return np.sum(xc[:-lag]*xc[lag:]) / denom if denom > 0 else 0.0


data = load_fano_npz(os.path.join(aa_upload_dir, aa_resonator_file))
f_grid = data['freq_hz']
s21 = data['s21'][aa_power_slice_index, :]
delay = data['cable_delay']
mag = np.abs(s21)

param_names_full = ['fr', 'Ql', 'Qc', 'phi', 'a', 'alpha']   # 참고용 전체 이름
param_names = ['fr', 'Ql', 'Qc', 'phi']   # 실제로 자유롭게 피팅할 4개


# =========================================================
# STEP 2(코드상 STEP1). circle fit으로 (a,alpha) 및 안전한 초기값 확보
# =========================================================
cf_result = fano_models.autofit(f_grid, s21, n_ports=aa_n_ports,
                                  fixed_delay=delay, isolation=aa_isolation_db)
a_fixed = cf_result['a']       # [STEP 6 채택안] circle fit의 a를 그대로 고정
alpha_fixed = cf_result['alpha']  # [STEP 6 채택안] circle fit의 alpha를 그대로 고정

print(f"\n[circle fit 참고값] fr={cf_result['fr']/1e9:.6f}GHz, "
      f"Ql={cf_result['Ql']:.1f}, Qc={cf_result['Qc']:.1f}, "
      f"a={a_fixed:.6f}, alpha={alpha_fixed:.4f}")

fixed_kwargs = {'delay': delay, 'n_ports': aa_n_ports, 'a': a_fixed, 'alpha': alpha_fixed}
    # a, alpha를 "고정 인자"로 취급 - fano_models.Sij 호출 시 매번
    # 이 값이 그대로 들어가고, 최적화 대상에서 제외됨.

prior_bounds = {
    'fr': (f_grid.min(), f_grid.max()), 'Ql': (1.0, 1e8),
    'Qc': (1.0, 1e8), 'phi': (-np.pi, np.pi),
}
log_prior = likelihood.make_uniform_log_prior(prior_bounds)
    # [라이브러리] fano_likelihood.make_uniform_log_prior.
log_probability = likelihood.make_log_probability_generic(
    fano_models.Sij, fixed_kwargs, param_names, log_prior, likelihood_type='gaussian'
)
    # [라이브러리] fano_likelihood.make_log_probability_generic - 오늘
    # 세션 초반에 범용 버전으로 확장해둔 함수. a,alpha가 fixed_kwargs에
    # 들어있으므로, theta는 (fr,Ql,Qc,phi) 4개짜리 벡터만 받으면 됨.


# =========================================================
# STEP 3(코드상 STEP2). baseline sigma로 1차 피팅 (multi-start)
# =========================================================
fr_guess = f_grid[np.argmin(mag)]
half_level = (mag.max()+mag.min())/2
above_half = np.where(mag > half_level)[0]
fwhm_guess = f_grid[above_half[-1]]-f_grid[above_half[0]] if len(above_half)>1 else 1e6
Ql_guess = fr_guess/max(fwhm_guess, 1e3)

candidates = [np.array([fr_guess, Ql_guess, Ql_guess*1.1, phi0]) for phi0 in aa_phi_candidates]
candidates.append(np.array([cf_result['fr'], cf_result['Ql'], cf_result['Qc'], cf_result['phi']]))
    # circle fit 결과도 초기값 후보로 추가 - 국소최적점 방지 안전장치.

baseline_mask = mag > np.percentile(mag, 80)
sigma_baseline = np.std(np.real(s21[baseline_mask]))


def find_map(sigma, theta0_list):
    """multi-start Nelder-Mead: 여러 초기값 후보 중 -log_prob가 가장
    낮은(=posterior가 가장 높은) 지점을 채택."""
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


popt_step2 = find_map(sigma_baseline, candidates)
model_step2 = fano_models.Sij(f_grid, *popt_step2, a_fixed, alpha_fixed, delay, n_ports=aa_n_ports)
residual_step2 = np.real(s21 - model_step2)


# =========================================================
# STEP 4~5. 이상치 5% 제거 + MAD 강건 sigma
# =========================================================
abs_res = np.abs(residual_step2)
threshold = np.percentile(abs_res, 100 - aa_outlier_trim_pct)
keep_mask = abs_res <= threshold
n_outliers = np.sum(~keep_mask)

residual_trimmed = residual_step2[keep_mask]
mad = np.median(np.abs(residual_trimmed - np.median(residual_trimmed)))
sigma_mad = mad * 1.4826
    # 1.4826: MAD를 표준편차와 같은 스케일로 맞추는 표준 보정 상수.

print(f"\n[잡음 진단] 이상치 {n_outliers}개 제거, MAD 기반 sigma={sigma_mad:.6f} "
      f"(baseline sigma={sigma_baseline:.6f}과 비교)")


# =========================================================
# STEP 6(채택안). (a,alpha) 고정 상태로, 강건한 sigma를 써서 최종 재피팅
# =========================================================
popt_final = find_map(sigma_mad, candidates)
fr_fit, Ql_fit, Qc_fit, phi_fit = popt_final
Qi_fit = 1.0/(1.0/Ql_fit - 1.0/Qc_fit) if Ql_fit != Qc_fit else np.nan

print(f"\n[최종 파라미터] fr={fr_fit/1e9:.6f}GHz, Ql={Ql_fit:.1f}, Qc={Qc_fit:.1f}, "
      f"phi={phi_fit:.4f}")
print(f"  Qi={Qi_fit:.1f}")

# 잔차 재검증 - 긴 거리 자기상관이 해소됐는지 확인
model_final = fano_models.Sij(f_grid, *popt_final, a_fixed, alpha_fixed, delay, n_ports=aa_n_ports)
residual_final = np.real(s21 - model_final)
print(f"\n[검증] 최종 잔차의 lag별 자기상관 (긴 거리가 백색화됐는지 확인):")
for lag in [1, 20, 50, 100, 200, 300]:
    if lag < len(residual_final):
        print(f"  lag={lag:4d}: rho={autocorr_at_lag(residual_final, lag):.4f}")


# =========================================================
# STEP 8. 피셔행렬 -> 공분산 -> 몬테카를로 샘플 (=corner plot 재료)
# =========================================================
F = fm.fisher_information_matrix(fano_models.Sij, popt_final, param_names, f_grid,
                                    sigma_mad, fixed_kwargs, step_fraction=aa_step_fraction)
    # [라이브러리] fano_fisher_matrix.fisher_information_matrix -
    # (fr,Ql,Qc,phi) 4x4 피셔행렬을 수치미분으로 계산.
cov = fm.covariance_from_fisher(F)
eigvals = np.linalg.eigvalsh(cov)
print(f"\n공분산 행렬 최소 고유값: {eigvals.min():.3e} "
      f"({'정상(양수)' if eigvals.min()>0 else '⚠️ 비정상'})")

mc_samples = np.random.default_rng(0).multivariate_normal(popt_final, cov, size=aa_mc_sample_size)
    # 공분산 행렬을 가진 다변량 정규분포에서 표본을 뽑는 것 - emcee
    # 없이도 MCMC posterior와 개념적으로 동일한 정보를 주는 방법
    # (Laplace 근사).

Qi_mc = 1.0/(1.0/mc_samples[:,1] - 1.0/mc_samples[:,2])
valid_qi = (Qi_mc>0) & (Qi_mc<1e7)
Qi_lo, Qi_med, Qi_hi = bt.credible_interval(Qi_mc[valid_qi], level=aa_credible_level)
    # [라이브러리] fano_bayesian_toolkit.credible_interval.
print(f"\n[최종 결과] Qi = {Qi_med:.1f}  [{Qi_lo:.1f}, {Qi_hi:.1f}] "
      f"({aa_credible_level*100:.0f}% 신뢰구간, 폭={Qi_hi-Qi_lo:.1f})")


# =========================================================
# STEP 9. Corner plot (fr, Ql, Qc, phi 4개 + Qi 파생량 함께 표시)
# =========================================================
def simple_corner_plot(samples, labels, truths=None, bins=40):
    """
    corner 패키지 없이 matplotlib만으로 만드는 corner plot. 대각선에는
    각 파라미터의 히스토그램(1차원 분포), 대각선 아래 삼각형에는 두
    파라미터 쌍의 2차원 산점도(상관관계)를 그림.
    """
    n = samples.shape[1]
    fig, axes = plt.subplots(n, n, figsize=(2.3*n, 2.3*n))
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


# fr,Ql,Qc,phi 4개 + Qi(파생량)를 합쳐서 5개짜리 corner plot으로 확장
samples_with_qi = np.column_stack([mc_samples, Qi_mc])
labels_with_qi = [r'$f_r$', r'$Q_l$', r'$Q_c$', r'$\phi$', r'$Q_i$']
truths_with_qi = list(popt_final) + [Qi_fit]

fig = simple_corner_plot(samples_with_qi, labels_with_qi, truths=truths_with_qi)
fig.suptitle(f'{aa_resonator_file}\n최종 파이프라인: circle fit (a,alpha)고정 + '
             f'5%이상치제거 + MAD sigma + 피셔행렬 Corner Plot',
             y=1.01, fontsize=11)
plt.savefig(os.path.join(aa_output_dir, 'fano_final_estimation_corner.png'), dpi=140, bbox_inches='tight')
print(f"\nCorner plot 저장 완료: "
      f"{os.path.join(aa_output_dir, 'fano_final_estimation_corner.png')}")

print("\n" + "="*60)
print("[다음 세션 과제 - 오늘 미해결로 남긴 것]")
print("  짧은 거리(lag 1~50) 자기상관이 여전히 남아있습니다(rho~0.6~0.9).")
print("  이를 AR(1)로 처리하는 것은 가능하지만, Qc를 다시 불안정하게")
print("  만들 위험이 있어(어제 경험) 신중한 접근이 필요합니다.")
print("="*60)
