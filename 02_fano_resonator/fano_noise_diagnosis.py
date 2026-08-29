"""
fano_noise_diagnosis.py
=================================================================
[목적] 지금까지는 잡음 크기(noise_sigma)를 "baseline(공진에서 먼
구간)의 실수부 표준편차"로 단순하게 추정했습니다. 이번엔 한 단계 더
나아가, 최적 모델로 피팅한 뒤 남는 "잔차(residual = 실측 - 모델)"를
직접 진단해서:
  1. 잔차가 정말 순수한 백색잡음(정규분포, 서로 무관)인지
  2. 이상치(outlier)가 섞여 있는지
  3. 이웃한 주파수 포인트끼리 잡음이 서로 얽혀있는지(색깔있는 잡음,
     colored noise / 1/f 잡음)
를 확인하고, 그 결과에 맞춰 더 정확한 잡음 추정 방법(강건한 MAD 기반
sigma, 필요하면 robust 우도)으로 피셔/베이지안 분석을 다시 합니다.

[용어 정리]
  백색잡음(white noise): 각 데이터 포인트의 잡음이 서로 완전히
    독립적이고, 크기 분포가 정규분포를 따르는 "가장 단순하고 이상적인"
    잡음. 우리가 지금까지 암묵적으로 가정해온 형태.
  이상치(outlier): 대부분의 점과 달리 유난히 크게 튀는 소수의 점.
    측정 중 순간적인 간섭, 트리거 오류 등으로 생길 수 있음.
  색깔있는 잡음(colored noise) / 1/f 잡음: 이웃한 점끼리 잡음이
    서로 관련되어 있는 경우(예: 온도 드리프트처럼 천천히 변하는
    요인). "백색"이 모든 주파수 성분이 고르게 섞인 잡음이라면,
    "색깔있는" 잡음은 특정 주파수(여기서는 "이웃 데이터 포인트
    사이의 거리"라는 의미의 주파수) 성분이 유난히 강한 잡음.
  자기상관(autocorrelation): 신호(또는 잔차)를 한 칸씩 밀어서(lag)
    자기 자신과 비교했을 때 얼마나 닮았는지를 나타내는 지표. 백색잡음
    이라면 바로 옆 점과도 전혀 안 닮아야 하므로 자기상관이 0에
    가까워야 하고, 1/f 잡음처럼 서서히 변하는 성분이 있으면 이웃한
    점끼리 닮아서 자기상관이 0보다 뚜렷하게 커짐.
  MAD(Median Absolute Deviation, 중앙값 절대편차): 표준편차(std)의
    "강건한(robust)" 대안. std는 이상치 하나만 있어도 크게 부풀려질
    수 있는데(제곱해서 더하므로), MAD는 중앙값을 기준으로 편차의
    "중앙값"을 쓰므로 이상치 몇 개에 거의 영향을 안 받음.
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
aa_resonator_choice = 'resonator_7_powersweep_overcoupled.npz'
aa_power_slice_index = 0
aa_n_ports = 1.0
aa_step_fraction = 1e-6
aa_phi_candidates = [-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5]
aa_outlier_threshold_sigma = 5.0
    # [조정 가능] 이 배수(x sigma)보다 잔차가 큰 점을 "이상치 의심"으로
    # 표시. 5-sigma는 통계에서 흔히 쓰는 "웬만해선 우연히 안 나오는"
    # 엄격한 기준(정규분포에서 5-sigma를 넘을 확률은 약 0.00006%).
aa_mc_sample_size = 30000
aa_output_dir = '/home/claude/qubit_analysis_template/outputs/'
os.makedirs(aa_output_dir, exist_ok=True)

print("=" * 60)
print("분석자 설정값(aa_*) 요약:")
for k, v in list(globals().items()):
    if k.startswith("aa_") and not isinstance(v, list):
        print(f"  {k:24s} = {v}")
print("=" * 60)


# =========================================================
# STEP 1. 데이터 로드 및 (1차) baseline 기반 잡음 추정으로 MAP 탐색
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

baseline_mask = mag > np.percentile(mag, 80)
noise_sigma_baseline = np.std(np.real(s21[baseline_mask]))
    # [1차, 기존 방식] 이전까지 계속 써온 "baseline 구간 표준편차" 추정.
    # 이 값으로 일단 MAP을 먼저 찾고, 그 잔차를 STEP 2에서 진단함.

log_probability = likelihood.make_log_probability_generic(
    fano_models.Sij, fixed_kwargs, param_names, log_prior, likelihood_type='gaussian'
)

fr_guess = f_grid[np.argmin(mag)]
half_level = (mag.max() + mag.min()) / 2
above_half = np.where(mag > half_level)[0]
fwhm_guess = f_grid[above_half[-1]] - f_grid[above_half[0]] if len(above_half) > 1 else 1e6
Ql_guess = fr_guess / max(fwhm_guess, 1e3)

best_theta, best_neg_lp = None, np.inf
for phi0 in aa_phi_candidates:
    theta0 = np.array([fr_guess, Ql_guess, Ql_guess*1.1, phi0, mag.max()-mag.min(), phi0])
    def neg_log_prob(theta, sigma=noise_sigma_baseline):
        val = log_probability(theta, f_grid, s21, sigma)
        return -val if np.isfinite(val) else 1e10
    res = minimize(neg_log_prob, theta0, method='Nelder-Mead',
                    options={'maxiter': 20000, 'xatol': 1e-6, 'fatol': 1e-6})
    if res.fun < best_neg_lp:
        best_neg_lp = res.fun
        best_theta = res.x

popt_v1 = best_theta
print(f"\n[1차 MAP - baseline noise_sigma={noise_sigma_baseline:.6f} 사용]")
for i, name in enumerate(param_names):
    print(f"  {name}: {popt_v1[i]:.6g}")


# =========================================================
# STEP 2. 잔차 진단 - 이 모델이 데이터를 얼마나 잘 설명하는지, 남은
# 잔차에 어떤 잡음 구조가 있는지 확인
# =========================================================
model_v1 = fano_models.Sij(f_grid, *popt_v1, delay, n_ports=aa_n_ports)
residual_complex = s21 - model_v1
residual_real = np.real(residual_complex)
residual_imag = np.imag(residual_complex)

print(f"\n[잔차 진단]")
print(f"  잔차(실수부) 평균: {residual_real.mean():.6f} (0에 가까워야 정상 - 편향 없음을 시사)")
print(f"  잔차(실수부) 표준편차: {residual_real.std():.6f}")

# --- 진단 A: MAD 기반 강건한 sigma (이상치에 덜 흔들리는 추정) ---
mad_real = np.median(np.abs(residual_real - np.median(residual_real)))
sigma_robust = mad_real * 1.4826
    # 1.4826: MAD를 표준편차와 같은 스케일로 맞춰주는 표준적인 보정 상수
    # (정규분포를 가정했을 때, MAD*1.4826 ≈ 표준편차가 되도록 유도된 값 -
    # 통계학에서 "consistency constant"라고 부르는 관례적인 계수).
print(f"  MAD 기반 강건한 sigma: {sigma_robust:.6f} "
      f"(단순 표준편차 {residual_real.std():.6f}와 차이가 크면 이상치 존재 의심)")

# --- 진단 B: 이상치 탐지 ---
z_scores = np.abs(residual_real) / sigma_robust
outlier_mask = z_scores > aa_outlier_threshold_sigma
n_outliers = np.sum(outlier_mask)
print(f"  {aa_outlier_threshold_sigma}-sigma 이상치 의심 포인트: {n_outliers}개 / {len(f_grid)}개")
if n_outliers > 0:
    print(f"    위치(주파수): {np.round(f_grid[outlier_mask][:10]/1e9, 6)} GHz (최대 10개만 표시)")

# --- 진단 C: 자기상관(lag-1) - 색깔있는 잡음(1/f 등)이 있는지 확인 ---
def lag1_autocorrelation(x):
    """
    잔차 배열 x에서, 바로 이웃한 두 점끼리의 상관계수(lag=1 자기상관)를
    계산. 0에 가까우면 "이웃 점끼리 무관"(백색잡음다움), 0.3~0.5
    이상이면 "이웃 점끼리 서로 닮아있음"(색깔있는/1/f 잡음 의심).
    """
    x_centered = x - np.mean(x)
    numerator = np.sum(x_centered[:-1] * x_centered[1:])
    denominator = np.sum(x_centered**2)
    return numerator / denominator if denominator > 0 else 0.0

ac_real = lag1_autocorrelation(residual_real)
ac_imag = lag1_autocorrelation(residual_imag)
print(f"  잔차(실수부) lag-1 자기상관: {ac_real:.4f} "
      f"({'백색잡음에 가까움' if abs(ac_real) < 0.2 else '⚠️ 색깔있는 잡음(상관 존재) 의심'})")
print(f"  잔차(허수부) lag-1 자기상관: {ac_imag:.4f}")


# =========================================================
# STEP 3. 진단 결과에 따라 (2차) sigma로 MAP 재탐색
# =========================================================
noise_sigma_v2 = sigma_robust
    # MAD 기반 강건한 값을 "최종적으로 신뢰할 잡음 크기"로 채택.
    # (만약 STEP 2에서 자기상관이 컸다면, 여기서 더 나아가 상관구조를
    # 반영한 공분산 행렬을 쓰는 것이 이상적이지만, 그건 훨씬 복잡한
    # 확장이라 오늘은 "백색잡음 가정 + 강건한 sigma"까지만 다룸.)

best_theta2, best_neg_lp2 = None, np.inf
for phi0 in aa_phi_candidates:
    theta0 = np.array([fr_guess, Ql_guess, Ql_guess*1.1, phi0, mag.max()-mag.min(), phi0])
    def neg_log_prob2(theta, sigma=noise_sigma_v2):
        val = log_probability(theta, f_grid, s21, sigma)
        return -val if np.isfinite(val) else 1e10
    res = minimize(neg_log_prob2, theta0, method='Nelder-Mead',
                    options={'maxiter': 20000, 'xatol': 1e-6, 'fatol': 1e-6})
    if res.fun < best_neg_lp2:
        best_neg_lp2 = res.fun
        best_theta2 = res.x

popt_v2 = best_theta2
print(f"\n[2차 MAP - 진단으로 얻은 강건한 noise_sigma={noise_sigma_v2:.6f} 사용]")
for i, name in enumerate(param_names):
    diff_pct = abs(popt_v2[i]-popt_v1[i])/abs(popt_v1[i])*100 if popt_v1[i]!=0 else 0
    print(f"  {name}: {popt_v2[i]:.6g}  (1차 대비 {diff_pct:.2f}% 변화)")


# =========================================================
# STEP 4. 두 버전의 Qi 신뢰구간 비교 (잡음 추정 개선 전후)
# =========================================================
def get_qi_credible_interval(popt, sigma):
    F = fm.fisher_information_matrix(fano_models.Sij, popt, param_names, f_grid,
                                        sigma, fixed_kwargs, step_fraction=aa_step_fraction)
    cov = fm.covariance_from_fisher(F)
    mc = np.random.default_rng(0).multivariate_normal(popt, cov, size=aa_mc_sample_size)
    Qi_mc = 1.0/(1.0/mc[:,1] - 1.0/mc[:,2])
    valid = (Qi_mc>0) & (Qi_mc<1e7)
    lo, med, hi = bt.credible_interval(Qi_mc[valid], level=0.68)
    return lo, med, hi, mc

Qi_lo1, Qi_med1, Qi_hi1, mc1 = get_qi_credible_interval(popt_v1, noise_sigma_baseline)
Qi_lo2, Qi_med2, Qi_hi2, mc2 = get_qi_credible_interval(popt_v2, noise_sigma_v2)

print(f"\n[Qi 신뢰구간 비교]")
print(f"  1차(baseline sigma={noise_sigma_baseline:.6f}): Qi={Qi_med1:.1f} [{Qi_lo1:.1f},{Qi_hi1:.1f}]")
print(f"  2차(강건한 sigma={noise_sigma_v2:.6f})     : Qi={Qi_med2:.1f} [{Qi_lo2:.1f},{Qi_hi2:.1f}]")


# =========================================================
# STEP 5. 시각화 - 잔차 진단 결과를 한눈에
# =========================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0,0].plot(f_grid/1e9, residual_real, '.', ms=3, alpha=0.5, label='잔차(실수부)')
if n_outliers > 0:
    axes[0,0].plot(f_grid[outlier_mask]/1e9, residual_real[outlier_mask], 'rx', ms=10, label='이상치 의심')
axes[0,0].axhline(0, color='gray', ls='--', lw=1)
axes[0,0].set_xlabel('Frequency (GHz)')
axes[0,0].set_ylabel('잔차 (실수부)')
axes[0,0].set_title('잔차 vs 주파수 (구조가 있으면 모델이 놓친 물리가 있다는 신호)')
axes[0,0].legend(fontsize=8)

axes[0,1].hist(residual_real, bins=50, color='tab:blue', alpha=0.7, density=True)
xs = np.linspace(residual_real.min(), residual_real.max(), 200)
gaussian_fit = (1/(sigma_robust*np.sqrt(2*np.pi))) * np.exp(-0.5*(xs/sigma_robust)**2)
axes[0,1].plot(xs, gaussian_fit, 'r-', lw=2, label='MAD 기반 정규분포')
axes[0,1].set_xlabel('잔차 값')
axes[0,1].set_ylabel('밀도')
axes[0,1].set_title('잔차 히스토그램 (정규분포와 얼마나 닮았는지)')
axes[0,1].legend(fontsize=8)

axes[1,0].hist(mc1[:,1]/1e0, bins=60, alpha=0.5, color='tab:orange', label=f'1차(baseline sigma)', density=True)
axes[1,0].hist(mc2[:,1]/1e0, bins=60, alpha=0.5, color='tab:green', label=f'2차(강건한 sigma)', density=True)
axes[1,0].set_xlabel('Ql')
axes[1,0].set_ylabel('밀도')
axes[1,0].set_title('Ql posterior 비교 (잡음 추정 방법 전후)')
axes[1,0].legend(fontsize=8)

lags = np.arange(1, 30)
def autocorr_at_lag(x, lag):
    xc = x - np.mean(x)
    return np.sum(xc[:-lag]*xc[lag:]) / np.sum(xc**2)
ac_values = [autocorr_at_lag(residual_real, l) for l in lags]
axes[1,1].bar(lags, ac_values, color='tab:purple')
axes[1,1].axhline(0, color='gray', lw=1)
axes[1,1].axhline(0.2, color='red', ls='--', lw=1, label='의심 기준선(±0.2)')
axes[1,1].axhline(-0.2, color='red', ls='--', lw=1)
axes[1,1].set_xlabel('lag')
axes[1,1].set_ylabel('자기상관 계수')
axes[1,1].set_title('잔차의 자기상관 함수 (색깔있는 잡음 여부 확인)')
axes[1,1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(aa_output_dir, 'fano_noise_diagnosis.png'), dpi=140, bbox_inches='tight')
print(f"\n진단 시각화 저장 완료: {os.path.join(aa_output_dir, 'fano_noise_diagnosis.png')}")
