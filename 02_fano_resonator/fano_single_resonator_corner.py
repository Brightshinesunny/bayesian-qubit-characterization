"""
fano_single_resonator_corner.py
=================================================================
9개 resonator 중 하나를 골라, 베이지안/피셔 방식(curve_fit +
multi-start + 피셔행렬 기반 공분산)으로 분석하고, 그 결과를
"corner plot"(파라미터 여러 개의 상관관계를 한 번에 보여주는 격자
형태 그래프)으로 시각화합니다.

[emcee 없이 corner plot을 만드는 방법 - 왜 이렇게 하는가]
이 환경에는 emcee(진짜 MCMC 샘플러)도, corner(전용 시각화
패키지)도 설치되어 있지 않고 네트워크도 막혀 있어 설치가
불가능합니다. 대신 다음 방식을 씁니다:
  1. curve_fit으로 최적점(popt)을 구함 (이게 "posterior의 최댓값"에
     해당 - 어제/오늘 계속 설명한 대로, uniform prior + gaussian
     likelihood 조합에서는 MAP=최소제곱 해와 정확히 같음)
  2. 피셔행렬로 그 최적점 주변의 곡률(공분산)을 구함
  3. 공분산 행렬을 가진 다변량 정규분포에서 수만 개의 가상 표본을
     뽑음 - 이게 "posterior가 정확히 가우시안이라고 가정했을 때의
     MCMC 샘플"과 수학적으로 동일합니다(선형화 근사, Laplace
     approximation이라고 부르는 표준적인 기법)
  4. 이 표본들을 파라미터 쌍마다 산점도로, 각 파라미터는 히스토그램
     으로 그리면 -> 이게 정확히 corner plot의 정의입니다.
matplotlib.pyplot.subplots로 N x N 격자를 직접 만들어서 구현합니다.

[분석자가 조정할 수 있는 설정값들 - 이 스크립트의 목적]
파일 맨 위 STEP 0에 있는 aa_ 변수들을 바꿔가며, 어떤 resonator를
볼지, multi-start 후보를 얼마나 촘촘히 할지, 몬테카를로 샘플
개수를 얼마나 크게 할지 등을 자유롭게 실험해볼 수 있습니다.
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import json

sys.path.insert(0, os.getcwd())
import fano_models
import fano_fisher_matrix as fm


# =========================================================
# STEP 0. 분석자 설정값 (aa_ 접두사) - 여기를 바꿔가며 실험해보세요
# =========================================================
aa_upload_dir = '/mnt/user-data/uploads/'
aa_resonator_choice = 'resonator_7_powersweep_overcoupled.npz'
    # [조정 가능] 9개 파일 중 하나를 골라 자세히 봄. 오늘 요약표에서
    # "베이지안 68%폭이 가장 넓었던"(=가장 불확실했던) resonator_7을
    # 기본값으로 골랐음 - corner plot에서 파라미터 간 상관관계
    # (축퇴)가 가장 뚜렷하게 보일 가능성이 높은 케이스.

aa_power_slice_index = 0     # [조정 가능] 어느 전력 slice를 볼지
aa_n_ports = 1.0              # [조정 가능] reflection(1.0) vs notch(2.0)
aa_isolation_db = 15          # [조정 가능] circle fit 참고용 isolation 가정값
aa_step_fraction = 1e-6       # [조정 가능] 피셔행렬 수치미분 스텝 크기.
    # 너무 크면(예: 1e-4) fr처럼 절대 스케일이 큰 파라미터의 미분이
    # 부정확해짐(어제/오늘 여러 번 확인한 문제). 너무 작으면(예:
    # 1e-10) 반대로 부동소수점 반올림 오차 때문에 미분이 불안정해질
    # 수 있음 - 1e-6~1e-8 사이가 보통 안전한 절충점.
aa_phi_candidates = [-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5]
    # [조정 가능] multi-start에 쓸 phi 초기값 후보들. 더 촘촘하게
    # (예: -3~3 사이를 0.5 간격 대신 0.2 간격으로) 늘리면 국소최적점을
    # 놓칠 확률이 줄어들지만, 그만큼 계산 시간이 늘어남 - 정확도와
    # 속도의 트레이드오프.
aa_mc_sample_size = 30000     # [조정 가능] 공분산 행렬에서 뽑을 몬테카를로
    # 표본 개수. 많을수록 히스토그램이 매끈해지지만 계산이 느려짐.

aa_output_dir = '/home/claude/qubit_analysis_template/outputs/'
os.makedirs(aa_output_dir, exist_ok=True)

print("=" * 60)
print("분석자 설정값(aa_*) 요약:")
for k, v in list(globals().items()):
    if k.startswith("aa_") and not isinstance(v, list):
        print(f"  {k:24s} = {v}")
print(f"  aa_phi_candidates        = {aa_phi_candidates}")
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
# STEP 2. Circle fit (참고용 - 베이지안 결과와 비교하기 위해)
# =========================================================
cf_result = fano_models.autofit(f_grid, s21, n_ports=aa_n_ports, fixed_delay=delay, isolation=aa_isolation_db)
print(f"\n[참고] circle fit 결과:")
print(f"  fr={cf_result['fr']/1e9:.6f}GHz, Ql={cf_result['Ql']:.1f}, Qc={cf_result['Qc']:.1f}, "
      f"Qi={cf_result['Qi']:.1f}, phi={cf_result['phi']:.4f}")
qi_max_str = 'inf' if np.isinf(cf_result['Qi_max']) else f"{cf_result['Qi_max']:.1f}"
print(f"  Qi_min={cf_result['Qi_min']:.1f}, Qi_max={qi_max_str}")


# =========================================================
# STEP 3. 베이지안/피셔 분석 (multi-start curve_fit)
# =========================================================
param_names = ['fr', 'Ql', 'Qc', 'phi', 'a', 'alpha']
param_labels = [r'$f_r$', r'$Q_l$', r'$Q_c$', r'$\phi$', r'$a$', r'$\alpha$']

mask = mag > np.percentile(mag, 80)
noise_sigma = np.std(np.real(s21[mask]))
print(f"\nnoise_sigma = {noise_sigma:.6f}")

fr_guess = f_grid[np.argmin(mag)]
half_level = (mag.max() + mag.min()) / 2
above_half = np.where(mag > half_level)[0]
fwhm_guess = f_grid[above_half[-1]] - f_grid[above_half[0]] if len(above_half) > 1 else 1e6
Ql_guess = fr_guess / max(fwhm_guess, 1e3)

def model_wrapper(f, fr, Ql, Qc, phi, a, alpha):
    m = fano_models.Sij(f, fr, Ql, Qc, phi, a, alpha, delay, n_ports=aa_n_ports)
    return np.concatenate([np.real(m), np.imag(m)])

y_data = np.concatenate([np.real(s21), np.imag(s21)])
sigma_for_fit = np.full_like(y_data, noise_sigma)
lower_bounds = [f_grid.min(), 1.0, 1.0, -np.pi, 1e-8, -np.pi]
upper_bounds = [f_grid.max(), 1e8, 1e8, np.pi, 1.0, np.pi]

print(f"\nmulti-start curve_fit 실행 중 (phi 후보 {len(aa_phi_candidates)}개)...")
best_popt, best_chi2 = None, np.inf
for phi0 in aa_phi_candidates:
    p0_trial = [fr_guess, Ql_guess, Ql_guess * 1.1, phi0, mag.max() - mag.min(), phi0]
    try:
        popt_trial, _ = curve_fit(
            model_wrapper, f_grid, y_data, p0=p0_trial,
            sigma=sigma_for_fit, absolute_sigma=True,
            bounds=(lower_bounds, upper_bounds), maxfev=30000
        )
        model_trial = fano_models.Sij(f_grid, *popt_trial, delay, n_ports=aa_n_ports)
        residual = s21 - model_trial
        chi2_trial = np.sum((np.real(residual)/noise_sigma)**2 + (np.imag(residual)/noise_sigma)**2)
        print(f"  phi0={phi0:+.1f} -> chi2={chi2_trial:.2f}")
        if chi2_trial < best_chi2:
            best_chi2 = chi2_trial
            best_popt = popt_trial
    except Exception as e:
        print(f"  phi0={phi0:+.1f} -> 실패 ({e})")

popt = best_popt
print(f"\n최종 채택 (chi2={best_chi2:.2f}):")
for i, name in enumerate(param_names):
    print(f"  {name}: {popt[i]:.6g}")


# =========================================================
# STEP 4. 피셔행렬 -> 공분산 -> 몬테카를로 샘플 (=corner plot 재료)
# =========================================================
fixed_kwargs = {'delay': delay, 'n_ports': aa_n_ports}
F = fm.fisher_information_matrix(fano_models.Sij, popt, param_names, f_grid,
                                    noise_sigma, fixed_kwargs, step_fraction=aa_step_fraction)
cov = fm.covariance_from_fisher(F)

eigvals = np.linalg.eigvalsh(cov)
print(f"\n공분산 행렬 최소 고유값: {eigvals.min():.3e} "
      f"({'정상(양수)' if eigvals.min() > 0 else '⚠️ 비정상(음수) - 결과 주의'})")

mc_samples = np.random.default_rng(0).multivariate_normal(popt, cov, size=aa_mc_sample_size)

Qi_mc = 1.0 / (1.0/mc_samples[:,1] - 1.0/mc_samples[:,2])
valid_qi = (Qi_mc > 0) & (Qi_mc < 1e7)
print(f"\nQi (베이지안/피셔): 중앙값={np.median(Qi_mc[valid_qi]):.1f}, "
      f"[16%,84%]=[{np.percentile(Qi_mc[valid_qi],16):.1f}, {np.percentile(Qi_mc[valid_qi],84):.1f}]")


# =========================================================
# STEP 5. 손수 그리는 corner plot
# =========================================================
def simple_corner_plot(samples, labels, truths=None, bins=40):
    """
    corner 패키지 없이 matplotlib만으로 만드는 간단한 corner plot.
    대각선에는 각 파라미터의 히스토그램(1차원 분포), 대각선 아래
    삼각형에는 두 파라미터 쌍의 2차원 산점도(상관관계)를 그림 -
    이게 corner plot의 표준적인 정의.
    """
    n = samples.shape[1]
    fig, axes = plt.subplots(n, n, figsize=(2.2*n, 2.2*n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                # 대각선 위쪽 삼각형은 정보 중복이므로 그리지 않고 숨김
                ax.axis('off')
                continue
            if i == j:
                ax.hist(samples[:, i], bins=bins, color='tab:blue', alpha=0.7)
                if truths is not None:
                    ax.axvline(truths[i], color='red', ls='--', lw=1)
            else:
                ax.plot(samples[:, j], samples[:, i], '.', ms=1, alpha=0.05, color='tab:blue')
                    # alpha를 아주 낮게(0.05) 줘서, 점이 수만 개 겹쳐도
                    # 밀도가 높은 곳이 자연스럽게 진하게 보이도록 함
                    # (전용 corner 패키지의 등고선 표시를 산점도로 흉내).
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
fig.suptitle(f'{aa_resonator_choice}\n베이지안/피셔 Corner Plot (multi-start curve_fit + 공분산 샘플링)',
             y=1.01, fontsize=11)
plt.savefig(os.path.join(aa_output_dir, 'fano_single_resonator_corner.png'), dpi=140, bbox_inches='tight')
print(f"\nCorner plot 저장 완료: {os.path.join(aa_output_dir, 'fano_single_resonator_corner.png')}")
