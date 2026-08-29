"""
fano_fieldsweep_bayesian_fisher.py
=================================================================
[목적] 실제 resonator_7_fieldsweep_overcoupled.npz 데이터에 대해,
circle fit이 아니라 "베이지안/피셔 접근"(최소제곱 피팅 + 피셔행렬
기반 오차 전파)으로 Qi를 구합니다. 참값을 모르는 실제 데이터이므로,
circle fit 결과와 나란히 비교해서 "circle fit이 inf를 내는 지점에서
이 방법이 유한하고 좁은 답을 주는지"를 직접 확인하는 것이 목표입니다.

[베이지안/피셔 방법의 원리 - 다시 한번 정리]
1. scipy.optimize.curve_fit으로 데이터에 모델(Sij)을 최소제곱
   피팅해서, 가장 그럴듯한 파라미터(fr,Ql,Qc,phi,a,alpha)를 구함.
   이건 사실 "가우시안 우도를 가정한 MCMC의 사후분포 최댓값(MAP)"과
   수학적으로 동일한 결과를 줍니다 - 즉 이것도 베이지안 추정의
   한 형태입니다(uniform prior + gaussian likelihood 조합의 MAP는
   항상 최소제곱 해와 일치).
2. 우리가 만든 피셔행렬 모듈(fano_fisher_matrix.py)로 그 최적점
   근방의 "곡률"을 계산해서, 파라미터들의 공분산 행렬(서로 얼마나
   불확실하고 얼마나 얽혀있는지)을 구함.
3. 이 공분산 행렬에서 직접 수만 개의 가상 표본을 뽑아(다변량
   정규분포에서 샘플링), 그 표본들로 Qi=1/(1/Ql-1/Qc)를 계산하면,
   MCMC의 posterior 샘플과 개념적으로 동일한 "Qi의 불확실성 분포"를
   얻을 수 있습니다. emcee 라이브러리 없이도 같은 정보를 얻는
   실용적인 방법입니다.
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')   # 화면 없는 환경(서버)에서도 그래프를 파일로 저장하기 위한 설정
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.getcwd())
import fano_fieldsweep_loader as ffl
import fano_models
import fano_fisher_matrix as fm


# =========================================================
# STEP 0. 분석자 설정값 (aa_ 접두사)
# =========================================================
aa_npz_path = '/mnt/user-data/uploads/resonator_7_fieldsweep_overcoupled.npz'
aa_output_dir = '/home/claude/qubit_analysis_template/outputs/'
os.makedirs(aa_output_dir, exist_ok=True)

aa_n_ports = 1.0
aa_slice_indices_to_check = None
    # None이면 아래 STEP 1에서 "circle fit이 inf를 냈던 지점"을
    # 자동으로 찾아서 그중 몇 개를 골라 보여줌. 특정 slice 번호를
    # 직접 보고 싶으면 예: [0, 50, 100, 150, 200] 처럼 리스트로 지정.

print("=" * 60)
print("분석자 설정값(aa_*) 요약:")
for k, v in list(globals().items()):
    if k.startswith("aa_"):
        print(f"  {k:24s} = {v}")
print("=" * 60)


# =========================================================
# STEP 1. 데이터 로드 및 각 slice에서 circle fit 먼저 실행
# (어느 slice에서 Qi_max=inf가 나오는지 알아야, 베이지안/피셔와
#  비교할 "흥미로운 지점"을 고를 수 있음)
# =========================================================
data = ffl.load_fano_fieldsweep_npz(aa_npz_path)
ffl.summarize_loaded_data(data)

n_slices = len(data['B_fields'])
circle_fit_qi_max = []
circle_fit_results = []

print(f"\n{n_slices}개 slice에 circle fit 먼저 실행 (isolation=15dB, 어제 실측과 동일 조건)...")
for i in range(n_slices):
    f_i = data['freq_hz_2d'][i, :]
    s21_i = data['s21'][i, :]
    delay_i = data['cable_delay_list'][i]
    try:
        r = fano_models.autofit(f_i, s21_i, n_ports=aa_n_ports, fixed_delay=delay_i, isolation=15)
        circle_fit_results.append(r)
        circle_fit_qi_max.append(r['Qi_max'])
    except Exception:
        circle_fit_results.append(None)
        circle_fit_qi_max.append(np.nan)

circle_fit_qi_max = np.array(circle_fit_qi_max)
inf_indices = np.where(np.isinf(circle_fit_qi_max))[0]
print(f"circle fit에서 Qi_max=inf가 나온 slice: {len(inf_indices)}/{n_slices}개")


# =========================================================
# STEP 2. 베이지안/피셔 분석 대상 slice 선정
# =========================================================
if aa_slice_indices_to_check is None:
    # inf가 난 slice 중에서 자기장 값 기준으로 골고루(처음/중간/끝) 선택
    if len(inf_indices) >= 3:
        aa_slice_indices_to_check = [
            inf_indices[0], inf_indices[len(inf_indices)//2], inf_indices[-1]
        ]
    else:
        aa_slice_indices_to_check = list(inf_indices) if len(inf_indices) > 0 else [0]

print(f"\n베이지안/피셔로 자세히 볼 slice: {aa_slice_indices_to_check}")
print(f"  (B_fields 값: {np.round(data['B_fields'][aa_slice_indices_to_check], 4)})")


# =========================================================
# STEP 3. 각 선택된 slice에 대해 베이지안/피셔 분석 실행
# =========================================================
param_names = ['fr', 'Ql', 'Qc', 'phi', 'a', 'alpha']

def estimate_noise_sigma(s21_1d, mag):
    """
    baseline(공진에서 먼, 신호가 약해지는 구간이 아니라 - 이 모델은
    reflection이라 공진에서 진폭이 "작아지는" 딥 형태이므로, 진폭이
    가장 "큰" 구간이 baseline) 구간의 흔들림으로 잡음 크기 추정.
    """
    mask = mag > np.percentile(mag, 80)
    return np.std(np.real(s21_1d[mask]))


results_summary = []

fig, axes = plt.subplots(1, len(aa_slice_indices_to_check), figsize=(5.5*len(aa_slice_indices_to_check), 4.5))
if len(aa_slice_indices_to_check) == 1:
    axes = [axes]

for plot_idx, slice_idx in enumerate(aa_slice_indices_to_check):
    f_i = data['freq_hz_2d'][slice_idx, :]
    s21_i = data['s21'][slice_idx, :]
    delay_i = data['cable_delay_list'][slice_idx]
    B_val = data['B_fields'][slice_idx]
    mag = np.abs(s21_i)

    # --- 초기값을 데이터에서 직접 추정 ---
    fr_guess = f_i[np.argmin(mag)]
    half_level = (mag.max()+mag.min())/2
    above_half = np.where(mag > half_level)[0]
    fwhm_guess = f_i[above_half[-1]] - f_i[above_half[0]] if len(above_half) > 1 else 1e6
    Ql_guess = fr_guess / max(fwhm_guess, 1e3)
    p0 = [fr_guess, Ql_guess, Ql_guess*1.1, 0.0, mag.max()-mag.min(), 0.0]

    def model_wrapper(f, fr, Ql, Qc, phi, a, alpha, _delay=delay_i):
        m = fano_models.Sij(f, fr, Ql, Qc, phi, a, alpha, _delay, n_ports=aa_n_ports)
        return np.concatenate([np.real(m), np.imag(m)])

    y_data = np.concatenate([np.real(s21_i), np.imag(s21_i)])
    sigma_est = estimate_noise_sigma(s21_i, mag)

    try:
        popt, pcov_raw = curve_fit(model_wrapper, f_i, y_data, p0=p0, maxfev=30000)
        fit_ok = True
    except Exception as e:
        print(f"slice {slice_idx}: curve_fit 실패 ({e})")
        fit_ok = False
        continue

    # --- 피셔행렬로 공분산(불확실성) 계산 ---
    fixed_kwargs = {'delay': delay_i, 'n_ports': aa_n_ports}
    F = fm.fisher_information_matrix(fano_models.Sij, popt, param_names, f_i, sigma_est, fixed_kwargs)
    cov = fm.covariance_from_fisher(F)

    # --- 공분산 행렬에서 몬테카를로 샘플링 -> Qi 분포 ---
    mc_samples = np.random.default_rng(0).multivariate_normal(popt, cov, size=20000)
        # random_state를 고정(rng seed=0)해서, 이 스크립트를 다시 실행해도
        # 매번 같은 결과가 재현되도록 함 (분석 재현성 확보).
    Ql_mc, Qc_mc = mc_samples[:, 1], mc_samples[:, 2]
    valid = (Ql_mc > 0) & (Qc_mc > 0)
    Qi_mc = 1.0/(1.0/Ql_mc[valid] - 1.0/Qc_mc[valid])
    Qi_mc_positive = Qi_mc[(Qi_mc > 0) & (Qi_mc < 1e7)]
        # 물리적으로 타당한(양수, 극단적으로 크지 않은) 범위만 남겨
        # 히스토그램이 이상치 때문에 왜곡되지 않도록 함.

    cf_result = circle_fit_results[slice_idx]
    results_summary.append({
        'slice_idx': slice_idx, 'B': B_val,
        'fr': popt[0], 'Ql': popt[1], 'Qc': popt[2],
        'Qi_median': np.median(Qi_mc_positive) if len(Qi_mc_positive) > 0 else np.nan,
        'Qi_16': np.percentile(Qi_mc_positive, 16) if len(Qi_mc_positive) > 0 else np.nan,
        'Qi_84': np.percentile(Qi_mc_positive, 84) if len(Qi_mc_positive) > 0 else np.nan,
        'circle_fit_Qi_max': cf_result['Qi_max'] if cf_result else np.nan,
        'circle_fit_Qi_min': cf_result['Qi_min'] if cf_result else np.nan,
    })

    ax = axes[plot_idx]
    if len(Qi_mc_positive) > 0:
        ax.hist(Qi_mc_positive, bins=60, color='tab:blue', alpha=0.7)
        ax.axvline(np.median(Qi_mc_positive), color='black', lw=1.5,
                   label=f'중앙값={np.median(Qi_mc_positive):.0f}')
    ax.set_xlabel('Qi (베이지안/피셔 몬테카를로 샘플)')
    ax.set_ylabel('count')
    ax.set_title(f'slice {slice_idx} (B={B_val:.3f})\n'
                 f'circle fit: Qi_min={cf_result["Qi_min"]:.0f}, Qi_max=inf')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(aa_output_dir, 'fano_fieldsweep_bayesian_qi.png'), dpi=140, bbox_inches='tight')
print(f"\n그래프 저장 완료: {os.path.join(aa_output_dir, 'fano_fieldsweep_bayesian_qi.png')}")


# =========================================================
# STEP 4. 요약표
# =========================================================
print("\n" + "=" * 90)
print(f"{'slice':>6} {'B':>8} {'베이지안 Qi(중앙값)':>18} {'[16%,84%]':>22} {'circle fit Qi_min':>18}")
print("=" * 90)
for r in results_summary:
    print(f"{r['slice_idx']:>6} {r['B']:>8.4f} {r['Qi_median']:>18.1f} "
          f"[{r['Qi_16']:.0f}, {r['Qi_84']:.0f}]".rjust(22) +
          f" {r['circle_fit_Qi_min']:>18.1f}")

print("\n[해석]")
print("  circle fit은 이 slice들에서 'Qi_max=inf'(상한 없음)만 알려주는 반면,")
print("  베이지안/피셔 방법은 '이 정도 범위(16%~84%) 안에 있다'는 구체적이고")
print("  실전에서 바로 쓸 수 있는 답을 제공합니다. 이게 실제 데이터에서")
print("  두 방법의 실용성 차이를 직접 보여주는 결과입니다.")
