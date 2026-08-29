"""
fano_mcmc_fit.py
=================================================================
지금까지 circle fit(대수적, algebraic 방법)으로 fr, Ql, Qc, phi를
구했습니다. 이번엔 완전히 독립적인 방법(MCMC, 반복적 iterative 방법)
으로 같은 데이터를 피팅해서, 두 결과가 서로 일치하는지 교차검증합니다.
어제 Zenodo 데이터에서 "MCMC vs Mathematica 참값"을 대조했던 것과
정확히 같은 발상을, 오늘은 "MCMC vs circle fit 결과"로 적용합니다.

추가로, 피셔 행렬로 fr-Ql-Qc-phi 사이에 축퇴(degeneracy)가 있는지도
확인합니다. phi가 거의 0에 가까운 slice(오늘 확인한 최저 전력)에서는
축퇴가 약할 것으로 예상되는데, 이걸 실제로 숫자로 확인하는 것이
이 스크립트의 목적입니다. phi가 큰(강한 Fano) slice에서는 이
축퇴가 더 심해질 수 있다는 가설도 나중에 다른 slice로 검증해볼 수
있습니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.getcwd())
import fano_loader
import fano_models
import fano_likelihood as likelihood
import fano_mcmc_pipeline as mcmc_pipeline
import fano_bayesian_toolkit as bt
import fano_fisher_matrix as fm


# =========================================================
# STEP 0. 분석자 설정값 (aa_ 접두사)
# =========================================================
aa_drive_folder = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module01/Fano/overcoupled/'
aa_filename = 'resonator_1_powersweep_overcoupled.npz'
aa_output_dir = './outputs/'
os.makedirs(aa_output_dir, exist_ok=True)

aa_slice_index_to_fit = 0
    # circle fit이 성공했던 그 slice(최저 전력, -80dBm)와 동일한 조건에서
    # 비교해야 공정한 교차검증이 됨.
aa_n_ports = 1.0   # circle fit과 동일 (reflection_port)

# --- [수정] prior 범위를 circle fit 참값 근방으로 과감하게 좁히는
# 실험 ---
# [중요한 방법론적 주의사항] 이건 더 이상 "완전히 독립적인 검증"이
# 아닙니다. circle fit 결과를 이미 알고 있는 상태에서 그 근처로만
# prior를 좁히는 것이므로, "MCMC가 스스로 답을 찾아냈다"기보다는
# "이미 아는 답 근처에서 posterior가 잘 수렴하는지"만 확인하는
# 셈입니다. 실전에서는 이렇게 참값을 미리 알고 prior를 좁히는 게
# 불가능하지만(그게 바로 우리가 알아내려는 것이므로), 지금은 "다중
# 국소최적점 문제가 prior 범위 때문이었는지"를 확인하려는 목적의
# 통제된 실험(controlled experiment)으로 의도적으로 진행합니다.
# 결과가 잘 나온다고 해도 "이 방법으로 실전 미지의 데이터를 풀 수
# 있다"는 뜻은 아니라는 걸 항상 염두에 둬야 함.
aa_prior_narrowing_factor = 0.3
    # circle fit 참값의 +-30% 범위로 prior를 좁힘. 이 배수를 조절하며
    # "얼마나 좁혀야 다중봉우리가 사라지는지" 실험해볼 수 있음.

aa_fixed_delay = None   # STEP 1에서 fano_loader가 알려주는 cable_delay로 채움

aa_nwalkers = 32
aa_nsteps = 8000
aa_burn_in_discard = 500
aa_thin_by = 15
aa_init_scatter = 1e-2
aa_init_scatter_floor = 0.05
    # [주의] 어제 Zenodo에서 겪은 교훈: floor를 파라미터 스케일에 안
    # 맞게 너무 크게 잡으면(예: fr처럼 큰 스케일 파라미터에 부적절한
    # floor를 적용하면) 워커가 prior 밖으로 밀려 clip되는 문제가
    # 생겼음. 이번엔 0.05로 작게 잡아 각 파라미터의 상대적 흩뿌림
    # (init_scatter*abs(값))이 대부분 grid를 지배하도록 함.

print("=" * 60)
print("분석자 설정값(aa_*) 요약:")
for k, v in list(globals().items()):
    if k.startswith("aa_") and not isinstance(v, dict):
        print(f"  {k:24s} = {v}")
print("=" * 60)


# =========================================================
# STEP 1. 데이터 로드 및 circle fit(참고용) 재실행
# =========================================================
full_path = os.path.join(aa_drive_folder, aa_filename)
data = fano_loader.load_fano_npz(full_path)
f_data_hz = data['freq_hz']
s21_slice = data['s21'][aa_slice_index_to_fit, :]
power_val = data['power_dbm'][aa_slice_index_to_fit]
aa_fixed_delay = data['cable_delay']

print(f"\n피팅 대상: power={power_val:.1f} dBm, cable_delay={aa_fixed_delay:.4e}초")

circle_fit_result = fano_models.autofit(
    f_data_hz, s21_slice, n_ports=aa_n_ports, fixed_delay=aa_fixed_delay, isolation=15
)
print("\n[참고] circle fit(대수적) 결과 - MCMC와 비교할 기준선:")
print(f"  fr={circle_fit_result['fr']/1e9:.6f} GHz, Ql={circle_fit_result['Ql']:.1f}, "
      f"Qc={circle_fit_result['Qc']:.1f}, phi={circle_fit_result['phi']:.4f} rad")

# --- circle fit 결과 근방으로 prior를 과감하게 좁힘 (통제된 실험) ---
def narrow_bounds(center, factor, hard_min=None, hard_max=None):
    """
    center 값의 +-factor*100% 범위로 prior 구간을 만드는 헬퍼 함수.
    hard_min/hard_max: 물리적으로 절대 넘을 수 없는 한계(예: Ql>0)가
    있으면 그 안으로 강제로 잘라냄 (예: center=100, factor=2.0이면
    이론상 하한이 -100이 되어버리는데, Ql처럼 반드시 양수여야 하는
    양은 이런 비물리적 하한을 hard_min=0 같은 값으로 막아줘야 함).
    """
    lo = center * (1 - factor)
    hi = center * (1 + factor)
    if hard_min is not None:
        lo = max(lo, hard_min)
    if hard_max is not None:
        hi = min(hi, hard_max)
    return (lo, hi)

aa_prior_bounds = {
    'fr':    narrow_bounds(circle_fit_result['fr'], aa_prior_narrowing_factor*0.001,
                            hard_min=f_data_hz.min(), hard_max=f_data_hz.max()),
        # fr은 스케일이 매우 커서(4.9965e9) 같은 30% 배수를 그대로 쓰면
        # prior가 스캔 범위를 훌쩍 벗어나므로, factor를 0.001배 더
        # 줄여 상대적으로 훨씬 좁은 창(약 ±0.03%)만 허용. 실제로는
        # fr이 이미 가장 정확하게 잡히는 파라미터였으므로 이 정도로도 충분.
    'Ql':    narrow_bounds(circle_fit_result['Ql'], aa_prior_narrowing_factor, hard_min=1.0),
    'Qc':    narrow_bounds(circle_fit_result['Qc'], aa_prior_narrowing_factor, hard_min=1.0),
    'phi':   narrow_bounds(circle_fit_result['phi'], aa_prior_narrowing_factor,
                            hard_min=-np.pi, hard_max=np.pi),
        # phi가 0에 가까우면 narrow_bounds가 거의 (0,0)에 가까운 폭이
        # 될 수 있으므로, 아래에서 최소 폭을 보장하는 안전장치 추가.
    'a':     narrow_bounds(circle_fit_result['a'], aa_prior_narrowing_factor, hard_min=1e-6),
    'alpha': narrow_bounds(circle_fit_result['alpha'], aa_prior_narrowing_factor,
                            hard_min=-np.pi, hard_max=np.pi),
}

# phi처럼 참값이 0에 아주 가까운 파라미터는 배율(factor)만으로 좁히면
# prior 폭 자체가 0에 가까워져 버리는 문제가 생김 (0 * 1.3 = 0.9*0=0
# 근처). 최소 절대 폭을 보장해줌.
phi_lo, phi_hi = aa_prior_bounds['phi']
if phi_hi - phi_lo < 0.2:
    center = circle_fit_result['phi']
    aa_prior_bounds['phi'] = (max(center - 0.2, -np.pi), min(center + 0.2, np.pi))

print(f"\n[좁혀진 prior 범위 - factor={aa_prior_narrowing_factor}]")
for name, (lo, hi) in aa_prior_bounds.items():
    print(f"  {name:8s}: ({lo:.6g}, {hi:.6g})")


# =========================================================
# STEP 2. 초기값을 데이터에서 직접 추정 (circle fit 참값을 그대로
# 베끼지 않고, 독립적으로 대략적인 추정치만 사용 - 완전한 교차검증을
# 위해)
# =========================================================
mag = np.abs(s21_slice)
fr_guess = f_data_hz[np.argmin(mag)]
    # circle fit과 달리, 여기서는 그냥 "|S21|이 최소인 지점"을 fr
    # 초기값으로 씀 - 반사(reflection) 측정에서 공진점은 진폭이 가장
    # 작아지는(에너지가 공진기에 흡수되는) 딥이므로 타당한 추정.

half_level = (mag.max() + mag.min()) / 2
above_half = np.where(mag > half_level)[0]
fwhm_guess = (f_data_hz[above_half[-1]] - f_data_hz[above_half[0]]
              if len(above_half) > 1 else 1e6)
Ql_guess = fr_guess / max(fwhm_guess, 1e3)
    # Ql = fr / kappa 관계(오늘 용어정리에서 다룬 것)를 이용해 FWHM
    # (반치폭)으로부터 Ql 초기값을 역산.

initial_guess = np.array([
    fr_guess,
    Ql_guess,
    Ql_guess * 1.5,   # Qc 초기값: Ql보다 살짝 큰 값에서 시작 (임의의 합리적 출발점)
    0.0,               # phi 초기값 - 이 slice는 Fano가 약할 것으로 예상되므로 0 근처가 합리적
    mag.max() - mag.min(),   # a(진폭 스케일) 초기값
    0.0,               # alpha 초기값
])

print(f"\n[데이터 기반 초기값 - circle fit 참값을 베끼지 않고 독립적으로 추정]")
print(f"  fr_guess={fr_guess/1e9:.6f} GHz, Ql_guess={Ql_guess:.1f}")


# =========================================================
# STEP 3. 우도 함수 정의 및 MCMC 실행
# =========================================================
param_order = ['fr', 'Ql', 'Qc', 'phi', 'a', 'alpha']
log_prior = likelihood.make_uniform_log_prior(
    {name: aa_prior_bounds[name] for name in param_order}
)

def log_probability(theta, f_grid, data_1d, sigma):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    fr, Ql, Qc, phi, a, alpha = theta
    model = fano_models.Sij(f_grid, fr, Ql, Qc, phi, a, alpha,
                              aa_fixed_delay, n_ports=aa_n_ports)
    real_res = np.real(data_1d - model)
    imag_res = np.imag(data_1d - model)
    chi2 = np.sum((real_res/sigma)**2 + (imag_res/sigma)**2)
    return lp + (-0.5 * chi2)

# 잡음 크기: 딥에서 먼(진폭이 baseline에 가까운) 구간의 흔들림으로 추정
baseline_mask = mag > np.percentile(mag, 80)
    # np.percentile(배열, 80): 데이터를 크기순으로 줄 세웠을 때 하위
    # 80%가 되는 지점의 값. "상위 20%(=진폭이 가장 큰, 즉 딥에서 가장
    # 먼 baseline 구간)만 골라내는" 필터로 사용.
noise_sigma_est = np.std(np.real(s21_slice[baseline_mask]))
print(f"추정된 noise_sigma: {noise_sigma_est:.6f}")

print("\nMCMC 실행 중...")
result = mcmc_pipeline.run_single_mcmc(
    log_probability, initial_guess, f_data_hz, s21_slice, noise_sigma_est,
    nwalkers=aa_nwalkers, nsteps=aa_nsteps,
    init_scatter=aa_init_scatter, init_scatter_floor=aa_init_scatter_floor,
    burn_in_discard=aa_burn_in_discard, thin_by=aa_thin_by,
    bounds={name: aa_prior_bounds[name] for name in param_order},
    suppress_warnings=True,
)

print(f"\n수렴 여부: {result['converged']}")
if result['autocorr_time'] is not None:
    print(f"자기상관 시간: {np.round(result['autocorr_time'], 1)}")


# =========================================================
# STEP 4. MCMC vs circle fit 대조표
# =========================================================
reference = [circle_fit_result['fr'], circle_fit_result['Ql'],
             circle_fit_result['Qc'], circle_fit_result['phi'],
             circle_fit_result['a'], circle_fit_result['alpha']]

print("\n" + "=" * 65)
print(f"{'파라미터':<8} {'MCMC 추정':>16} {'circle fit':>16} {'상대오차(%)':>13}")
print("=" * 65)
for i, name in enumerate(param_order):
    est = result['median'][i]
    ref = reference[i]
    rel_err = abs(est-ref)/abs(ref)*100 if ref != 0 else float('nan')
    flag = "  ⚠️" if rel_err > 10 else ""
    print(f"{name:<8} {est:>16.6f} {ref:>16.6f} {rel_err:>12.2f}%{flag}")


# =========================================================
# STEP 5. 피셔 행렬 진단 - fr,Ql,Qc,phi 사이 축퇴 확인
# =========================================================
print("\n" + "=" * 60)
print("STEP 5. 피셔 행렬 진단 (독립적인 방법으로 축퇴 재확인)")
print("=" * 60)

sigma_fisher, corr_fisher = fm.fisher_parameter_uncertainties(
    fano_models.Sij, result['median'], param_order, f_data_hz, noise_sigma_est,
    {'delay': aa_fixed_delay, 'n_ports': aa_n_ports}
)

mcmc_sigma = (result['err_lo'] + result['err_hi']) / 2
print(f"\n{'파라미터':<8} {'MCMC 1σ':>14} {'피셔 1σ':>14} {'비율':>10}")
print("-" * 55)
for i, name in enumerate(param_order):
    if np.isnan(sigma_fisher[i]):
        print(f"{name:<8} {mcmc_sigma[i]:>14.6f} {'NaN(특이)':>14} {'축퇴 신호':>10}")
    else:
        ratio = mcmc_sigma[i] / sigma_fisher[i]
        print(f"{name:<8} {mcmc_sigma[i]:>14.6f} {sigma_fisher[i]:>14.6f} {ratio:>10.3f}")

print(f"\nfr-Ql 상관계수: {corr_fisher[0,1]:.4f}")
print(f"Ql-Qc 상관계수: {corr_fisher[1,2]:.4f}")
print(f"Qc-phi 상관계수: {corr_fisher[2,3]:.4f}")


# =========================================================
# STEP 6. Corner plot
# =========================================================
try:
    import corner
    fig_corner = corner.corner(
        result['flat_samples'],
        labels=[r'$f_r$', r'$Q_l$', r'$Q_c$', r'$\phi$', r'$a$', r'$\alpha$'],
        truths=reference,
        truth_color='red',
        show_titles=True,
        title_kwargs={'fontsize': 9},
    )
    fig_corner.suptitle(f'Fano MCMC Posterior (power={power_val:.1f}dBm)', y=1.01, fontsize=13)
    plt.savefig(os.path.join(aa_output_dir, 'fano_mcmc_corner.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nCorner plot 저장 완료: {os.path.join(aa_output_dir, 'fano_mcmc_corner.png')}")
except ImportError:
    print("\n[안내] corner 패키지 없음, corner plot 생략.")
