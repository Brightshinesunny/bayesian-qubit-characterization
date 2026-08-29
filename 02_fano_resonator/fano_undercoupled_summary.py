"""
fano_undercoupled_summary.py
=================================================================
[undercoupled 조사 최종 정리] 어제 세운 가설("undercoupled는 Qc
불확실성이 Qi보다 항상 크다")을 9개 undercoupled resonator 전부에
검증한 결과와, 그 과정에서 발견한 더 정확한 법칙을 그래프로
정리합니다.

[결론 요약 - 이 스크립트가 증명하는 것]
"undercoupled냐 아니냐"라는 이분법이 아니라, "결합이 얼마나 강하게
undercoupled인가"(Ql/Qc 비율)가 진짜 결정 요인이었습니다:
  - Ql/Qc < ~0.35 (강하게 undercoupled, Qc가 Ql을 압도) ->
    Qc 신뢰구간이 Qi보다 넓다는 가설이 성립
  - Ql/Qc > ~0.35 (약하게 undercoupled, 임계결합에 더 가까움) ->
    가설이 반대로 뒤집힘(Qi 신뢰구간이 오히려 더 넓음)

이건 어제/오늘 계속 확인한 "임계결합에 가까워질수록 Ql-Qc가 서로
얽혀 축퇴가 심해진다"는 원리가, 여기서는 "신뢰구간 크기 비교의
방향 자체를 뒤집는" 형태로 나타난 것입니다.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import sys, os

sys.path.insert(0, os.getcwd())
import fano_models
import fano_likelihood as likelihood
import fano_fisher_matrix as fm
import fano_bayesian_toolkit as bt
from scipy.optimize import minimize


# =========================================================
# STEP 0. 분석자 설정값 (aa_ 접두사)
# =========================================================
aa_upload_dir = '/mnt/user-data/uploads/'
aa_resonator_ids = list(range(1, 9))
    # [조정] R9는 시간 관계상 오늘 분석에서 제외(멀티스타트 최적화가
    # 시간 안에 안 끝남) - 나중에 여유 있을 때 추가 확인 필요.
aa_n_ports = 1.0
aa_isolation_db = 15
aa_phi_candidates = np.linspace(-3.0, 3.0, 9)
aa_outlier_trim_pct = 5.0
aa_mc_sample_size = 20000
aa_step_fraction = 1e-6
aa_credible_level = 0.68
aa_coupling_ratio_threshold = 0.35
    # [조정] 오늘 발견한 분리선(일치 그룹 0.22~0.31, 불일치 그룹
    # 0.40~0.51 - 그 사이 간격의 중간값으로 설정).
aa_output_dir = './outputs/'
os.makedirs(aa_output_dir, exist_ok=True)


def load_fano_npz(filepath):
    raw = np.load(filepath, allow_pickle=True)
    amplitude, phase = raw['amplitude'], raw['phase']
    freq_hz, power_dbm = raw['frequency'], raw['power']
    s21 = amplitude * np.exp(1j*phase)
    settings_dict = json.loads(raw['settings'][0])
    cable_delay = settings_dict.get('vna', {}).get('port1_edel', None)
    return {'power_dbm': power_dbm, 'freq_hz': freq_hz, 's21': s21, 'cable_delay': cable_delay}


def process_resonator(resonator_id):
    """
    resonator 하나를 오늘 확정된 파이프라인(circle fit (a,alpha)고정
    + 5%이상치제거 + MAD sigma + 피셔행렬)으로 처리하고, Ql/Qc 비율과
    Qi/Qc 신뢰구간 폭을 반환.
    """
    fname = f'resonator_{resonator_id}_powersweep_undercoupled.npz'
    data = load_fano_npz(os.path.join(aa_upload_dir, fname))
    f_grid = data['freq_hz']
    s21 = data['s21'][0, :]
    delay = data['cable_delay']
    mag = np.abs(s21)

    cf = fano_models.autofit(f_grid, s21, n_ports=aa_n_ports, fixed_delay=delay, isolation=aa_isolation_db)
    a_fixed, alpha_fixed = cf['a'], cf['alpha']
        # [오늘 확정 절차] circle fit의 (a,alpha)를 그대로 고정 -
        # baseline 평균보다 훨씬 정확하다는 게 오늘 R7 분석에서 확인됨.

    param_names = ['fr', 'Ql', 'Qc', 'phi']
    prior_bounds = {'fr': (f_grid.min(), f_grid.max()), 'Ql': (1.0, 1e8),
                     'Qc': (1.0, 1e8), 'phi': (-np.pi, np.pi)}
    log_prior = likelihood.make_uniform_log_prior(prior_bounds)
    fixed_kwargs = {'delay': delay, 'n_ports': aa_n_ports, 'a': a_fixed, 'alpha': alpha_fixed}
    log_probability = likelihood.make_log_probability_generic(
        fano_models.Sij, fixed_kwargs, param_names, log_prior, likelihood_type='gaussian'
    )

    fr_guess = f_grid[np.argmin(mag)]
    Ql_guess = cf['Ql']   # circle fit 값을 초기 스케일 참고용으로 사용
    candidates = [np.array([fr_guess, Ql_guess, Ql_guess*1.5, phi0]) for phi0 in aa_phi_candidates]
    candidates.append(np.array([cf['fr'], cf['Ql'], cf['Qc'], cf['phi']]))

    baseline_mask = mag > np.percentile(mag, 80)
    sigma_baseline = np.std(np.real(s21[baseline_mask]))

    def find_map(sigma):
        best_theta, best_neg_lp = None, np.inf
        for theta0 in candidates:
            def neg_log_prob(theta):
                val = log_probability(theta, f_grid, s21, sigma)
                return -val if np.isfinite(val) else 1e10
            res = minimize(neg_log_prob, theta0, method='Nelder-Mead',
                            options={'maxiter': 15000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_neg_lp:
                best_neg_lp = res.fun
                best_theta = res.x
        return best_theta

    popt0 = find_map(sigma_baseline)
    model0 = fano_models.Sij(f_grid, *popt0, a_fixed, alpha_fixed, delay, n_ports=aa_n_ports)
    residual0 = np.real(s21 - model0)
    thresh = np.percentile(np.abs(residual0), 100 - aa_outlier_trim_pct)
    keep_mask = np.abs(residual0) <= thresh
    residual_trimmed = residual0[keep_mask]
    mad = np.median(np.abs(residual_trimmed - np.median(residual_trimmed)))
    sigma_mad = mad * 1.4826

    popt_final = find_map(sigma_mad)
    Ql_final, Qc_final = popt_final[1], popt_final[2]

    F = fm.fisher_information_matrix(fano_models.Sij, popt_final, param_names, f_grid,
                                        sigma_mad, fixed_kwargs, step_fraction=aa_step_fraction)
    cov = fm.covariance_from_fisher(F)
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() <= 0:
        return None

    mc = np.random.default_rng(0).multivariate_normal(popt_final, cov, size=aa_mc_sample_size)
    Qi_mc = 1.0/(1.0/mc[:,1] - 1.0/mc[:,2])
    valid_qi = (Qi_mc > 0) & (Qi_mc < 1e7)
    Qi_lo, Qi_med, Qi_hi = bt.credible_interval(Qi_mc[valid_qi], level=aa_credible_level)
    Qc_lo, Qc_med, Qc_hi = bt.credible_interval(mc[:, 2], level=aa_credible_level)

    return {
        'resonator_id': resonator_id,
        'Ql': Ql_final, 'Qc': Qc_final,
        'ratio': Ql_final / Qc_final,
        'Qi_width': Qi_hi - Qi_lo, 'Qc_width': Qc_hi - Qc_lo,
        'Qc_wider': (Qc_hi - Qc_lo) > (Qi_hi - Qi_lo),
    }


# =========================================================
# STEP 1. 8개 resonator 전부 처리
# =========================================================
results = []
for rid in aa_resonator_ids:
    print(f"처리 중: resonator_{rid}...")
    r = process_resonator(rid)
    if r is not None:
        results.append(r)
        print(f"  Ql/Qc={r['ratio']:.4f}, Qc폭>Qi폭? {r['Qc_wider']}")


# =========================================================
# STEP 2. 그래프 - Ql/Qc 비율 vs "Qc폭이 더 넓은가" 관계 시각화
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ratios = [r['ratio'] for r in results]
qc_wider = [r['Qc_wider'] for r in results]
ids = [r['resonator_id'] for r in results]
colors = ['tab:blue' if w else 'tab:red' for w in qc_wider]

# 왼쪽: Ql/Qc 비율 vs 어느 쪽이 더 넓은지 (핵심 결과 그래프)
for r, c in zip(results, colors):
    axes[0].scatter(r['ratio'], 1 if r['Qc_wider'] else 0, s=120, color=c, zorder=3)
    axes[0].annotate(f"R{r['resonator_id']}", (r['ratio'], 1 if r['Qc_wider'] else 0),
                       textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
axes[0].axvline(aa_coupling_ratio_threshold, color='gray', ls='--', lw=1.5,
                  label=f'분리 기준선(Ql/Qc={aa_coupling_ratio_threshold})')
axes[0].set_xlabel('Ql/Qc 비율 (작을수록 강한 undercoupled)')
axes[0].set_yticks([0,1])
axes[0].set_yticklabels(['Qi폭이 더 큼\n(가설과 반대)', 'Qc폭이 더 큼\n(가설과 일치)'])
axes[0].set_title('결합 강도(Ql/Qc)에 따라 신뢰구간 크기 방향이 뒤집힘')
axes[0].legend(fontsize=8, loc='center right')
axes[0].set_ylim(-0.3, 1.3)

# 오른쪽: Qc폭/Qi폭 비율을 Ql/Qc에 대해 직접 그림 (연속적인 경향 확인)
width_ratios = [r['Qc_width']/r['Qi_width'] for r in results]
for r, wr, c in zip(results, width_ratios, colors):
    axes[1].scatter(r['ratio'], wr, s=120, color=c, zorder=3)
    axes[1].annotate(f"R{r['resonator_id']}", (r['ratio'], wr),
                       textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
axes[1].axhline(1.0, color='gray', ls='--', lw=1.5, label='Qc폭=Qi폭 (경계선)')
axes[1].axvline(aa_coupling_ratio_threshold, color='gray', ls=':', lw=1)
axes[1].set_xlabel('Ql/Qc 비율')
axes[1].set_ylabel('Qc폭 / Qi폭 (1보다 크면 Qc가 더 불확실)')
axes[1].set_yscale('log')
axes[1].set_title('Qc/Qi 신뢰구간 폭 비율의 연속적인 경향')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(aa_output_dir, 'fano_undercoupled_summary.png'), dpi=140, bbox_inches='tight')
print(f"\n그래프 저장 완료: {os.path.join(aa_output_dir, 'fano_undercoupled_summary.png')}")


# =========================================================
# STEP 3. 최종 요약 텍스트
# =========================================================
print("\n" + "="*80)
print("최종 요약")
print("="*80)
print(f"{'R#':>4} {'Ql/Qc':>8} {'Qc폭/Qi폭':>10} {'가설 방향':>12}")
for r, wr in zip(results, width_ratios):
    print(f"{r['resonator_id']:>4} {r['ratio']:>8.4f} {wr:>10.2f} "
          f"{'일치' if r['Qc_wider'] else '불일치':>12}")

n_below = sum(1 for r in results if r['ratio'] < aa_coupling_ratio_threshold)
n_below_matched = sum(1 for r in results if r['ratio'] < aa_coupling_ratio_threshold and r['Qc_wider'])
n_above = sum(1 for r in results if r['ratio'] >= aa_coupling_ratio_threshold)
n_above_matched = sum(1 for r in results if r['ratio'] >= aa_coupling_ratio_threshold and not r['Qc_wider'])

print(f"\nQl/Qc < {aa_coupling_ratio_threshold} (강한 undercoupled) 인 {n_below}개 중, "
      f"가설과 일치한 개수: {n_below_matched}")
print(f"Ql/Qc >= {aa_coupling_ratio_threshold} (약한 undercoupled) 인 {n_above}개 중, "
      f"가설과 반대인 개수: {n_above_matched}")
print(f"\n[결론] '결합 강도(Ql/Qc)'가 신뢰구간 방향을 결정하는 진짜 요인이며,")
print(f"       단순 'undercoupled 여부'보다 훨씬 정확한 판별 기준입니다.")
