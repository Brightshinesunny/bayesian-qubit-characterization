"""
tls_avcross_fit_seg12.py
=================================================================
[오늘의 최종] seg12(raw_index=12, alpha 전극) 2D 데이터(21x401,
dispamp)에, avoided-crossing 2D 모델을 직접 피팅합니다.

[오늘 하루 종일 한 것과의 차이 - 핵심]
지금까지는 픽셀 하나씩 argmin으로 딥 위치를 "추적"해서 직선을
피팅했습니다(계통오차, 교차점 문제로 5번 실패했던 그 방법). 이번엔
2D 맵 전체를 avoided-crossing 물리모델(hybrid mode 공식)로 한 번에
설명합니다 - 픽셀 추적이 아예 필요 없어서, 오늘 겪은 문제들에서
자유롭습니다.

[STEP 1: curve_fit(빠른 점추정)] -> [STEP 2: MCMC 준비(emcee 필요,
Colab에서 실행)] 순서로 둘 다 진행합니다.
"""

import h5py
import numpy as np
from scipy.optimize import curve_fit
import sys, os
sys.path.insert(0, os.getcwd())
import tls_avcross_models as models
import tls_avcross_likelihood as likelihood
import tls_avcross_mcmc_pipeline as mcmc_pipeline   # emcee 필요 - Colab에서만 동작

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
aa_target_raw_index = 12
aa_observable = 'dispamp'
aa_prior_bounds = {
    'f_TLS0': (5.10, 5.20), 'gamma_stark': (-0.2, 0.2),
    'g': (1e-6, 0.03), 'baseline': (-10.0, 10.0), 'contrast': (-10.0, 10.0),
}
    # [버그 수정 - 중요] gamma_stark 범위가 기존 (-0.01,0.01)이라, 오늘
    # 픽셀 추적으로 확인한 0.109가 애초에 prior 밖이었음! MCMC는
    # 원리적으로 그쪽을 절대 탐색 못 함(log_prior=-inf) - "우물이
    # 하나뿐이었다"는 결론 자체가 무효였던 것. (-0.2,0.2)로 넓혀서
    # 0.109를 포함시킴.
    # g 상한도 0.5(비물리적으로 큼, 500MHz)에서 0.03(30MHz, 문헌상
    # 일반적인 큐빗-TLS 결합강도 상한 근처)으로 물리적으로 조임 -
    # 이전에 g가 상한(0.5)까지 안 붙던 걸 보면, 이 정도로 조여도
    # 실제 최적점을 못 자를 가능성이 높음.
aa_freq_window_narrow = (5.13, 5.19)
    # [신규] 육안으로 대각선이 보였던 범위로 탐색을 좁힘 - 넓은
    # 범위(5.03~5.26)에는 대각선과 무관한 행이 훨씬 많아서, chi2에서
    # 대각선의 기여가 희석됐을 가능성.
aa_param_names = ['f_TLS0', 'gamma_stark', 'g', 'baseline', 'contrast']
aa_V0 = None
aa_known_artifact_centers = [0.045, -0.030, -0.065]
aa_artifact_halfwidth = 0.006
aa_freq_artifact_center = 5.168   # GHz - 오늘 픽셀 추적에서 이 세그먼트(raw_index=12)
aa_freq_artifact_halfwidth = 0.002   # GHz
    # [신규] 위 3개(플럭스 기준) 마스킹만으로는 불충분했음 - 오늘
    # 픽셀 추적 때 바로 이 segment에서 발견한 "네 번째 계통오차"
    # (5.168GHz 근처)가 여전히 안 지워진 채 남아있어서, gamma_stark가
    # 마스킹 전후로 거의 안 바뀌었던 것으로 확인됨. 이건 sweep2(플럭스)
    # 기준이 아니라 주파수 자체에 고정된 것으로 보이므로, 변환 후
    # 주파수 공간에서 별도로 제외.
aa_nwalkers = 32
aa_nsteps = 4000


def mat_string(f, dataset_or_ref):
    try:
        if isinstance(dataset_or_ref, h5py.Reference):
            dataset_or_ref = f[dataset_or_ref]
        arr = np.array(dataset_or_ref)
        return ''.join(chr(int(c)) for c in arr.flatten())
    except Exception:
        return None


with h5py.File(aa_mat_filepath, 'r') as f:
    qdat_group = f[f['qdats']['qdat'][aa_target_raw_index, 0]]
    sweep1vals = np.array(qdat_group['sweep1vals']).flatten()   # 게이트 전압(21개)
    sweep2vals = np.array(qdat_group['sweep2vals']).flatten()   # 플럭스(401개)

    calib_field = np.array(f['qset']['calibdat']['field']).flatten()
    calib_freq_hz = np.array(f['qset']['calibdat']['qubitfreq']).flatten()
    order = np.argsort(calib_field)
    fq_axis = np.interp(sweep2vals, calib_field[order], calib_freq_hz[order]) / 1e9
        # [버그 수정] calibdat.qubitfreq는 Hz 단위(~5.03e9)로 저장돼
        # 있음. 우리 모델(tls_swap_spectroscopy_2d_model)과 prior
        # 범위(aa_prior_bounds)는 전부 GHz 단위(5.10~5.20 같은 한 자리
        # 수)로 설계/검증했으므로, 여기서 1e9로 나눠 GHz로 통일해야
        # 함 - 안 그러면 "GHz라고 표시되지만 실제로는 Hz"인 채로
        # 모델에 들어가 완전히 어긋난 결과가 나옴.

    obs_val_refs = qdat_group['obs']['vals']
    obs_name_refs = qdat_group['observables']
    target_col = None
    for i in range(obs_name_refs.shape[0]):
        if mat_string(f, obs_name_refs[i, 0]) == aa_observable:
            target_col = i
            break
    data_2d = np.array(f[obs_val_refs[target_col, 0]])
        # [주의] 임의 단위 변환 없이 원본 그대로 사용. baseline/contrast의
        # 초기값(p0)은 아래에서 실제 데이터 통계(중앙값, 최대-최소)로
        # 자동 결정되므로, 스케일을 미리 짐작해서 나눌 필요가 없음.

V_grid, fq_grid = np.meshgrid(sweep1vals, fq_axis)   # shape (401,21) - data_2d와 동일

# =====================================================
# [신규] 계통오차 마스킹 - (1) 플럭스 기준 3개 + (2) 주파수 기준 1개
# =====================================================
row_ok = np.ones(len(sweep2vals), dtype=bool)
for center in aa_known_artifact_centers:
    row_ok &= (np.abs(sweep2vals - center) > aa_artifact_halfwidth)
n_masked_flux = np.sum(~row_ok)

row_ok &= (np.abs(fq_axis - aa_freq_artifact_center) > aa_freq_artifact_halfwidth)
n_masked_after_artifacts = np.sum(~row_ok)

# [신규] 육안으로 대각선이 확인된 좁은 주파수 창으로 추가 제한
row_ok &= (fq_axis >= aa_freq_window_narrow[0]) & (fq_axis <= aa_freq_window_narrow[1])
n_masked_total = np.sum(~row_ok)
print(f"계통오차 마스킹: 플럭스기준 {n_masked_flux}개 + 주파수기준(계통오차) "
      f"{n_masked_after_artifacts-n_masked_flux}개 + 창축소(추가) "
      f"{n_masked_total-n_masked_after_artifacts}개 = 총 {n_masked_total}개 행 제외 (전체 {len(sweep2vals)}개 중)")

V_grid = V_grid[row_ok, :]
fq_grid = fq_grid[row_ok, :]
data_2d = data_2d[row_ok, :]

if aa_V0 is None:
    aa_V0 = float(np.median(sweep1vals))

print(f"데이터 shape(마스킹 후): {data_2d.shape}, 전압범위=[{sweep1vals.min():.2f},{sweep1vals.max():.2f}]V, "
      f"주파수범위=[{fq_axis.min():.4f},{fq_axis.max():.4f}]GHz, V0={aa_V0:.2f}")

# =========================================================
# STEP 1. curve_fit으로 빠른 점추정
# =========================================================
def model_wrapper(xy_flat, f_TLS0, gamma_stark, g, baseline, contrast):
    n = V_grid.size
    V_flat, fq_flat = xy_flat[:n], xy_flat[n:]
    V_2d = V_flat.reshape(V_grid.shape)
    fq_2d = fq_flat.reshape(fq_grid.shape)
    return models.tls_swap_spectroscopy_2d_model(
        V_2d, fq_2d, f_TLS0, gamma_stark, g, baseline, contrast, V0=aa_V0
    ).flatten()

xy_flat = np.concatenate([V_grid.flatten(), fq_grid.flatten()])

# [수정] p0 하나만 시도하지 말고, 여러 후보로 multi-start - 오늘 픽셀
# 추적 때 겪은 것과 같은 이유(국소최적점 문제)로, gamma_stark 초기값에
# 따라 결과가 크게 요동치는 게 확인됨. 특히 오늘 이미 픽셀 추적으로
# 확보한 gamma_stark≈0.109(=109MHz/V) 근처 후보를 반드시 포함시켜서,
# 그쪽으로 수렴하는 해가 진짜 존재하는지, 있다면 chi2가 다른 후보보다
# 낮은지(=더 나은 설명인지) 직접 비교.
gamma_candidates = [-0.109, -0.05, 0.0, 0.05, 0.109]
    # 부호를 모르니 +/-109MHz/V 둘 다 포함.
f_TLS0_candidates = [5.10, 5.15, 5.20]

lower_bounds = [aa_freq_window_narrow[0], -0.2, 1e-6, data_2d.min()-1, -5.0]
upper_bounds = [aa_freq_window_narrow[1], 0.2, 0.03, data_2d.max()+1, 5.0]
    # [수정] g 상한을 물리적으로 조임(0.5->0.03GHz=30MHz), f_TLS0/gamma
    # 범위도 새 prior와 맞춤(창 축소, gamma 넓힘).

best_popt, best_chi2 = None, np.inf
print("\n[multi-start 탐색]")
for f0 in f_TLS0_candidates:
    for gc in gamma_candidates:
        p0_trial = [f0, gc, 0.001, np.median(data_2d), (data_2d.max()-data_2d.min())/2]
        try:
            popt_trial, _ = curve_fit(model_wrapper, xy_flat, data_2d.flatten(), p0=p0_trial,
                                        bounds=(lower_bounds, upper_bounds), maxfev=30000)
            model_pred = model_wrapper(xy_flat, *popt_trial).reshape(data_2d.shape)
            chi2_trial = np.sum((data_2d - model_pred)**2)
            print(f"  p0(f_TLS0={f0:.2f}, gamma={gc:+.3f}) -> chi2={chi2_trial:.4f}, "
                  f"수렴결과: gamma={popt_trial[1]:.5f}, g={popt_trial[2]:.5f}")
            if chi2_trial < best_chi2:
                best_chi2 = chi2_trial
                best_popt = popt_trial
        except Exception as e:
            print(f"  p0(f_TLS0={f0:.2f}, gamma={gc:+.3f}) -> 실패({e})")

popt = best_popt
print(f"\n[최종 채택(multi-start 중 최선) - chi2={best_chi2:.4f}]")
for name, val in zip(aa_param_names, popt):
    print(f"  {name:12s}: {val:.6g}")
print("\n[경고] chi2들이 서로 비슷한데 파라미터가 크게 다르면, 이건")
print("점추정으로 해결 못 하는 다중봉우리(multi-modal) posterior라는 뜻.")
print("아래 진짜 MCMC로 posterior 전체 모양을 확인해야 함.")

# =========================================================
# STEP 2. 진짜 MCMC 실행 (emcee 필요 - Colab에서)
# =========================================================
log_prior = likelihood.make_uniform_log_prior(aa_prior_bounds)
log_probability_2d = likelihood.make_log_probability_2d_real(
    models.tls_swap_spectroscopy_2d_model, {'V0': aa_V0}, aa_param_names, log_prior, likelihood_type='gaussian'
)

def log_probability(theta, xy_tuple, data_2d_arg, sigma):
    """
    [어댑터] run_single_mcmc는 emcee의 관례대로 f_grid를 "인자 하나"로만
    넘기는데(내부적으로 args=(f_grid, data_1d, sigma)), 우리 2D 모델은
    (x_grid, y_grid) 두 개가 필요함. 그래서 f_grid 자리에 (V_grid,fq_grid)
    튜플을 통째로 넘기고, 여기서 풀어서(unpack) 원래 함수에 전달.
    """
    x_grid, y_grid = xy_tuple
    return log_probability_2d(theta, x_grid, y_grid, data_2d_arg, sigma)

residual = data_2d - model_wrapper(xy_flat, *popt).reshape(data_2d.shape)
sigma_est = np.std(residual)
print(f"\n[STEP 2] 진짜 MCMC 실행 (posterior 전체 탐색 - 우물이 몇 개인지 직접 확인)")

result = mcmc_pipeline.run_single_mcmc(
    log_probability, np.array(popt), (V_grid, fq_grid), data_2d, sigma_est,
    nwalkers=aa_nwalkers, nsteps=aa_nsteps, init_scatter=1e-2,
    burn_in_discard=500, thin_by=15, bounds=aa_prior_bounds, suppress_warnings=True
)
    # [버그 수정] init_scatter_floor는 이 버전의 run_single_mcmc에 없는
    # 인자였음(넣으면 TypeError) - 제거. f_grid 자리에는 (V_grid,fq_grid)
    # 튜플을 그대로 전달 - 위 어댑터 함수가 이걸 풀어서 처리함.
print(f"\n수렴 여부: {result['converged']}")
for i, name in enumerate(aa_param_names):
    print(f"  {name:12s}: {result['median'][i]:.6g} "
          f"(+{result['err_hi'][i]:.6g}/-{result['err_lo'][i]:.6g})")

gamma_samples = result['flat_samples'][:, 1]
print(f"\ngamma_stark posterior 요약:")
print(f"  중앙값: {np.median(gamma_samples):.5f}")
print(f"  [16%,84%]: [{np.percentile(gamma_samples,16):.5f}, {np.percentile(gamma_samples,84):.5f}]")
print(f"  히스토그램에 여러 봉우리가 보이면(예: 0 근처와 0.109 근처 둘 다),")
print(f"  이게 바로 오늘 multi-start에서 본 '여러 답이 비슷한 chi2' 현상의 정체입니다.")

# =========================================================
# STEP 3. [신규] 큰-gamma(0.109 근처) 시작점 전용 MCMC - 그 우물의
# 무게(posterior 확률)를 직접 확인
# =========================================================
# STEP2의 MCMC는 popt(작은 gamma 근처)에서 시작했으므로, 그 근처
# 우물만 확인했을 수 있음. 이번엔 gamma=0.109 근처에서 "타이트하게"
# 시작시켜서, 체인이 그 자리에 머무는지(=그 우물도 유효한 확률을
# 가짐) 아니면 다른 곳(예: STEP2의 우물)으로 흘러가버리는지 확인.
initial_guess_large_gamma = np.array([5.15, 0.109, 0.005, np.median(data_2d),
                                        (data_2d.max()-data_2d.min())/2])
print(f"\n[STEP 3] 큰-gamma(0.109) 시작점 전용 MCMC")
result_large = mcmc_pipeline.run_single_mcmc(
    log_probability, initial_guess_large_gamma, (V_grid, fq_grid), data_2d, sigma_est,
    nwalkers=aa_nwalkers, nsteps=aa_nsteps, init_scatter=1e-3,
        # init_scatter를 STEP2보다 훨씬 작게(1e-3) - 워커들이 0.109
        # 근처에 "타이트하게" 모여 시작하게 해서, 그 우물이 진짜
        # 존재하면 벗어나지 않고 머물러야 정상.
    burn_in_discard=500, thin_by=15, bounds=aa_prior_bounds, suppress_warnings=True
)
print(f"수렴 여부: {result_large['converged']}")
for i, name in enumerate(aa_param_names):
    print(f"  {name:12s}: {result_large['median'][i]:.6g} "
          f"(+{result_large['err_hi'][i]:.6g}/-{result_large['err_lo'][i]:.6g})")

gamma_samples_large = result_large['flat_samples'][:, 1]
print(f"\ngamma_stark(큰-gamma 시작) posterior 요약:")
print(f"  중앙값: {np.median(gamma_samples_large):.5f}")
print(f"  [16%,84%]: [{np.percentile(gamma_samples_large,16):.5f}, {np.percentile(gamma_samples_large,84):.5f}]")

print("\n[최종 판정 가이드]")
print("  이 체인이 0.109 근처에 그대로 머물러 있으면(중앙값이 0.05 이상)")
print("    -> 진짜 우물이 두 개 있고, 어느 쪽이 더 그럴듯한지는 chi2/우도 비교 필요")
print("  이 체인도 STEP2와 비슷한 작은 값으로 흘러가버렸으면")
print("    -> 0.109 근처는 진짜 우물이 아니라 국소적으로만 그럴듯해 보였던 것")
print("       (2D 모델+이 마스킹 조건에서는 픽셀추적 결과를 재현 못 함)")
