"""
zenodo_multi_file_check.py
=================================================================
[오늘의 실험 1번] "다른 kappa 파일에서도 같은 ke1-ke2 축퇴가
나타나는가?"를 확인하는 스크립트입니다.

지금까지는 kappa≈8MHz 파일(S21_power_sweep_152...) 하나만 봤는데,
원저자 데이터셋에는 kappa가 1.1MHz~32MHz까지 서로 다른 6개의 공진기
측정 파일이 있습니다. 물리적으로 kappa(감쇠율)가 다르면 ke1/ke2/ki의
"비율"도 다르게 설계되어 있을 가능성이 높은데, 그래도 여전히
ke1×ke2(곱셈 형태로만 등장) 축퇴 자체는 "이 물리 모델의 수학적
구조" 때문에 생기는 것이므로, kappa 크기와 무관하게 모든 파일에서
똑같이 나타나야 합니다. 이걸 실제로 6개 파일 전부에 대해 반복
확인하는 것이 이 스크립트의 목적입니다.

[구조적으로 바뀐 부분]
어제까지의 zenodo_fit_and_compare.py는 파일 하나를 대상으로 위에서
아래로 순서대로 실행하는 "스크립트" 형태였습니다. 오늘은 같은
로직을 "함수 하나(fit_one_file)"로 감싸서, 파일 목록을 순회하며
반복 호출하는 구조로 바꿨습니다. 이렇게 하면 코드 중복 없이 6개
파일을 한 번에 처리할 수 있습니다 (이전에 mcmc_pipeline.py의
run_batch_mcmc_warmstart를 만들 때와 같은 설계 원칙 - "반복되는
작업은 함수로 감싼다").
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import zenodo_loader
import zenodo_models
import zenodo_likelihood as likelihood
import zenodo_mcmc_pipeline as mcmc_pipeline
import zenodo_diagnostics as diagnostics
import zenodo_fisher_matrix as zfm


# =========================================================
# STEP 0. 분석자 설정값 (aa_ 접두사)
# =========================================================
aa_drive_folder = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module01/Figure1/'
aa_output_dir = './outputs/'
os.makedirs(aa_output_dir, exist_ok=True)

# 6개 파일과 각각의 원저자 Mathematica 참값을 하나의 딕셔너리로 정리.
# 이렇게 표 형태로 모아두면, 아래 반복문에서 "파일 하나씩 순서대로
# 꺼내 쓰기"가 매우 간단해짐 (딕셔너리를 코드 밖으로 분리해두면
# 나중에 파일이 더 늘어나도 이 표만 추가하면 되는 구조).
aa_file_reference_table = {
    'kappa_1.1MHz': {
        'filename': 'S21_pow_swp_200_data_210516_14h15m20s.txt',
        'ref': dict(f0=10.48227, ke1=0.000502175, ke2=0.0000967125,
                    ki=0.000584727, A0=0.00966272, phi=-678.506, tau=60.2624),
    },
    'kappa_2.6MHz': {
        'filename': 'S21_power_sweep_163_data_210827_16h25m15s.txt',
        'ref': dict(f0=10.4788, ke1=0.000529924, ke2=0.00146995,
                    ki=0.000581952, A0=0.0217612, phi=-53.6596, tau=62.7926),
    },
    'kappa_4.7MHz': {
        'filename': 'S21_power_sweep_162_data_210827_15h10m20s.txt',
        'ref': dict(f0=10.475, ke1=0.000532856, ke2=0.00363597,
                    ki=0.000529381, A0=0.0209334, phi=-44.5223, tau=62.6534),
    },
    'kappa_8.0MHz': {
        'filename': 'S21_power_sweep_152_data_210823_21h50m19s.txt',
        'ref': dict(f0=10.4701, ke1=0.000492529, ke2=0.00699359,
                    ki=0.000457205, A0=0.0208633, phi=-43.4931, tau=62.6379),
    },
    'kappa_13MHz': {
        'filename': 'S21_power_sweep_159_data_210826_16h55m01s.txt',
        'ref': dict(f0=10.464, ke1=0.000464665, ke2=0.0118367,
                    ki=0.000522775, A0=0.0197085, phi=-41.6398, tau=62.6088),
    },
    'kappa_32MHz': {
        'filename': 'S21_power_sweep_161_data_210827_13h03m23s.txt',
        'ref': dict(f0=10.4488, ke1=0.000480211, ke2=0.0284462,
                    ki=0.00077413, A0=0.0190133, phi=-38.0968, tau=62.5546),
    },
}

# MCMC 설정 (어제 zenodo_fit_and_compare.py에서 검증한 값을 그대로 재사용)
aa_power_slice_index = -1   # 각 파일에서 항상 "가장 높은 전력" 슬라이스를 씀
aa_prior_bounds = {
    'f0':  (10.40, 10.55),
    'ke1': (1e-6, 0.05),
    'ke2': (1e-6, 0.05),
    'ki':  (1e-6, 0.05),
    'A0':  (1e-6, 1.0),
    'phi': (-1000, 1000),
}
aa_nwalkers = 32
aa_nsteps = 8000
    # [주의] 6개 파일을 전부 도니 총 실행시간이 어제 1개 파일 실행의
    # 약 6배가 됩니다. 시간이 너무 오래 걸리면 이 값을 줄여서(예:
    # 3000) 먼저 "구조가 맞는지"만 빠르게 확인하고, 이후 값을 늘려
    # 다시 돌리는 것도 좋은 전략입니다.
aa_burn_in_discard = 500
aa_thin_by = 15
aa_init_scatter = 1e-2
aa_init_scatter_floor = 0.5

print("=" * 60)
print("분석자 설정값(aa_*) 요약:")
for k, v in list(globals().items()):
    if k.startswith("aa_") and not isinstance(v, dict):
        print(f"  {k:24s} = {v}")
print("=" * 60)


# =========================================================
# STEP 1. 파일 하나를 피팅하는 함수로 감싸기
# =========================================================
def fit_one_file(filename, tau_fixed):
    """
    파일 하나를 읽어서, tau를 고정한 6개 파라미터(f0,ke1,ke2,ki,A0,phi)를
    MCMC로 추정하고, 피셔 행렬 진단까지 수행한 결과를 반환.

    tau_fixed : 이 파일에서 사용할 고정 tau 값. 파일마다(공진기마다)
        배선/캘리브레이션이 조금씩 다를 수 있으므로, 파일별 원저자
        참값의 tau를 그대로 가져다 씀 (어제 했던 것과 같은 논리 -
        tau는 물리와 무관한 장비 배경이라 별도로 안다고 가정).
    """
    full_path = os.path.join(aa_drive_folder, filename)
    data = zenodo_loader.load_power_sweep_txt(full_path)

    f_grid = data['freq_ghz']
    s21_measured = data['I'][aa_power_slice_index, :] + 1j * data['Q'][aa_power_slice_index, :]

    # --- 초기값을 데이터에서 직접 추정 (어제와 동일한 로직) ---
    mag = np.abs(s21_measured)
    f0_guess = f_grid[np.argmax(mag)]
        # 피크 위치를 f0 초기값으로. 물리적으로: 이 모델은 avoided-crossing
        # 딥이 아니라 단일 공진 "피크"이므로, |S21|이 최댓값을 갖는
        # 지점이 곧 공진 주파수(f0)에 대한 가장 직접적인 추정치가 됨.

    half_level = mag.max() / 2.0
    above_half = np.where(mag > half_level)[0]
    if len(above_half) > 1:
        fwhm_guess = f_grid[above_half[-1]] - f_grid[above_half[0]]
    else:
        fwhm_guess = 0.001
    total_kappa_guess = np.maximum(fwhm_guess, 1e-5)
        # FWHM(반치폭): 로렌츠 피크가 최대 높이의 절반이 되는 두
        # 지점 사이의 주파수 간격. 이론적으로 이 폭이 정확히
        # total_kappa(=ke1+ke2+ki)와 같아야 함 - 물리적으로 "얼마나
        # 빨리 에너지가 빠져나가는지"가 곧 공진 봉우리가 얼마나
        # 넓게 퍼지는지를 결정하기 때문.

    initial_guess = np.array([
        f0_guess,
        total_kappa_guess / 3, total_kappa_guess / 3, total_kappa_guess / 3,
        mag.max(),
        np.angle(s21_measured[np.argmax(mag)]),
    ])

    noise_sigma_est = (
        np.std(np.real(s21_measured[mag < mag.max() * 0.1]))
        if np.any(mag < mag.max() * 0.1)
        else np.std(np.real(s21_measured))
    )
        # baseline(공진에서 먼, 신호가 약한 구간)의 흔들림을 잡음 크기로
        # 추정. 여기서는 "피크"가 신호이므로, |S21|이 최댓값의 10%
        # 미만인 지점들을 baseline으로 간주 (avoided-crossing 딥 모델
        # 때와는 반대 - 그쪽은 |S21|이 1에 가까운 곳이 baseline이었음)

    # --- 우도/prior 준비 (어제와 동일 구조) ---
    param_order = ['f0', 'ke1', 'ke2', 'ki', 'A0', 'phi']
    log_prior = likelihood.make_uniform_log_prior(
        {name: aa_prior_bounds[name] for name in param_order}
    )

    def log_probability(theta, f_grid_local, data_1d, sigma):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        f0, ke1, ke2, ki, A0, phi = theta
        model = zenodo_models.s21_single_resonance_transmission(
            f_grid_local, f0, ke1, ke2, ki, A0, phi, tau_fixed
        )
        real_res = np.real(data_1d - model)
        imag_res = np.imag(data_1d - model)
        chi2 = np.sum((real_res / sigma) ** 2 + (imag_res / sigma) ** 2)
        return lp + (-0.5 * chi2)

    # --- MCMC 실행 ---
    result = mcmc_pipeline.run_single_mcmc(
        log_probability, initial_guess, f_grid, s21_measured, noise_sigma_est,
        nwalkers=aa_nwalkers, nsteps=aa_nsteps,
        init_scatter=aa_init_scatter, init_scatter_floor=aa_init_scatter_floor,
        burn_in_discard=aa_burn_in_discard, thin_by=aa_thin_by,
        bounds={name: aa_prior_bounds[name] for name in param_order},
        suppress_warnings=True,
    )

    # --- 피셔 행렬 진단 (NaN 여부로 축퇴를 자동 판정) ---
    fixed_kwargs_fisher = {'tau': tau_fixed}
    sigma_fisher, corr_fisher = zfm.fisher_parameter_uncertainties(
        zenodo_models.s21_single_resonance_transmission,
        result['median'], param_order, f_grid, noise_sigma_est, fixed_kwargs_fisher
    )
    n_singular = np.sum(np.isnan(sigma_fisher))
        # 이 파일에서 피셔 행렬이 특이(NaN)가 된 파라미터 개수.
        # 0이면 "이번엔 축퇴가 안 잡혔다"는 뜻이고, 이전 kappa=8MHz
        # 결과처럼 2 이상이면 "이번에도 축퇴가 확인됐다"는 뜻.

    return {
        'param_order': param_order,
        'median': result['median'],
        'sigma_fisher': sigma_fisher,
        'n_singular': n_singular,
        'converged': result['converged'],
    }


# =========================================================
# STEP 2. 6개 파일 전부 순회하며 반복 실행
# =========================================================
all_results = {}
print(f"\n총 {len(aa_file_reference_table)}개 파일에 대해 순차적으로 피팅 시작...\n")

for label, info in aa_file_reference_table.items():
    print(f"--- {label} ({info['filename']}) 처리 중 ---")
    tau_fixed = info['ref']['tau']
    fit_result = fit_one_file(info['filename'], tau_fixed)
    all_results[label] = fit_result

    # 이 파일에서 ke1-ke2 축퇴가 재현됐는지 즉시 요약 출력
    print(f"  수렴 여부: {fit_result['converged']}, "
          f"피셔 특이(NaN) 파라미터 개수: {fit_result['n_singular']} / 6")
    print()


# =========================================================
# STEP 3. 전체 요약표 - "축퇴가 kappa 크기와 무관하게 항상 나타나는가?"
# =========================================================
print("\n" + "=" * 70)
print("전체 요약: 6개 파일 모두에서 ke1-ke2-ki 축퇴가 재현되는가?")
print("=" * 70)
print(f"{'파일(kappa)':<16} {'수렴':>6} {'피셔 특이 개수':>14}  {'특이 파라미터'}")
print("-" * 70)
for label, res in all_results.items():
    singular_names = [res['param_order'][i] for i in range(len(res['param_order']))
                       if np.isnan(res['sigma_fisher'][i])]
    print(f"{label:<16} {str(res['converged']):>6} {res['n_singular']:>14}  {singular_names}")

n_files_with_degeneracy = sum(1 for res in all_results.values() if res['n_singular'] > 0)
print("-" * 70)
print(f"\n{n_files_with_degeneracy} / {len(all_results)} 개 파일에서 축퇴(피셔 특이)가 확인됨.")
if n_files_with_degeneracy == len(all_results):
    print("-> 모든 파일에서 축퇴가 재현됨: 이건 특정 데이터의 우연이 아니라")
    print("   이 물리 모델(단일 공진기 투과식) 자체의 수학적 구조")
    print("   (ke1*ke2가 항상 곱셈으로만 등장)에서 비롯된 근본적인")
    print("   한계라는 것이 강하게 뒷받침됩니다.")
else:
    print("-> 일부 파일에서만 축퇴가 나타남: kappa 크기(즉 ke1,ke2,ki의")
    print("   실제 비율)에 따라 축퇴 정도가 달라질 수 있다는 뜻이며,")
    print("   어느 조건에서 덜 심한지 추가로 살펴볼 가치가 있습니다.")
