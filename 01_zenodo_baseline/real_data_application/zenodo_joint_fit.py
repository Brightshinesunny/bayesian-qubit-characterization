"""
zenodo_joint_fit.py
=================================================================
[오늘의 실험 A: Joint fit] 여러 전력(power) 슬라이스를 "동시에"
피팅해서, 단일 slice에서 확인한 ke1-ke2-ki 축퇴가 실제로 풀리는지
확인하는 스크립트입니다.

[핵심 아이디어 - 왜 정보가 늘어나는가]
공진기 자체의 물리량(f0, ke1, ke2, ki)은 전력이 달라져도 바뀌지
않아야 합니다(선형 영역이라는 전제 하에). 반면 진폭 A0는 "이 특정
스캔에서 신호가 얼마나 세게 들어갔는지"를 나타내는 값이라 전력마다
다릅니다. 이렇게 "여러 데이터셋이 일부 파라미터는 공유하고 일부는
각자 갖는" 모델을 계층적 모델(hierarchical model)이라고 부릅니다
(likelihood.py의 설계 노트에서 이미 언급했던 확장 방향).

수학적으로, N개의 전력 슬라이스를 합치면:
  - 파라미터 개수: 4(공유: f0,ke1,ke2,ki) + N(개별: A0_1,...,A0_N) + 1(phi, 공유)
  - 데이터 포인트 개수: N * 101 (슬라이스 개수 * 주파수 포인트 수)
단일 slice보다 데이터는 N배 늘어나는데 공유 파라미터 개수는 그대로라,
f0/ke1/ke2/ki에 대한 "정보 밀도"가 N배로 늘어나는 효과가 생깁니다.
이게 축퇴를 풀 수 있는 이론적 근거입니다 - 다만 ke1*ke2가 "항상
곱셈으로만" 등장하는 구조 자체는 각 슬라이스에서 똑같이 반복되므로,
정말 풀리는지는 실제로 돌려봐야 압니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import zenodo_loader
import zenodo_models
import zenodo_likelihood as likelihood
import zenodo_mcmc_pipeline as mcmc_pipeline
import zenodo_diagnostics as diagnostics
import zenodo_bayesian_toolkit as bt
import zenodo_fisher_matrix as zfm


# =========================================================
# STEP 0. 분석자 설정값 (aa_ 접두사)
# =========================================================
aa_drive_folder = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module01/Figure1/'
aa_filename = 'S21_power_sweep_152_data_210823_21h50m19s.txt'   # kappa≈8MHz 파일 재사용
aa_output_dir = './outputs/'
os.makedirs(aa_output_dir, exist_ok=True)

aa_power_indices_to_join = [-1, -6, -11, -16, -21]
    # 어떤 전력 슬라이스들을 "동시에" 피팅할지 인덱스로 지정.
    # -1(최고 전력)부터 5dB 간격으로 5개를 골랐음 (원본 파일의 전력
    # 간격이 1dB이므로 -6은 5dB 낮은 지점). 너무 낮은 전력(신호가
    # 약해 잡음에 묻힘)은 피하기 위해 상위권 전력들 위주로 선택.
    # 늘리면(예: 10개) 정보가 더 늘어나지만 계산 시간도 그만큼 늘어남.

# tau는 어제와 동일하게 원저자 참값으로 고정 (phi-tau 축퇴 회피)
aa_reference_tau = 62.6379
aa_fixed_tau = aa_reference_tau

# 원저자 참값 (비교용으로만 사용, MCMC 어디에도 직접 넣지 않음)
aa_reference = dict(f0=10.4701, ke1=0.000492529, ke2=0.00699359,
                     ki=0.000457205, A0=0.0208633, phi=-43.4931)

aa_prior_bounds = {
    'f0':  (10.42, 10.52),
        # [수정] 원래 (10.40, 10.55)로, 실제 스캔 데이터 범위
        # (10.42~10.52 GHz)보다 넓게 잡았던 게 문제였음. corner plot에서
        # f0가 10.47(진짜 봉우리)과 10.54(가짜 봉우리) 두 곳에 나뉘어
        # 있는 게 확인됐는데, 10.54는 데이터 범위(10.52) 밖이라 애초에
        # "그 주파수에서 공진기가 어떻게 반응하는지" 정보 자체가
        # 데이터에 없는 위치임. 그런데도 prior가 그곳까지 열려 있으니,
        # 우연히 그쪽에서 출발한 워커들이 "정보가 없어 평평한 우도
        # 지형"에 갇혀 못 빠져나오는 문제가 생김 (일종의 "지도 밖" 함정).
        # prior를 실제 데이터가 존재하는 범위로 좁히면, 애초에 워커가
        # 그런 무의미한 위치에서 시작하는 일 자체를 막을 수 있음.
    'ke1': (1e-6, 0.05),
    'ke2': (1e-6, 0.05),
    'ki':  (1e-6, 0.05),
    'phi': (-1000, 1000),
    # A0는 슬라이스마다 따로 있으므로 아래에서 별도 처리
}
aa_A0_prior_bounds = (1e-4, 0.2)
    # [수정] 원래 (1e-6, 1.0)으로 너무 넓게 잡았던 게 문제였음.
    # 결과에서 A0 추정치가 0.19~0.50으로, 원저자 참값(~0.02)의
    # 10~25배나 큰 엉뚱한 영역에 워커들이 몰려있었던 것으로 확인됨.
    # 물리적으로: A0는 "신호 진폭 스케일"이고, 데이터 자체의 |S21|
    # 최대값(대략 0.005~0.02 수준, STEP 3의 A0_guess 출력 참고)에서
    # 크게 벗어날 이유가 없음. prior를 (1e-4, 0.2)로 좁혀서, MCMC가
    # 물리적으로 말이 안 되는 먼 영역까지 헤매지 않도록 제한.
    # (이렇게 좁히는 것도 일종의 "정보를 추가하는 것"이지만, 이번엔
    # "장비가 낼 수 있는 신호 크기의 상식적 범위"라는 약한 사전지식을
    # 반영하는 것이라, 어제 tau를 정확한 참값으로 고정했던 것보다는
    # 훨씬 약한 형태의 정보 주입입니다.)

aa_nwalkers = 48
    # [수정] 파라미터 개수가 늘어나므로(4 공유 + 1 phi + N개 A0)
    # 워커 개수도 여유있게 늘림. emcee는 관례적으로 최소
    # "차원 수의 2배 이상"을 권장.
aa_nsteps = 8000
aa_burn_in_discard = 500
aa_thin_by = 15
aa_init_scatter = 1e-2
aa_init_scatter_floor = 0.02
    # [수정] 원래 0.5로, "phi가 정확히 0에서 시작하면 모든 워커가
    # 겹쳐 시작한다"는 문제(예전 avoided-crossing 실험에서 확인했던
    # 것)를 막으려고 도입했던 안전장치였음. 하지만 이번엔 STEP 3에서
    # phi 초기값을 원형평균으로 구해 이미 0이 아닌 값(-0.983)에서
    # 출발하므로, 그 정도로 큰 floor가 더는 필요 없음.
    # 오히려 floor=0.5는 f0(prior 범위 폭이 겨우 0.1)처럼 스케일이
    # 작은 파라미터에는 지나치게 큰 흩뿌림을 강제해서, 워커들이 prior
    # 경계 밖으로 밀려나 clip되고 그 경계에 쌓이는 부작용을 낳았음
    # (corner plot에서 f0=10.54 근처의 가짜 봉우리가 정확히 이 문제의
    # 증거). floor를 0.02로 크게 낮춰서, "0에서 시작하는 파라미터가
    # 있을 때만 최소한으로 흩어지게 하는" 원래 목적만 살리고, 다른
    # 파라미터(특히 f0)에는 과도한 영향을 주지 않도록 함.

print("=" * 60)
print("분석자 설정값(aa_*) 요약:")
for k, v in list(globals().items()):
    if k.startswith("aa_") and not isinstance(v, dict):
        print(f"  {k:24s} = {v}")
print("=" * 60)


# =========================================================
# STEP 1. 데이터 로드 - 여러 전력 슬라이스를 한 번에 준비
# =========================================================
full_path = os.path.join(aa_drive_folder, aa_filename)
data = zenodo_loader.load_power_sweep_txt(full_path)
f_grid = data['freq_ghz']

n_slices = len(aa_power_indices_to_join)
s21_slices = [
    data['I'][idx, :] + 1j * data['Q'][idx, :]
    for idx in aa_power_indices_to_join
]
power_values_used = [data['power_dbm'][idx] for idx in aa_power_indices_to_join]
print(f"\n합쳐서 피팅할 전력값들: {power_values_used} dBm ({n_slices}개 slice)")


# =========================================================
# STEP 2. 파라미터 벡터 설계 - "공유 4개 + phi 1개 + 개별 A0 N개"
# =========================================================
# theta = [f0, ke1, ke2, ki, phi, A0_1, A0_2, ..., A0_N]
#          <-------- 공유(5개) --------> <-- 슬라이스별(N개) -->
param_order_shared = ['f0', 'ke1', 'ke2', 'ki', 'phi']
n_shared = len(param_order_shared)
n_total_params = n_shared + n_slices
    # 예: n_slices=5 이면 총 5(공유)+5(개별 A0)=10개 파라미터.
    # 단일 slice 피팅(6개)보다 늘었지만, 데이터도 5배 늘었으므로
    # "파라미터당 정보량"은 오히려 개선될 수 있음 (STEP 0 설명 참고).


def unpack_theta(theta):
    """
    theta 배열을 (공유 파라미터 딕셔너리, 개별 A0 리스트)로 분리하는
    헬퍼 함수. joint 우도 계산과 초기값 생성 양쪽에서 반복해서 쓰이는
    로직이라 함수로 뽑아둠 (같은 코드를 두 번 쓰지 않기 위함).
    """
    shared = dict(zip(param_order_shared, theta[:n_shared]))
    A0_list = theta[n_shared:]
    return shared, A0_list


# =========================================================
# STEP 3. 초기값을 각 slice 데이터에서 개별 추정 후 결합
# =========================================================
def estimate_initial_guess_single(f_grid_local, s21_1d):
    """
    단일 slice에서 (f0, kappa_total, A0, phi) 초기값을 추정.
    zenodo_fit_and_compare.py의 STEP 2 로직과 동일 (재사용을 위해
    이번엔 아예 함수로 뽑아둠).
    """
    mag = np.abs(s21_1d)
    f0_guess = f_grid_local[np.argmax(mag)]

    half_level = mag.max() / 2.0
    above_half = np.where(mag > half_level)[0]
    if len(above_half) > 1:
        fwhm_guess = f_grid_local[above_half[-1]] - f_grid_local[above_half[0]]
    else:
        fwhm_guess = 0.001
    kappa_guess = np.maximum(fwhm_guess, 1e-5)

    A0_guess = mag.max()
    phi_guess = np.angle(s21_1d[np.argmax(mag)])
    return f0_guess, kappa_guess, A0_guess, phi_guess


# 여러 slice의 개별 추정치를 모아서, 공유 파라미터는 "평균"으로,
# 개별 A0는 "각자의 값"으로 초기값을 구성
f0_guesses, kappa_guesses, A0_guesses, phi_guesses = [], [], [], []
for s21_1d in s21_slices:
    f0_g, kappa_g, A0_g, phi_g = estimate_initial_guess_single(f_grid, s21_1d)
    f0_guesses.append(f0_g)
    kappa_guesses.append(kappa_g)
    A0_guesses.append(A0_g)
    phi_guesses.append(phi_g)

phi_mean = np.angle(np.mean(np.exp(1j * np.array(phi_guesses))))
    # [수정] phi(위상)는 -pi~+pi 사이로 "감기는(wrap)" 값이라, 그냥
    # np.mean()으로 산술평균을 내면 위험함 - 예를 들어 phi 값이
    # 각각 -3.1과 +3.1이면(둘 다 거의 같은 방향인데 부호만 다르게
    # 감긴 것) 산술평균은 0이 되어버려 완전히 엉뚱한 초기값이 됨.
    # 이를 피하려면 "원형 평균(circular mean)"을 씀: 각 각도를
    # 단위원 위의 점(cos,sin)으로 바꿔서 평균낸 뒤, 그 평균 벡터의
    # 각도를 다시 구하는 방식. np.exp(1j*phi)로 phi를 복소평면의
    # 단위벡터로 바꾸고, 평균낸 뒤 np.angle로 다시 각도를 뽑아냄.

initial_guess = np.concatenate([
    [np.mean(f0_guesses)],                       # f0: 여러 slice 평균
    [np.mean(kappa_guesses) / 3] * 3,             # ke1,ke2,ki: 평균 kappa를 3등분
    [phi_mean],                                    # phi: 원형 평균 (수정됨)
    A0_guesses,                                    # 개별 A0: slice마다 그대로
])
    # np.concatenate: 여러 배열(또는 리스트)을 하나의 1차원 배열로
    # 이어붙이는 함수. 여기서는 "공유 파라미터 4개 묶음"과 "개별 A0
    # 리스트"를 순서대로 이어붙여 최종 theta 초기값을 만듦.

print(f"\n[결합 초기값] f0={initial_guess[0]:.4f}, "
      f"ke1=ke2=ki={initial_guess[1]:.5f}, phi={initial_guess[4]:.3f}")
print(f"  개별 A0 초기값: {np.round(initial_guess[n_shared:], 5)}")
print(f"  (참고: slice별 phi 원본값 {np.round(phi_guesses, 3)} -> 원형평균 {phi_mean:.3f})")



# =========================================================
# STEP 4. Joint likelihood 정의 - 모든 slice의 잔차를 한꺼번에 합산
# =========================================================
def estimate_noise_sigma_peak(s21_1d):
    mag = np.abs(s21_1d)
    mask = mag < mag.max() * 0.1
    if np.any(mask):
        return np.std(np.real(s21_1d[mask]))
    return np.std(np.real(s21_1d))


noise_sigmas = [estimate_noise_sigma_peak(s21_1d) for s21_1d in s21_slices]
print(f"\n각 slice의 noise_sigma: {np.round(noise_sigmas, 6)}")

# prior: 공유 파라미터 4개 + 개별 A0 N개를 모두 포함하는 하나의 균일 prior
full_bounds = {name: aa_prior_bounds[name] for name in param_order_shared}
for i in range(n_slices):
    full_bounds[f'A0_{i}'] = aa_A0_prior_bounds
log_prior = likelihood.make_uniform_log_prior(full_bounds)
    # make_uniform_log_prior는 {이름:(min,max)} 딕셔너리의 "순서"를
    # 그대로 param_names로 사용하므로, 여기서 딕셔너리를 만든 순서
    # (공유 4개 -> A0_0, A0_1, ...)가 theta의 순서와 정확히 일치해야 함
    # (파이썬 3.7+ 에서는 딕셔너리가 삽입 순서를 보장하므로 안전).


def joint_log_likelihood(theta):
    """
    N개 slice 전체의 잔차를 하나로 합쳐 로그우도를 계산.

    물리적으로: 여러 개의 독립적인 측정(서로 다른 전력에서 잰 스펙트�럼)이
    있을 때, 전체 데이터가 주는 정보는 각 측정의 로그우도를 "그냥
    더한 것"과 같습니다 (측정끼리 서로 독립이라고 가정하면, 전체
    확률은 각 확률의 곱이 되고, 로그를 취하면 곱셈이 덧셈이 됨 -
    likelihood.py에서 이미 다룬 원리와 동일).
    """
    shared, A0_list = unpack_theta(theta)
    total_log_lik = 0.0
    for i in range(n_slices):
        model = zenodo_models.s21_single_resonance_transmission(
            f_grid, shared['f0'], shared['ke1'], shared['ke2'], shared['ki'],
            A0_list[i], shared['phi'], aa_fixed_tau
        )
        real_res = np.real(s21_slices[i] - model)
        imag_res = np.imag(s21_slices[i] - model)
        chi2 = np.sum((real_res / noise_sigmas[i]) ** 2 + (imag_res / noise_sigmas[i]) ** 2)
        total_log_lik += -0.5 * chi2
            # 여러 slice의 chi2를 각자 계산해서 로그우도에 누적(+=).
            # 이게 "joint"(결합) 우도의 핵심 - 각 slice가 독립적으로
            # 참여하되, f0/ke1/ke2/ki/phi는 모든 slice가 "같은 값을
            # 공유"하도록 강제됨 (shared 딕셔너리를 모든 slice에
            # 동일하게 사용하므로).
    return total_log_lik


def joint_log_probability(theta, *_unused_args):
    """
    mcmc_pipeline.run_single_mcmc는 log_probability(theta, f_grid,
    data_1d, sigma) 형태의 4-인자 함수를 기대하지만, joint fit은
    "여러 개의 f_grid/data_1d/sigma"를 이미 함수 안에 클로저로 담고
    있으므로 나머지 인자는 안 씀 (*_unused_args로 받아서 무시).
    이렇게 하면 run_single_mcmc를 코드 수정 없이 그대로 재사용 가능.
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + joint_log_likelihood(theta)


# =========================================================
# STEP 5. MCMC 실행
# =========================================================
print(f"\n총 파라미터 개수: {n_total_params} (공유 {n_shared}개 + 개별 A0 {n_slices}개)")
print("Joint MCMC 실행 중...")

result = mcmc_pipeline.run_single_mcmc(
    joint_log_probability, initial_guess,
    f_grid, None, None,
        # [주의] f_grid, data_1d, sigma 자리에 None을 넘김 - joint_log_probability가
        # 이 인자들을 안 쓰고 클로저로 이미 필요한 데이터를 다 갖고
        # 있기 때문. run_single_mcmc는 이 값들을 그대로 emcee의 args로
        # 전달만 할 뿐 직접 사용하지 않으므로 None이어도 문제없음.
    nwalkers=aa_nwalkers, nsteps=aa_nsteps,
    init_scatter=aa_init_scatter, init_scatter_floor=aa_init_scatter_floor,
    burn_in_discard=aa_burn_in_discard, thin_by=aa_thin_by,
    bounds=full_bounds,
    suppress_warnings=True,
)

print(f"\n수렴 여부: {result['converged']}")
if result['autocorr_time'] is not None:
    print(f"자기상관 시간: {np.round(result['autocorr_time'], 1)}")


# =========================================================
# STEP 6. 결과 vs 참값 비교 (공유 파라미터만 - A0는 slice마다 다르므로 생략)
# =========================================================
print("\n" + "=" * 65)
print(f"{'파라미터':<6} {'Joint 추정':>14} {'단일slice 추정':>16} {'원저자 참값':>14}")
print("=" * 65)
# 참고: "단일slice 추정"은 어제 zenodo_fit_and_compare.py의 kappa=8MHz
# 결과(가장 최근 실행값)를 하드코딩 참고용으로 넣음. 실제 비교시
# 자신의 최신 실행 결과로 교체 가능.
single_slice_estimate = {'f0': 10.470470, 'ke1': 0.000825, 'ke2': 0.000561,
                           'ki': 0.007513, 'phi': 0.479402}
for i, name in enumerate(param_order_shared):
    est = result['median'][i]
    ref = aa_reference[name]
    single = single_slice_estimate.get(name, float('nan'))
    print(f"{name:<6} {est:>14.6f} {single:>16.6f} {ref:>14.6f}")

print("\n개별 A0 결과:")
for i in range(n_slices):
    est = result['median'][n_shared + i]
    err_lo = result['err_lo'][n_shared + i]
    err_hi = result['err_hi'][n_shared + i]
    print(f"  A0 (power={power_values_used[i]:.0f}dBm): "
          f"{est:.5f} (+{err_hi:.5f}/-{err_lo:.5f})")


# =========================================================
# STEP 7. 피셔 행렬로 축퇴가 실제로 줄었는지 재확인
# =========================================================
# joint 모델 전용 피셔 행렬을 계산하려면, zenodo_fisher_matrix.py의
# numerical_derivative가 기대하는 "model_func(f, **kwargs)" 형태가
# 아니라 "여러 slice에 대한 결합 모델"이 필요하므로, 여기서는 간단히
# "공유 파라미터 4개만" 떼어내 단일 slice 모델의 민감도를 비교하는
# 대신, MCMC posterior 자체의 R-hat과 상관계수로 축퇴 감소를 확인.
print("\n" + "=" * 60)
print("STEP 7. R-hat 및 공유 파라미터 상관관계 확인")
print("=" * 60)
for i, name in enumerate(param_order_shared):
    raw_chain_i = result['raw_chain'][:, :, i].T
    rhat_i = bt.gelman_rubin_rhat(raw_chain_i)
    flag = "  ⚠️" if rhat_i > 1.01 else "  ✅ 수렴 양호"
    print(f"  {name:<6} R-hat = {rhat_i:.4f}{flag}")

# ke1-ke2 상관계수를 posterior 샘플에서 직접 계산해, 어제(단일
# slice, corner plot에서 거의 완벽한 대각선)와 비교
ke1_samples = result['flat_samples'][:, 1]
ke2_samples = result['flat_samples'][:, 2]
corr_ke1_ke2 = np.corrcoef(ke1_samples, ke2_samples)[0, 1]
    # np.corrcoef: 두 배열 사이의 피어슨 상관계수 행렬을 계산하는 함수.
    # [0,1] 위치가 곧 두 변수 사이의 상관계수 값.
print(f"\nke1-ke2 posterior 상관계수: {corr_ke1_ke2:.4f}")
print(f"  (어제 단일 slice에서는 거의 -1 또는 +1에 가까운 완벽한 대각선이었음.")
print(f"   이 값이 그보다 0에 더 가까워졌다면, joint fit으로 축퇴가")
print(f"   실제로 완화된 것으로 해석할 수 있습니다.)")


# =========================================================
# STEP 8. Corner plot (공유 파라미터만 표시 - A0는 너무 많아 생략)
# =========================================================
try:
    import corner
    shared_samples = result['flat_samples'][:, :n_shared]
        # [:, :n_shared] : 전체 posterior 샘플에서 앞의 5개 열(공유
        # 파라미터)만 슬라이싱. 개별 A0(N개)는 corner plot이 너무
        # 커지는 것을 피하기 위해 제외.
    reference_shared = [aa_reference[name] for name in param_order_shared]

    fig_corner = corner.corner(
        shared_samples,
        labels=[r'$f_0$', r'$k_{e1}$', r'$k_{e2}$', r'$k_i$', r'$\phi$'],
        truths=reference_shared,
        truth_color='red',
        show_titles=True,
        title_kwargs={'fontsize': 9},
    )
    fig_corner.suptitle(f'Joint Fit ({n_slices} slices) - 공유 파라미터 Corner Plot',
                          y=1.01, fontsize=13)
    plt.savefig(os.path.join(aa_output_dir, 'zenodo_joint_corner_plot.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nCorner plot 저장 완료: "
          f"{os.path.join(aa_output_dir, 'zenodo_joint_corner_plot.png')}")
except ImportError:
    print("\n[안내] 'corner' 패키지가 없어 corner plot을 건너뜁니다.")
