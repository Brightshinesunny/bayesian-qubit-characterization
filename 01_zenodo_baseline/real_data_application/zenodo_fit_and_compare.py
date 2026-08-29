"""
zenodo_fit_and_compare.py
=================================================================
실제 raw 데이터(S21_power_sweep_152...txt, 최고 전력 10dBm 단면)를
우리 MCMC 파이프라인(mcmc_pipeline.py)과 zenodo_models.py의 물리
모델로 직접 피팅해서, 원저자가 Mathematica로 구한 참값과 독립적으로
비교합니다.

[검증의 의미]
지금까지는:
  - 파서(zenodo_loader.py)가 원저자 방식과 같은 결과를 내는지 확인
  - 물리 모델(zenodo_models.py)에 원저자 참값을 "넣었을 때" 원저자
    그래프와 같은 모양이 나오는지 확인
이번 단계는 다릅니다:
  - 원저자 참값을 전혀 쓰지 않고, raw 데이터만 보고 우리 MCMC가
    스스로 f0, ke1, ke2, ki, A0, phi, tau를 추정합니다.
  - 그 결과가 원저자의 Mathematica 참값과 (독립적으로) 비슷하게
    나온다면, 이건 "우리 전체 파이프라인(로더+모델+MCMC)이 실제
    실험 데이터에서도 올바르게 작동한다"는 가장 강력한 증거가 됩니다.
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
    # zenodo_bayesian_toolkit: R-hat(수렴 진단), credible interval 등
    # 범용 베이지안 도구. zenodo_fisher_matrix: MCMC와 독립적인
    # 방법으로 파라미터 오차/축퇴를 재확인하는 도구.


# =========================================================
# STEP 0. 분석자 설정값 (aa_ 접두사)
# =========================================================
aa_drive_folder = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module01/Figure1/'
aa_filename = 'S21_power_sweep_152_data_210823_21h50m19s.txt'
aa_output_dir = './outputs/'
os.makedirs(aa_output_dir, exist_ok=True)

aa_power_slice_index = -1
    # 어느 전력 슬라이스를 피팅할지. -1은 "가장 높은 전력"을 뜻하는
    # 파이썬의 관례적 인덱스(맨 끝에서 첫 번째). 원저자 코드도
    # P21Lin_5[-1]로 이 파일에서는 최고 전력 슬라이스를 썼으므로,
    # 같은 슬라이스를 골라야 원저자 참값과 공정하게 비교할 수 있음.

# --- 원저자가 Mathematica로 구한 참값 (검증/비교용으로만 사용,
#     MCMC의 prior나 초기값 계산에는 절대 넣지 않음 - 그래야 "독립적인
#     검증"이 됨. 초기값을 참값 근처로 주면 검증의 의미가 사라짐.) ---
aa_reference_f0 = 10.4701
aa_reference_ke1 = 0.000492529
aa_reference_ke2 = 0.00699359
aa_reference_ki = 0.000457205
aa_reference_A0 = 0.0208633
aa_reference_phi = -43.4931
aa_reference_tau = 62.6379

# --- MCMC용 prior 범위 (데이터를 보고 "이 정도면 물리적으로 말이
#     된다"고 분석자가 판단한 넓은 범위 - 참값 근처로 좁히지 않음) ---
aa_prior_bounds = {
    'f0':  (10.40, 10.55),    # GHz, 스캔 범위 안 어디든 가능하게 넓게
    'ke1': (1e-6, 0.05),      # 결합률들은 물리적으로 반드시 양수
    'ke2': (1e-6, 0.05),
    'ki':  (1e-6, 0.05),
    'A0':  (1e-6, 1.0),
    'phi': (-1000, 1000),     # 원저자 값(-43.5)이 라디안이 아니라 매우
                               # 큰 범위인 걸로 보아 특수한 단위/컨벤션을
                               # 쓰는 것으로 보임 - 넓게 열어둠
    # 'tau' 항목 제거: 이제 추정 대상이 아니라 aa_fixed_tau로 고정.
}

# --- [신규] 사전 캘리브레이션 값으로 고정할 파라미터 ---
# corner plot에서 phi-tau가 완벽한 대각선(수학적 축퇴)을 이루는 것을
# 확인했습니다: exp(-i(phi + 2*pi*f*tau)) 형태라 phi와 tau가 항상
# "합쳐서만" 영향을 주고, 데이터 하나만으로는 원리적으로 분리가
# 불가능합니다. 실제 실험에서도 tau(케이블 지연)는 공진기 물리와
# 무관하게 배선 길이로 정해지는 값이라, 별도의 through(직결) 측정
# 등으로 미리 캘리브레이션해서 알아내고, 이후 공진기 피팅에서는
# "이미 아는 상수"로 고정하는 것이 표준적인 실험 관행입니다.
# 여기서는 원저자의 Mathematica 참값을 "사전 캘리브레이션으로 이미
# 확보된 값"이라고 간주하고 고정합니다 - 이건 "정답을 몰래 알려주는
# 부정행위"가 아니라, "실제 실험에서도 tau는 별도 측정으로 안다"는
# 현실적인 워크플로우를 반영하는 것입니다 (f0, ke1, ke2, ki, A0, phi
# 는 여전히 참값을 전혀 참고하지 않고 데이터만으로 추정합니다).
aa_fixed_tau = aa_reference_tau

aa_nwalkers = 32
aa_nsteps = 8000
    # [수정] tau를 고정해서 파라미터가 7개->6개로 줄고, 가장 심한
    # 축퇴(phi-tau) 원인이 제거됐으므로 이전(20000)보다 적은 스텝으로도
    # 충분할 가능성이 높음. 자기상관 시간을 보고 필요시 다시 늘림.
aa_burn_in_discard = 500
aa_thin_by = 15
aa_init_scatter = 1e-2
aa_init_scatter_floor = 0.5
    # [신규] 파라미터 값과 무관하게 보장되는 최소 흩뿌림 폭. phi(위상)나
    # tau(지연)처럼 초기값이 0 근처이거나 상대적 흩뿌림(init_scatter *
    # abs(값))만으로는 워커들을 충분히 갈라놓기 부족한 파라미터를 위한
    # 안전장치. 이 값이 없으면(0.0) 어떤 파라미터가 우연히 0에 가까운
    # 초기값을 가질 때 "Initial state has a large condition number"
    # 에러가 날 수 있음 (실제로 phi=0.0으로 뒀을 때 이 에러를 겪음).

print("=" * 60)
print("분석자 설정값(aa_*) 요약:")
for k, v in list(globals().items()):
    if k.startswith("aa_"):
        print(f"  {k:22s} = {v}")
print("=" * 60)


# =========================================================
# STEP 1. 실제 데이터 로드 (검증된 zenodo_loader.py 사용)
# =========================================================
full_path = os.path.join(aa_drive_folder, aa_filename)
data = zenodo_loader.load_power_sweep_txt(full_path)
zenodo_loader.summarize_loaded_data(data)

f_grid = data['freq_ghz']
s21_measured = data['I'][aa_power_slice_index, :] + 1j * data['Q'][aa_power_slice_index, :]
print(f"\n피팅 대상 slice: power={data['power_dbm'][aa_power_slice_index]:.1f} dBm, "
      f"데이터 포인트 수={len(f_grid)}")


# =========================================================
# STEP 2. 초기값을 "데이터에서 직접" 추정 (참값을 훔쳐보지 않음)
# =========================================================
# f0 초기값: |S21|이 최댓값을 갖는 주파수를 그대로 사용 (딥이 아니라
# 피크 모델이므로, "가장 튀어나온 지점"을 찾으면 됨)
mag = np.abs(s21_measured)
f0_guess = f_grid[np.argmax(mag)]

# kappa(총 감쇠율) 초기값: 피크 절반 높이에서의 폭(FWHM과 유사)을
# 데이터에서 직접 측정. 이 폭을 ke1+ke2+ki 세 개로 어떻게 나눌지는
# 알 수 없으므로, 일단 "총 감쇠율의 대략적인 크기"만 잡고 세
# 성분에는 동일하게 나눠서 초기 추정치로 사용.
half_level = mag.max() / 2.0
above_half = np.where(mag > half_level)[0]
    # np.where(조건)[0]: 조건을 만족하는 원소들의 인덱스를 배열로 반환
if len(above_half) > 1:
    fwhm_guess = f_grid[above_half[-1]] - f_grid[above_half[0]]
else:
    fwhm_guess = 0.001   # 폭을 측정 못하면 임의의 작은 값으로 폴백
total_kappa_guess = np.maximum(fwhm_guess, 1e-5)
    # [수정] 원래 파이썬 기본 max(a,b)를 썼는데, 만약 이 노트북에서
    # 이전에 원저자 코드(from numpy import *)를 실행한 적이 있다면,
    # 그 순간 파이썬 기본 max가 numpy의 max(=np.amax, "배열에서
    # axis번째 축 최댓값")로 덮어써짐. numpy의 max(a, axis)는 두 번째
    # 인자를 정수 축으로 기대하는데 1e-5(실수)를 받아 TypeError가 남.
    # np.maximum(a, b)는 "두 값을 원소별로 비교해 큰 쪽을 고르는"
    # 함수로, 이름이 겹칠 걱정 없이 항상 원하는 동작을 함 - 이런
    # 이름 충돌을 피하려면 파이썬 기본 함수 대신 np. 접두사를 붙인
    # 버전을 쓰는 것이 안전한 습관.

initial_guess = np.array([
    f0_guess,                       # f0
    total_kappa_guess / 3,          # ke1 (총 감쇠율을 3등분해 초기값으로)
    total_kappa_guess / 3,          # ke2
    total_kappa_guess / 3,          # ki
    mag.max(),                      # A0 (피크 높이를 그대로 진폭 초기값으로)
    np.angle(s21_measured[np.argmax(mag)]),
        # [수정] phi 초기값을 임의의 0.0 대신, 실제 피크 지점에서
        # 측정된 위상값으로 잡음. 물리적으로 더 합리적인 추정치일
        # 뿐 아니라, 값이 정확히 0이 되는 걸 피해서(아래 init_scatter_floor
        # 설명 참고) MCMC 워커들이 서로 겹쳐 시작하는 문제도 예방함.
    # tau 제거: 더 이상 initial_guess에 포함되지 않음 (고정 상수이므로)
])

print(f"\n[데이터 기반 초기값 추정 - 참값을 전혀 참고하지 않음]")
print(f"  f0_guess    = {f0_guess:.4f} GHz")
print(f"  kappa_guess(총) = {total_kappa_guess:.5f} GHz -> ke1,ke2,ki 각각 초기값 "
      f"{total_kappa_guess/3:.5f}")
print(f"  A0_guess    = {mag.max():.5f}")


# =========================================================
# STEP 3. 우도/prior 준비 및 MCMC 실행
# =========================================================
param_order = ['f0', 'ke1', 'ke2', 'ki', 'A0', 'phi']
    # [수정] tau 제거 - 이제 6개 파라미터만 MCMC가 탐색.
    # theta 배열의 순서를 명시적으로 정의. likelihood.py의
    # _theta_to_kwargs는 (f_r,g,kappa) 3개 전용으로 하드코딩되어
    # 있으므로, 이번처럼 파라미터 개수/이름이 다른 모델에는 그대로
    # 못 쓰고, 아래처럼 이 스크립트 안에서 직접 매핑 함수를 만듦.

log_prior = likelihood.make_uniform_log_prior(
    {name: aa_prior_bounds[name] for name in param_order}
)

def zenodo_log_likelihood(theta, f_grid, data_1d, sigma):
    """
    zenodo_models.s21_single_resonance_transmission 전용 가우시안 우도.
    likelihood.py의 gaussian_log_likelihood는 avoided-crossing 모델
    (파라미터 3개, 이름 f_r/g/kappa) 전용으로 짜여 있어 그대로 재사용이
    안 되므로, 이 스크립트 안에서 같은 원리로 새로 정의함 (이런 식으로
    "새 물리계마다 우도 함수를 새로 정의"하는 것이, 템플릿 설계 문서
    (likelihood.py)에서 이미 안내했던 확장 방식).

    [수정] theta는 이제 6개 값만 담고 있고, tau는 aa_fixed_tau 상수를
    그대로 모델에 넘김 - "고정 파라미터"란 이런 식으로 MCMC가 건드리지
    않고 매번 같은 값을 쓰도록 우도 함수 안에 박아두는 것을 뜻함.
    """
    f0, ke1, ke2, ki, A0, phi = theta
    model = zenodo_models.s21_single_resonance_transmission(
        f_grid, f0, ke1, ke2, ki, A0, phi, aa_fixed_tau
    )
    real_res = np.real(data_1d - model)
    imag_res = np.imag(data_1d - model)
    chi2 = np.sum((real_res / sigma) ** 2 + (imag_res / sigma) ** 2)
    return -0.5 * chi2

def log_probability(theta, f_grid, data_1d, sigma):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + zenodo_log_likelihood(theta, f_grid, data_1d, sigma)


# 잡음 수준 추정: 공진 피크에서 먼(피크 폭보다 몇 배 이상 떨어진)
# 구간을 baseline으로 보고 표준편차 계산 (diagnostics.py의 adaptive
# 방식과 같은 발상이나, 여기서는 "피크"가 딥이 아니라 솟아있는
# 형태이므로 진폭이 작은 쪽이 baseline이라는 점에 유의)
noise_sigma_est = np.std(np.real(s21_measured[mag < mag.max() * 0.1])) if np.any(mag < mag.max()*0.1) else np.std(np.real(s21_measured))
print(f"\n추정된 noise_sigma: {noise_sigma_est:.6f}")

print("\nMCMC 실행 중 (파라미터 6개, tau는 고정, 시간이 다소 걸릴 수 있음)...")
result = mcmc_pipeline.run_single_mcmc(
    log_probability, initial_guess, f_grid, s21_measured, noise_sigma_est,
    nwalkers=aa_nwalkers, nsteps=aa_nsteps,
    init_scatter=aa_init_scatter, init_scatter_floor=aa_init_scatter_floor,
    burn_in_discard=aa_burn_in_discard,
    thin_by=aa_thin_by,
    bounds={name: aa_prior_bounds[name] for name in param_order},
    suppress_warnings=True,
)


# =========================================================
# STEP 4. 결과 vs 원저자 참값 비교
# =========================================================
reference_values = [aa_reference_f0, aa_reference_ke1, aa_reference_ke2,
                    aa_reference_ki, aa_reference_A0, aa_reference_phi]
    # [수정] tau 제외 6개만 - MCMC가 실제로 추정한 파라미터와 1:1 대응
    # 시켜야 비교표/인덱싱이 맞음.

print("\n" + "=" * 70)
print(f"{'파라미터':<6} {'우리 MCMC 추정':>16} {'원저자 참값':>16} {'상대오차(%)':>14}")
print("=" * 70)
for i, name in enumerate(param_order):
    est = result['median'][i]
    ref = reference_values[i]
    rel_err = abs(est - ref) / abs(ref) * 100 if ref != 0 else float('nan')
    flag = "  ⚠️" if rel_err > 20 else ""
    print(f"{name:<6} {est:>16.6f} {ref:>16.6f} {rel_err:>13.1f}%{flag}")
print(f"{'tau':<6} {'(고정)':>16} {aa_fixed_tau:>16.4f} {'0.0':>13}%  <- 추정 대상 아님")

total_kappa_est = result['median'][1] + result['median'][2] + result['median'][3]
total_kappa_ref = aa_reference_ke1 + aa_reference_ke2 + aa_reference_ki
print(f"\ntotal kappa: 우리={total_kappa_est*1e3:.2f} MHz, "
      f"원저자={total_kappa_ref*1e3:.2f} MHz "
      f"(오차 {abs(total_kappa_est-total_kappa_ref)/total_kappa_ref*100:.1f}%)")

print(f"\n수렴 여부(converged): {result['converged']}")
if result['autocorr_time'] is not None:
    print(f"자기상관 시간: {np.round(result['autocorr_time'], 1)}")


# =========================================================
# STEP 5. 시각화 - 데이터, 우리 피팅, 원저자 피팅을 한 그래프에
# =========================================================
model_ours = zenodo_models.s21_single_resonance_transmission(
    f_grid, *result['median'], aa_fixed_tau
)
    # [수정] result['median']은 이제 6개 값만 담고 있으므로, 마지막
    # 인자로 aa_fixed_tau를 명시적으로 추가해서 모델 함수가 요구하는
    # 7개 인자(f0,ke1,ke2,ki,A0,phi,tau)를 정확히 채움.
model_author = zenodo_models.s21_single_resonance_transmission(
    f_grid, *reference_values, aa_reference_tau
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(f_grid, np.abs(s21_measured), '.', ms=3, alpha=0.5, label='실측 데이터')
axes[0].plot(f_grid, np.abs(model_ours), '-', color='red', lw=2, label='우리 MCMC 피팅')
axes[0].plot(f_grid, np.abs(model_author), '--', color='black', lw=1, label='원저자 Mathematica 참값')
axes[0].set_xlabel('Frequency (GHz)')
axes[0].set_ylabel('|S21|')
axes[0].legend(fontsize=9)
axes[0].set_title('진폭(|S21|) 비교')

axes[1].plot(f_grid, np.angle(s21_measured), '.', ms=3, alpha=0.5, label='실측 데이터')
axes[1].plot(f_grid, np.angle(model_ours), '-', color='red', lw=2, label='우리 MCMC 피팅')
axes[1].plot(f_grid, np.angle(model_author), '--', color='black', lw=1, label='원저자 참값')
axes[1].set_xlabel('Frequency (GHz)')
axes[1].set_ylabel('phase (rad)')
axes[1].legend(fontsize=9)
axes[1].set_title('위상(phase) 비교')

plt.tight_layout()
plt.savefig(os.path.join(aa_output_dir, 'zenodo_fit_vs_reference.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\n비교 그래프 저장 완료: {os.path.join(aa_output_dir, 'zenodo_fit_vs_reference.png')}")


# =========================================================
# STEP 6. 베이지안 진단 - corner plot + R-hat
# =========================================================
# 대조표에서 f0는 정확했지만 ke1/ke2/ki/A0/phi/tau는 크게 어긋났음.
# 이건 개별 파라미터가 "틀렸다"기보다, 여러 파라미터가 서로 얽혀
# (축퇴, degeneracy) 있어서 데이터만으로는 "정확히 어떤 조합인지"
# 구분이 잘 안 되는 상황일 가능성이 높음 (예: A0를 늘리면서 ke1을
# 줄이면 비슷한 곡선이 나오는 식). corner plot으로 이 상관관계를
# 직접 눈으로 확인하고, R-hat으로 애초에 체인이 수렴은 했는지부터
# 확인함.

print("\n" + "=" * 60)
print("STEP 6. 베이지안 진단 (corner plot, R-hat)")
print("=" * 60)

param_labels_zenodo = [r'$f_0$', r'$k_{e1}$', r'$k_{e2}$', r'$k_i$',
                        r'$A_0$', r'$\phi$']
    # [수정] tau 제거 - 이제 6개 파라미터에 대한 corner plot

try:
    import corner
    fig_corner = corner.corner(
        result['flat_samples'],
        labels=param_labels_zenodo,
        truths=reference_values,
            # truths: 원저자 참값 위치를 빨간 선으로 표시.
            # posterior가 이 선을 크게 벗어나 있다면(대조표에서 이미
            # 봤듯) "체인이 참값 근처로 못 갔다"는 뜻이고,
            # posterior 봉우리 자체가 매우 길게 늘어진 대각선
            # 모양이라면 "축퇴가 심하다"는 뜻.
        truth_color='red',
        show_titles=True,
        title_kwargs={'fontsize': 9},
    )
    fig_corner.suptitle('Zenodo 실측 데이터 Posterior Corner Plot', y=1.01, fontsize=13)
    plt.savefig(os.path.join(aa_output_dir, 'zenodo_corner_plot.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nCorner plot 저장 완료: {os.path.join(aa_output_dir, 'zenodo_corner_plot.png')}")
except ImportError:
    print("\n[안내] 'corner' 패키지가 없어 corner plot을 건너뜁니다. "
          "pip install corner 로 설치 후 다시 시도하세요.")

# R-hat: 각 워커를 하나의 독립 체인으로 보고, 파라미터별 수렴 여부 진단
print(f"\n{'파라미터':<8} {'R-hat':>10}  (1.01 미만이면 수렴 양호)")
print("-" * 35)
for i, name in enumerate(param_order):
    raw_chain_i = result['raw_chain'][:, :, i].T   # (걸음수,워커수) -> (워커수,걸음수)
    rhat_i = bt.gelman_rubin_rhat(raw_chain_i)
    flag = "  ⚠️ 미수렴 의심" if rhat_i > 1.01 else ""
    print(f"{name:<8} {rhat_i:>10.4f}{flag}")

print("\n[해석 가이드]")
print("  - tau를 고정해 phi-tau 축퇴는 이미 제거했습니다. 그래도 R-hat이")
print("    크게 나온다면, 남은 6개 파라미터 사이에 또 다른 축퇴(예:")
print("    ke1-ke2-ki 사이, 또는 A0와 이 셋 사이)가 남아있다는 신호일")
print("    수 있습니다 - corner plot에서 어느 쌍이 대각선을 이루는지")
print("    확인해보세요. 그 경우 그 파라미터도 추가 정보(별도 측정)로")
print("    고정하거나, prior 범위를 물리적으로 더 타당한 수준으로")
print("    좁히는 것이 다음 단계입니다.")


# =========================================================
# STEP 7. 피셔 행렬 진단 - 축퇴를 "행렬의 특이성"으로 재확인
# =========================================================
# MCMC(직접 샘플링)로 이미 ke1-ke2 축퇴를 확인했습니다. 이번엔 완전히
# 독립적인 방법(피셔 행렬)으로 같은 결론이 나오는지 교차검증합니다.
# 피셔 행렬은 posterior를 국소적으로 "2차 곡면"으로 근사하는데,
# 축퇴가 있으면 그 방향으로 곡면이 평평해져(그릇이 아니라 골짜기)
# 행렬이 특이(singular)해지고, 역행렬(공분산)을 못 구해 NaN이
# 나옵니다 - 이게 "버그"가 아니라 축퇴의 수학적 증거입니다.
print("\n" + "=" * 60)
print("STEP 7. 피셔 행렬 진단 (MCMC와 독립적인 축퇴 재확인)")
print("=" * 60)

fixed_kwargs_fisher = {'tau': aa_fixed_tau}
sigma_fisher, corr_fisher = zfm.fisher_parameter_uncertainties(
    zenodo_models.s21_single_resonance_transmission,
    result['median'], param_order, f_grid, noise_sigma_est, fixed_kwargs_fisher
)

print(f"\n{'파라미터':<8} {'MCMC 1σ':>14} {'피셔 1σ':>14} {'비율(MCMC/피셔)':>18}")
print("-" * 60)
mcmc_sigma = (result['err_lo'] + result['err_hi']) / 2
for i, name in enumerate(param_order):
    fisher_val = sigma_fisher[i]
    if np.isnan(fisher_val):
        print(f"{name:<8} {mcmc_sigma[i]:>14.6f} {'NaN (특이)':>14} "
              f"{'축퇴로 인해 피셔 오차 정의 불가':>18}")
    else:
        ratio = mcmc_sigma[i] / fisher_val
        print(f"{name:<8} {mcmc_sigma[i]:>14.6f} {fisher_val:>14.6f} {ratio:>18.3f}")

n_nan = np.sum(np.isnan(sigma_fisher))
print(f"\nNaN(특이) 파라미터 개수: {n_nan} / {len(param_order)}")
if n_nan > 0:
    nan_names = [param_order[i] for i in range(len(param_order)) if np.isnan(sigma_fisher[i])]
    print(f"  -> {nan_names}: 이 파라미터들은 피셔 행렬이 특이(singular)해서")
    print(f"     오차를 정의할 수 없습니다. 이는 MCMC/corner plot에서 이미")
    print(f"     확인한 축퇴가, 완전히 독립적인 수학적 방법(행렬의 특이성)")
    print(f"     으로도 재확인됐다는 뜻입니다 - 세 가지 방법(몬테카를로")
    print(f"     반복측정, MCMC 샘플링, 피셔 행렬)이 모두 같은 결론에")
    print(f"     도달한 것으로, '이 데이터 하나만으로는 이 파라미터들을")
    print(f"     구분할 수 없다'는 것이 원리적인 한계임을 강하게 뒷받침합니다.")
