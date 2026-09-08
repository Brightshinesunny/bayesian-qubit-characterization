"""
Full Flux-Sweep MCMC Batch Fitting (용어 주석 보강판)
=================================================================
101개 flux slice(자속 값) 전체에 대해 avoided-crossing S21 모델을
순차적으로 MCMC(Markov Chain Monte Carlo, 마르코프 연쇄 몬테카를로) 피팅합니다.

핵심 아이디어 4가지:
  1. flux(자속)를 하나씩 스캔하며 각 지점에서 독립적으로 피팅
  2. 인접 slice의 결과를 다음 slice의 초기값(warm start)으로 재사용
     -> 파라미터가 flux에 따라 연속적(smooth)으로 변한다는 물리적 사실을 이용해
        MCMC 수렴을 빠르고 안정적으로 만드는 표준 기법
        (중력파 분야로 치면, template bank를 순차 탐색할 때
         이전 최적점을 다음 탐색의 seed로 쓰는 것과 같은 발상)
  3. 피팅 실패(발산, 봉우리 탐색 실패 등)를 감지하고 로그로 남김
     -> Δ(디튜닝)가 큰, 즉 avoided crossing이 약해지는 flux 영역에서
        특히 잘 생기는 문제
  4. 최종적으로 g(Φ), κ(Φ), fr(Φ) 곡선을 뽑아 시각화
     (Φ: 그리스 문자 "파이", 자속(flux)을 나타내는 관례적 기호)

주의: 이 스크립트는 이전 단일-slice 파이프라인이 이미 검증된 상태를
전제로 합니다 (fr, g, kappa 모두 True 값과 일치 확인됨).
"""

import numpy as np            # 수치 배열 연산 라이브러리 (벡터/행렬 계산)
import matplotlib.pyplot as plt  # 그래프 시각화 라이브러리
import emcee                  # 베이지안 MCMC 샘플링 전용 라이브러리
                               # ("affine-invariant ensemble sampler"라는
                               #  알고리즘을 구현. 여러 개의 "walker"가
                               #  파라미터 공간을 동시에 탐색하며 수렴)
import h5py                   # HDF5(계층적 데이터 포맷) 파일을 읽고 쓰는 라이브러리
                               # 대용량 실험 데이터(다차원 배열+메타데이터)를
                               # 저장하는 실험물리 표준 포맷 중 하나
import os                     # 파일 경로, 디렉토리 생성 등 운영체제 기능
import warnings                # 경고 메시지 제어 (여기선 emcee의 반복 경고를 숨김용)

# =========================================================
# STEP 0. 분석자 설정값 (aa_ 접두사)
#   -> "분석하는 사람이 데이터를 보거나 물리적 판단을 근거로
#       직접 정해야 하는 값들"을 한곳에 모아둔 구역입니다.
#       실험/코드가 바뀌면 이 블록만 조정하면 되도록 설계했습니다.
# =========================================================

# --- 파일 경로 관련 ---
aa_h5_filename = 'qubit_2d_flux_sweep_mock.h5'   # 읽어올 HDF5 데이터 파일 이름
aa_drive_path  = '/content/drive/MyDrive/eunjunglee/SuperQuantum/'  # 구글 드라이브 상 폴더 경로
aa_output_dir  = '/content/drive/MyDrive/eunjunglee/SuperQuantum/outputs/'  # 결과(그래프, npz) 저장 폴더

# --- 사전 캘리브레이션 상수 ---
# 별도의 자속-주파수 스윕 실험(①번 식)에서 이미 구했다고 "가정"하는 값들.
# 지금 이 스크립트에서는 다시 추정하지 않고 고정 상수로 사용합니다.
aa_fq_max = 5.150   # GHz. 자속 Φ=0일 때(조셉슨 에너지 EJ 최대일 때)의 큐빗 주파수
aa_EC     = 0.250   # GHz. 충전 에너지(Charging Energy). 트랜스몬의 비조화성을 결정
aa_tau    = 0.12    # (시간 단위). 케이블/회로에서 생기는 신호 지연(cable delay).
                     # S21 위상에 e^{-i2πfτ} 형태로 곱해져 들어감 (④번 식 맨 끝 항)

# --- 잡음(noise) 수준 추정 방법 ---
# S21 스펙트�럼에서 "공진(resonance)과 멀리 떨어진, 신호가 거의 1인 구간"을
# 잡음만 있는 baseline(기준선)으로 보고, 그 구간의 표준편차를 잡음 크기로 씁니다.
aa_baseline_fraction = 0.15  # 전체 주파수 구간 중 양쪽 끝 각 15%씩을 baseline으로 사용

# --- Prior(사전분포) 범위 ---
# 베이지안 추론에서 "이 파라미터가 물리적으로 가질 수 있는 합리적 범위"를
# 미리 정해주는 것. 이 범위 밖으로 MCMC가 나가면 확률을 0(log_prob=-inf)으로 만들어 차단.
aa_prior_fr_min, aa_prior_fr_max = 4.8, 5.2         # GHz, 공진기 주파수 fr의 허용 범위
aa_prior_g_min,  aa_prior_g_max  = 0.001, 0.100     # GHz, 결합강도 g의 허용 범위
aa_prior_kappa_min, aa_prior_kappa_max = 0.001, 0.100  # GHz, 공진기 감쇠율 κ의 허용 범위

# --- 초기값(initial guess) 추정용 스무딩 ---
# 첫 slice에서 |S21| 곡선의 "딥(dip, 국소적으로 움푹 파인 지점 = 공진 위치)"을
# 자동으로 찾을 때, 잡음 때문에 생기는 가짜 딥을 줄이기 위해 이동평균으로
# 먼저 곡선을 매끈하게 만듭니다. 그 이동평균 윈도우(창) 크기.
aa_smoothing_window = 5   # 포인트 개수, 홀수 권장

# --- MCMC(마르코프 연쇄 몬테카를로) 설정 ---
aa_nwalkers = 32
    # "walker"란 emcee가 파라미터 공간(fr, g, κ의 3차원 공간)을
    # 동시에 탐색하는 여러 개의 "탐사 에이전트"입니다.
    # 32개의 워커가 각자 다른 위치에서 출발해 서로 정보를 주고받으며
    # 확률이 높은 영역(posterior가 큰 곳)으로 몰려갑니다.
aa_nsteps_per_slice = 800
    # 워커 하나가 몇 걸음(step)을 걸을지. 걸음 하나하나가 새로운 후보
    # 파라미터 값을 제안(propose)하고 받아들이거나(accept) 기각(reject)하는 과정.
    # 101개 slice를 반복하는 배치 작업이라, 단일 분석(2000~6000)보다 줄여서
    # 전체 실행시간을 관리 가능한 수준으로 낮췄습니다.
aa_init_scatter = 5e-3
    # 워커들을 초기 추정치 주변에 얼마나 "흩뿌릴지"(scatter) 정하는 비율.
    # 값이 너무 작으면 워커들이 서로 겹쳐 다양성이 부족해지고,
    # 너무 크면 prior 범위를 벗어나거나 수렴이 늦어질 수 있습니다.
aa_burn_in_discard = 200
    # "Burn-in"이란 MCMC 체인이 아직 초기값의 영향을 강하게 받아
    # 진짜 posterior 분포에 도달하지 못한 초반 구간을 말합니다.
    # 이 구간(첫 200 스텝)은 통계적으로 신뢰할 수 없어 버립니다(discard).
aa_thin_by = 10
    # "Thinning"이란 연속된 샘플끼리 서로 닮아있는(자기상관, autocorrelation)
    # 문제를 줄이기 위해 몇 개마다 하나씩만 골라 쓰는 것.
    # 10이면 10 스텝마다 1개씩만 최종 샘플로 채택.

aa_first_slice_index = 0   # flux 배열에서 스캔을 시작할 인덱스 (0 = 첫 번째 flux 값부터)

# --- 피팅 실패/불안정 판정 기준 ---
# posterior(사후분포)의 폭(불확실성)이 너무 크면 "제대로 못 찾았다"고 보고
# 자동으로 상태(status)를 표시하기 위한 임계값(threshold).
aa_failure_std_threshold_g = 0.02       # GHz, g의 posterior 폭이 이보다 크면 'unstable'
aa_failure_std_threshold_kappa = 0.02   # GHz, κ의 posterior 폭이 이보다 크면 'unstable'

os.makedirs(aa_output_dir, exist_ok=True)
    # 지정한 폴더가 없으면 새로 생성. exist_ok=True는 "이미 있으면 에러내지 말고 넘어가라"는 옵션.

print("=" * 60)
print("분석자 설정값(aa_*) 요약:")
for k, v in list(globals().items()):
    # globals(): 현재 코드에 정의된 모든 전역 변수를 딕셔너리로 반환하는 파이썬 내장 함수
    if k.startswith("aa_"):
        print(f"  {k:28s} = {v}")
print("=" * 60)

# =========================================================
# STEP 1. 데이터 로드
# =========================================================
full_path = os.path.join(aa_drive_path, aa_h5_filename)
    # os.path.join: 운영체제에 맞는 구분자(/)로 경로 문자열을 안전하게 이어붙이는 함수
if not os.path.exists(full_path):
    full_path = aa_h5_filename   # Drive 경로에 없으면 로컬(현재 작업 디렉토리)에서 시도 (fallback)

with h5py.File(full_path, 'r') as f:
    # 'r' = read-only 모드로 파일 열기. with 구문은 블록이 끝나면 자동으로 파일을 닫아줌.
    i_data_2d = f["data/I_voltage"][:]
        # I(In-phase): 복소수 신호의 실수부에 해당하는 전압 성분 (IQ 복조 방식의 표준 표기)
    q_data_2d = f["data/Q_voltage"][:]
        # Q(Quadrature): 복소수 신호의 허수부에 해당하는 전압 성분
    flux_data = f["data/flux_Phi0"][:]
        # 각 slice에 대응하는 자속 값 배열, 단위는 Φ0(자속 양자)로 정규화됨
    f_data    = f["data/frequency_GHz"][:]
        # 각 주파수 스캔 포인트 배열 (GHz)

    true_fr    = f["metadata"].attrs.get("f_r0_GHz", None)
        # HDF5의 "attribute"(속성): 배열이 아니라 파일에 붙은 메타데이터(설명값).
        # mock 데이터라 실제 참값을 알고 있으므로 검증용으로만 사용.
    true_g     = f["metadata"].attrs.get("g_GHz", None)
    true_kappa = f["metadata"].attrs.get("kappa_GHz", None)

real_s21_2d = i_data_2d + 1j * q_data_2d
    # I, Q 두 실수 배열을 하나의 복소수 배열로 결합.
    # 1j는 파이썬에서 허수단위(imaginary unit, 수학의 i)를 나타내는 표기법.
n_flux = len(flux_data)
print(f"\n총 flux slice 개수: {n_flux}")

# =========================================================
# STEP 2. 모델 / 우도 함수 정의
# =========================================================
def s21_anticrossing_model(f_grid, flux_val, f_r, g, kappa,
                            fq_max=aa_fq_max, EC=aa_EC, tau=aa_tau):
    """
    ①~④번 식을 그대로 구현한 물리 모델(forward model) 함수.
    "forward model"이란 파라미터(f_r, g, kappa 등)를 입력하면
    측정될 것으로 예상되는 신호(S21)를 계산해내는 방향의 함수입니다.
    (반대로 "역문제, inverse problem"는 측정된 S21로부터
     파라미터를 역산하는 것 - 이게 바로 우리가 MCMC로 하려는 일)

    Parameters
    ----------
    f_grid : array. 주파수 스캔 값들 (GHz)
    flux_val : float. 이 slice의 자속 값 (Φ/Φ0)
    f_r : float. 공진기(bare resonator) 고유 주파수
    g : float. 큐빗-공진기 결합강도(coupling strength)
    kappa : float. 공진기 감쇠율(decay rate, 선폭)
    """
    # ①번 식: 자속에 따른 큐빗 주파수 (SQUID 구조의 유효 조셉슨 에너지 변화 반영)
    fq = (fq_max + EC) * np.sqrt(np.abs(np.cos(np.pi * flux_val))) - EC

    # 디튜닝(detuning): 큐빗과 공진기 주파수의 차이
    delta = fq - f_r

    # ②번 식: 결합에 의해 갈라지는 두 혼성 모드(hybridized eigenmodes) 주파수
    hybrid_plus  = 0.5 * (f_r + fq + np.sqrt(delta**2 + 4 * g**2))
    hybrid_minus = 0.5 * (f_r + fq - np.sqrt(delta**2 + 4 * g**2))

    # ③번 식: 믹싱 각도(mixing angle) θ의 sin^2, cos^2 값
    # = 각 혼성 모드가 큐빗/공진기 성분을 얼마나 섞고 있는지를 나타내는 가중치
    # (Hopfield 계수라고도 부름)
    sin2_theta = 0.5 * (1.0 - delta / np.sqrt(delta**2 + 4 * g**2))
    cos2_theta = 1.0 - sin2_theta

    # ④번 식: 두 개의 로렌츠 함수(Lorentzian) 합으로 S21 스펙트럼을 모델링.
    # 로렌츠 함수는 공진 현상을 나타내는 표준적인 종 모양(peak/dip) 곡선.
    s21_mode1 = cos2_theta / (1.0 + 1j * (f_grid - hybrid_minus) / (kappa / 2.0))
    s21_mode2 = sin2_theta / (1.0 + 1j * (f_grid - hybrid_plus)  / (kappa / 2.0))

    # 케이블 지연으로 인한 위상 회전 (크기(magnitude)에는 영향 없고 위상만 회전시킴)
    cable_delay = np.exp(-1j * 2 * np.pi * f_grid * tau)

    return (1.0 - (s21_mode1 + s21_mode2)) * cable_delay


def log_prior(theta):
    """
    Prior(사전분포)의 로그값을 반환.
    베이지안 통계에서 prior란 "데이터를 보기 전에 이미 알고 있는/가정하는
    파라미터에 대한 믿음"을 확률분포로 표현한 것.
    여기서는 "이 범위 안이면 모두 동등하게 가능하다"는 균일분포(flat/uniform prior)를 사용.
    범위 밖이면 -inf(로그 확률 -무한대, 즉 확률 0)를 반환해 MCMC가 그 영역을 버리게 함.
    """
    f_r, g, kappa = theta   # theta: 파라미터 벡터를 관례적으로 부르는 그리스 문자 이름
    if (aa_prior_fr_min < f_r < aa_prior_fr_max and
        aa_prior_g_min  < g  < aa_prior_g_max  and
        aa_prior_kappa_min < kappa < aa_prior_kappa_max):
        return 0.0   # 균일분포이므로 범위 안에서는 로그확률이 상수(여기선 0으로 정규화)
    return -np.inf


def log_likelihood(theta, f_grid, flux_val, data_1d, sigma):
    """
    Likelihood(우도)의 로그값을 반환.
    "이 파라미터(theta)가 맞다고 가정했을 때, 지금 관측된 데이터가
    나올 확률이 얼마나 되는가"를 나타내는 함수.
    측정 잡음이 가우시안(정규분포)이라고 가정하면, 로그우도는
    -0.5 * chi^2 (카이제곱, 잔차 제곱합을 잡음 분산으로 정규화한 값) 형태가 됨.
    """
    f_r, g, kappa = theta
    model = s21_anticrossing_model(f_grid, flux_val, f_r, g, kappa)

    # residual(잔차): 실제 데이터와 모델 예측값의 차이
    real_res = np.real(data_1d - model)
    imag_res = np.imag(data_1d - model)

    # chi-square(카이제곱): 잔차를 잡음 크기(sigma)로 나눠 제곱해서 합한 값.
    # 모델이 데이터를 잘 설명할수록 이 값이 작아짐.
    chi2 = np.sum((real_res / sigma) ** 2 + (imag_res / sigma) ** 2)
    return -0.5 * chi2


def log_probability(theta, f_grid, flux_val, data_1d, sigma):
    """
    Posterior(사후분포)의 로그값 = log_prior + log_likelihood.
    베이즈 정리(Bayes' theorem): posterior ∝ prior × likelihood
    로그를 취하면 곱셈이 덧셈이 되어 계산이 편해짐.
    emcee는 바로 이 함수의 반환값을 최대화하는 방향으로 walker들을 이동시킴.
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):   # np.isfinite: 값이 무한대(inf)나 NaN이 아닌지 확인
        return -np.inf
    return lp + log_likelihood(theta, f_grid, flux_val, data_1d, sigma)


def moving_average(x, w):
    """이동평균(moving average) 스무딩 함수. w는 평균을 낼 윈도우(창) 크기."""
    if w <= 1:
        return x
    kernel = np.ones(w) / w
        # "커널(kernel)": 신호처리에서 convolution(합성곱) 연산에 쓰이는 가중치 배열.
        # 여기서는 모든 값이 1/w로 동일한 "균일 평균 커널".
    return np.convolve(x, kernel, mode='same')
        # convolve: 커널을 신호 위에서 미끄러뜨리며 각 위치의 가중합을 계산하는 연산.
        # mode='same'은 출력 길이를 입력과 동일하게 맞추는 옵션.


def estimate_noise_sigma(s21_1d, baseline_fraction):
    """
    측정 잡음의 표준편차(sigma)를 데이터에서 직접 추정.
    공진에서 먼 양 끝 구간(baseline)은 신호가 거의 변하지 않으므로,
    그 구간의 흔들림(분산)을 순수한 잡음으로 간주.
    """
    n_pts = len(s21_1d)
    n_edge = max(int(n_pts * baseline_fraction), 5)
    baseline_region = np.concatenate([s21_1d[:n_edge], s21_1d[-n_edge:]])
        # np.concatenate: 여러 배열을 하나로 이어붙이는 함수
    return np.mean([np.std(np.real(baseline_region)), np.std(np.imag(baseline_region))])
        # np.std: 표준편차(standard deviation) 계산 함수


def estimate_initial_guess_from_data(f_grid, s21_1d, smoothing_window,
                                      prior_fr_min, prior_fr_max):
    """
    첫 slice에서만 사용: |S21| 곡선에서 봉우리(딥, dip)를 자동 탐색해
    fr, g, kappa의 초기 추정치를 데이터로부터 직접 뽑아냄.
    (사람이 그래프를 눈으로 보고 "여기가 공진점이네" 하고 어림잡는 것을
     코드로 자동화한 버전)
    """
    mag = np.abs(s21_1d)   # np.abs: 복소수의 절댓값(=크기, magnitude)을 계산
    mag_smooth = moving_average(mag, smoothing_window)

    # 좌우 이웃보다 값이 작은(움푹 파인) 지점들을 "딥 후보"로 수집
    dip_candidates = []
    for i in range(2, len(mag_smooth) - 2):
        if mag_smooth[i] < mag_smooth[i - 1] and mag_smooth[i] < mag_smooth[i + 1]:
            dip_candidates.append(i)

    if len(dip_candidates) >= 2:
        # 딥이 2개 이상이면(avoided crossing이 뚜렷이 갈라진 경우),
        # 가장 깊은 두 개를 골라 두 혼성 모드로 간주
        dip_candidates.sort(key=lambda i: mag_smooth[i])
            # sort(key=...): 리스트를 특정 기준(여기선 mag_smooth 값)으로 정렬
        two_deepest = sorted(dip_candidates[:2])
        f_dip1, f_dip2 = f_grid[two_deepest[0]], f_grid[two_deepest[1]]
        fr_guess = 0.5 * (f_dip1 + f_dip2)     # 두 딥의 중간점을 fr 초기값으로
        g_guess = abs(f_dip2 - f_dip1) / 2.0    # 딥 간격의 절반을 g 초기값으로
        main_dip_idx = two_deepest[np.argmin([mag_smooth[two_deepest[0]], mag_smooth[two_deepest[1]]])]
            # np.argmin: 배열에서 최솟값의 위치(인덱스)를 반환하는 함수
    elif len(dip_candidates) == 1:
        # 딥이 하나만 보이면(결합이 약하거나 두 봉우리가 뭉개진 경우)
        fr_guess = f_grid[dip_candidates[0]]
        g_guess = 0.005   # 보수적인(작은) 초기값으로 시작
        main_dip_idx = dip_candidates[0]
    else:
        # 딥 탐색 자체가 실패하면 prior 범위의 중앙값으로 폴백(fallback, 대체 수단)
        fr_guess = 0.5 * (prior_fr_min + prior_fr_max)
        g_guess = 0.01
        main_dip_idx = np.argmin(mag_smooth)

    # kappa 초기값: 딥의 "절반 깊이 폭"(FWHM과 유사한 개념)을 측정해 어림잡음
    dip_depth = 1.0 - mag_smooth[main_dip_idx]
    half_level = 1.0 - dip_depth / 2.0
    left_idx = main_dip_idx
    while left_idx > 0 and mag_smooth[left_idx] < half_level:
        left_idx -= 1
    right_idx = main_dip_idx
    while right_idx < len(mag_smooth) - 1 and mag_smooth[right_idx] < half_level:
        right_idx += 1
    kappa_guess = max(f_grid[right_idx] - f_grid[left_idx], 0.005)

    return np.array([fr_guess, g_guess, kappa_guess])

# =========================================================
# STEP 3. 배치 루프: 모든 flux slice에 대해 순차 피팅 (warm start)
# =========================================================
results = {
    'flux': [],
    'fr': [], 'fr_err_lo': [], 'fr_err_hi': [],
    'g': [], 'g_err_lo': [], 'g_err_hi': [],
    'kappa': [], 'kappa_err_lo': [], 'kappa_err_hi': [],
    'status': [],   # 각 slice의 피팅 상태: 'ok' / 'unstable' / 'failed'
}

current_guess = None   # warm-start용 변수: 직전 slice의 피팅 결과를 저장해두는 곳

order = list(range(n_flux))   # 0번 slice부터 마지막 slice까지 순서대로 처리

print(f"\n총 {n_flux}개 slice에 대해 순차 배치 피팅을 시작합니다 "
      f"(slice당 {aa_nsteps_per_slice} step)...\n")

for count, idx in enumerate(order):
    # enumerate: (순번, 값) 쌍을 동시에 꺼내주는 파이썬 내장 함수
    flux_val = flux_data[idx]
    s21_1d = real_s21_2d[idx, :]   # 2D 배열에서 idx번째 행(row) 전체를 슬라이싱
    noise_sigma_est = estimate_noise_sigma(s21_1d, aa_baseline_fraction)

    # 초기값 결정: 첫 slice는 데이터에서 직접 추정, 이후는 warm-start(이전 결과 재사용)
    if current_guess is None:
        initial_guess = estimate_initial_guess_from_data(
            f_data, s21_1d, aa_smoothing_window, aa_prior_fr_min, aa_prior_fr_max
        )
    else:
        initial_guess = current_guess.copy()

    ndim = 3   # 추정할 파라미터 개수 (fr, g, kappa)
    pos = initial_guess + aa_init_scatter * np.abs(initial_guess) * np.random.randn(aa_nwalkers, ndim)
        # np.random.randn: 표준정규분포(평균0, 표준편차1)에서 난수를 뽑는 함수.
        # 32개 워커를 초기 추정치 주변에 살짝씩 무작위로 흩뿌려 시작점을 만듦.

    # prior 범위를 벗어난 워커를 경계값으로 강제 이동(clip)시켜, 시작부터
    # log_prior=-inf가 되어버리는(=워커가 죽어버리는) 상황을 방지
    pos[:, 0] = np.clip(pos[:, 0], aa_prior_fr_min + 1e-4, aa_prior_fr_max - 1e-4)
    pos[:, 1] = np.clip(pos[:, 1], aa_prior_g_min + 1e-4, aa_prior_g_max - 1e-4)
    pos[:, 2] = np.clip(pos[:, 2], aa_prior_kappa_min + 1e-4, aa_prior_kappa_max - 1e-4)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # emcee가 반복적으로 내는 수렴 경고를 배치 처리 중에는 숨김
        sampler = emcee.EnsembleSampler(
            aa_nwalkers, ndim, log_probability,
            args=(f_data, flux_val, s21_1d, noise_sigma_est)
        )
        sampler.run_mcmc(pos, aa_nsteps_per_slice, progress=False)

    flat_samples = sampler.get_chain(discard=aa_burn_in_discard, thin=aa_thin_by, flat=True)
        # get_chain: MCMC로 뽑힌 모든 샘플(파라미터 후보값들)을 가져옴.
        # discard: burn-in 구간 제거, thin: thinning 간격 적용
        # flat=True: (걸음수, 워커수, 파라미터수) 3차원 배열을
        #            (샘플수, 파라미터수) 2차원으로 평평하게(flatten) 펼침

    status = 'ok'
    if flat_samples.shape[0] < 10:
        # 샘플이 거의 안 남았다는 건 대부분의 워커가 prior 밖으로 나가 죽었다는 뜻
        status = 'failed'
        med = initial_guess
        lo = np.zeros(3)
        hi = np.zeros(3)
    else:
        pct = np.percentile(flat_samples, [16, 50, 84], axis=0)
            # np.percentile: 데이터의 특정 백분위수 값을 계산.
            # 16/50/84 백분위수는 가우시안 분포의 "평균 ± 1 표준편차"에 해당하는
            # 관례적인 신뢰구간 표기 방식 (중앙값 및 비대칭 오차 표현에 자주 사용)
        med = pct[1]                 # 중앙값(median) = 대표 추정치
        lo = pct[1] - pct[0]         # 하한 오차(median - 16th percentile)
        hi = pct[2] - pct[1]         # 상한 오차(84th percentile - median)

        # posterior 폭이 임계값보다 크면 "불안정"(추정 신뢰도가 낮음)으로 표시
        if hi[1] > aa_failure_std_threshold_g or hi[2] > aa_failure_std_threshold_kappa:
            status = 'unstable'

    # 결과를 딕셔너리에 차곡차곡 기록
    results['flux'].append(flux_val)
    results['fr'].append(med[0]); results['fr_err_lo'].append(lo[0]); results['fr_err_hi'].append(hi[0])
    results['g'].append(med[1]); results['g_err_lo'].append(lo[1]); results['g_err_hi'].append(hi[1])
    results['kappa'].append(med[2]); results['kappa_err_lo'].append(lo[2]); results['kappa_err_hi'].append(hi[2])
    results['status'].append(status)

    # 다음 slice의 warm-start 초기값 갱신 (실패하지 않았을 때만 갱신,
    # 실패한 결과를 다음 시작점으로 물려주면 연쇄적으로 계속 실패할 수 있기 때문)
    if status != 'failed':
        current_guess = med

    if (count + 1) % 10 == 0 or count == 0:
        # % (나머지 연산자): 10으로 나눈 나머지가 0일 때, 즉 10번째마다 진행상황 출력
        print(f"  [{count+1:3d}/{n_flux}] flux={flux_val:+.3f}  "
              f"fr={med[0]:.4f}  g={med[1]:.4f}  kappa={med[2]:.4f}  status={status}")

n_ok = results['status'].count('ok')
n_unstable = results['status'].count('unstable')
n_failed = results['status'].count('failed')
print(f"\n완료: ok={n_ok}, unstable={n_unstable}, failed={n_failed} (전체 {n_flux})")

# =========================================================
# STEP 4. 결과 정리 및 저장
# =========================================================
for k in results:
    results[k] = np.array(results[k]) if k != 'status' else np.array(results[k], dtype=object)
        # dtype=object: 문자열('ok' 등)처럼 숫자가 아닌 값을 담기 위한 배열 타입 지정

np.savez(
    os.path.join(aa_output_dir, 'flux_sweep_fit_results.npz'),
    **{k: v for k, v in results.items()}
        # np.savez: 여러 배열을 하나의 .npz(압축 아카이브) 파일로 저장하는 함수.
        # **딕셔너리 형태로 넘기면 각 키가 저장된 배열의 이름이 됨.
)
print(f"\n결과 저장 완료: {os.path.join(aa_output_dir, 'flux_sweep_fit_results.npz')}")

# =========================================================
# STEP 5. 시각화: g(Φ), kappa(Φ), fr(Φ) + 상태별 색상 구분
# =========================================================
status_colors = {'ok': 'tab:blue', 'unstable': 'tab:orange', 'failed': 'tab:red'}

fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    # subplots(3,1): 세로로 3개의 그래프 패널을 만듦. sharex=True는 x축(flux)을 공유.

for status_name, color in status_colors.items():
    mask = results['status'] == status_name
        # mask: 조건을 만족하는 위치만 True인 불리언(참/거짓) 배열.
        # 이걸로 특정 status에 해당하는 데이터만 골라낼 수 있음 (불리언 인덱싱)
    if not np.any(mask):
        # np.any: 배열 안에 True가 하나라도 있는지 확인
        continue   # 해당 status의 데이터가 하나도 없으면 이번 루프는 건너뜀
    axes[0].errorbar(results['flux'][mask], results['fr'][mask],
                      yerr=[results['fr_err_lo'][mask], results['fr_err_hi'][mask]],
                      fmt='o', ms=3, color=color, label=status_name, alpha=0.7)
        # errorbar: 데이터 점과 함께 오차막대(error bar)를 같이 그리는 함수.
        # yerr에 [하한오차, 상한오차] 리스트를 넣으면 비대칭 오차막대가 그려짐.
    axes[1].errorbar(results['flux'][mask], results['g'][mask],
                      yerr=[results['g_err_lo'][mask], results['g_err_hi'][mask]],
                      fmt='o', ms=3, color=color, alpha=0.7)
    axes[2].errorbar(results['flux'][mask], results['kappa'][mask],
                      yerr=[results['kappa_err_lo'][mask], results['kappa_err_hi'][mask]],
                      fmt='o', ms=3, color=color, alpha=0.7)

if true_fr is not None:
    axes[0].axhline(true_fr, color='gray', ls='--', lw=1, label='true (Φ=0 reference)')
        # axhline: 그래프에 수평 기준선(horizontal line)을 그리는 함수
if true_g is not None:
    axes[1].axhline(true_g, color='gray', ls='--', lw=1)
if true_kappa is not None:
    axes[2].axhline(true_kappa, color='gray', ls='--', lw=1)

axes[0].set_ylabel(r'$f_r$ (GHz)')
axes[1].set_ylabel(r'$g$ (GHz)')
axes[2].set_ylabel(r'$\kappa$ (GHz)')
axes[2].set_xlabel(r'Flux $\Phi/\Phi_0$')
axes[0].legend(loc='upper right', fontsize=9)
axes[0].set_title('Full Flux-Sweep MCMC Parameter Recovery (Warm-Start Batch Fit)')

plt.tight_layout()   # 서브플롯 사이 여백을 자동으로 보기 좋게 조정
plt.savefig(os.path.join(aa_output_dir, 'full_flux_sweep_result.png'), dpi=150, bbox_inches='tight')
    # dpi: 이미지 해상도(dots per inch). bbox_inches='tight': 여백을 딱 맞게 잘라 저장
plt.show()

print("\n시각화 저장 완료:", os.path.join(aa_output_dir, 'full_flux_sweep_result.png'))
