"""
stress_test.py — Fano / TLS / baseline-tilt 주입 스트레스 테스트
=================================================================

[이 스크립트가 하는 일]
참값이 알려진 합성 데이터에 계통 효과(systematic effect)를 주입하고,
그 효과를 포함하지 않는 표준 모델로 피팅한다. 그러면 "모델 오설정이
파라미터 추정을 얼마나 틀리게 하는가"를 백분율로 말할 수 있다.

실측 데이터에는 참값이 없으므로, 피팅이 잘 되어 보여도 그것이
모델이 옳아서인지 틀렸는데도 그렇게 보이는 것인지 구분할 수 없다.
주입 시험(injection test)이 그 구분을 가능하게 한다.


=================================================================
물리 용어
=================================================================
반교차 (avoided crossing)
    큐빗과 공진기가 결합강도 g 로 상호작용하면, 두 계의 고유주파수가
    접근할 때 직접 교차하지 않고 서로 밀어낸다. 양자역학의 준위 반발
    (level repulsion)이며 Jaynes-Cummings 모형의 결과.

디튜닝 (detuning, delta)
    delta = f_q - f_r. 큐빗과 공진기 주파수의 차이. 0이면 정확히 공명.

혼성 모드 (hybrid mode)
    결합에 의해 생기는 새로운 두 고유주파수.
        f± = (f_r + f_q)/2 ± sqrt(delta² + 4g²)/2
    delta=0 에서 간격이 최소 2g 가 된다. g 를 직접 측정하는 표준 방법.

호프필드 계수 (Hopfield coefficient, 믹싱 각도)
    각 혼성 모드가 원래 큐빗/공진기 성분을 얼마나 섞고 있는지.
        sin²θ = (1 - delta/sqrt(delta²+4g²))/2,   cos²θ = 1 - sin²θ
    delta=0 이면 0.5/0.5 로 균등. |delta|>>g 이면 한쪽으로 완전히 치우침.
    이 계수가 두 딥의 상대적 깊이를 결정한다.

분산 영역 (dispersive regime)
    |delta| >> g 인 영역. 두 계가 에너지를 직접 주고받지 않고 서로의
    주파수만 살짝 밀어주는 상태. 큐빗 판독에는 쓰지만 g 측정에는 최악.

kappa (감쇠율, decay rate)
    공진기가 에너지를 잃는 속도. 로렌츠 곡선의 반치반폭(HWHM)에 대응.
    품질계수와의 관계: Q_l = f_r / kappa_total,  1/Q_l = 1/Q_i + 1/Q_c
    -> kappa 의 오차는 Q 계열 보고값에 직접 전파된다.

Fano 공명
    이상적 로렌츠 공진은 "공진기로 들어가는 경로가 하나"라는 가정 위에
    있다. 실제 배선에는 공진기를 거치지 않는 직접 경로가 섞이고, 두
    경로가 간섭하면 선형이 좌우 비대칭으로 일그러진다. 원자물리학에서
    이산 상태와 연속 상태의 간섭을 설명하려 도입된 개념.
    표준 구현은 결합 Q 를 복소수로 두는 것(Khalil 2012, Probst 2015).

TLS (Two-Level System, 이준위계)
    초전도 칩 표면이나 유전체 층의 미세 결함이 작은 큐빗처럼 행동하며
    특정 주파수에서 공진기와 약하게 결합한다. 이론 모델에 없는 좁은
    추가 딥을 만든다.

트랜스몬의 자속 의존성
    f_q(Φ) = (f_q_max + E_C)·sqrt(|cos(πΦ)|) - E_C
    E_C 는 충전 에너지(charging energy).


=================================================================
통계 용어
=================================================================
베이즈 추론
    posterior ∝ likelihood × prior
    사후분포 ∝ 우도 × 사전분포

prior (사전분포)
    데이터를 보기 전의 파라미터에 대한 믿음. 여기서는 균등 분포.
    [중요] 추정값이 prior 경계에 붙으면 그 값은 "데이터가 말한 값"이
    아니라 "탐색 범위의 끝"이다. 그 상태의 오차는 바이어스의 크기로
    해석할 수 없다.

우도 (likelihood)
    주어진 파라미터에서 이 데이터가 나올 확률. 가우스 우도는 잡음
    sigma 를 요구하는데, sigma 를 과대평가하면 우도가 넓어져서
    잔차가 커도 통계적으로 허용된다 -> 오설정을 숨기는 방향.

MCMC / walker / 앙상블 샘플러
    emcee 는 여러 walker 가 서로의 위치를 참조하며 사후분포를 탐색한다.

자기상관시간 (autocorrelation time, tau)
    체인이 독립적인 표본 하나를 만드는 데 걸리는 스텝 수.
    유효표본수 N_eff ≈ N/tau.
    emcee 표준 기준: N > 50·tau  <=>  tau < N/50
    (유효표본 50개 이상 확보하라는 뜻)

burn-in / thinning
    burn-in: 초기 과도 구간을 버림. thinning: 상관된 표본을 솎아냄.

MAP vs median
    MAP(Maximum A Posteriori)은 사후분포가 최대인 점, 즉 실재하는 표본.
    median 은 성분별 중앙값이라, 사후분포가 비대칭이거나 파라미터 간
    상관이 강하면 어느 표본과도 일치하지 않는 점이 될 수 있다.

식별 가능성 (identifiability) / 축퇴 (degeneracy)
    데이터가 파라미터를 유일하게 결정할 수 있는가.
    관측량보다 미지수가 많거나 여러 조합이 같은 예측을 내면 축퇴한다.
    축퇴하면 피셔 행렬이 특이(singular)해져 역행렬이 없고, MCMC 는
    prior 경계까지 흘러간다.

계통 바이어스 vs 통계 오차
    통계 오차는 잡음에서 오며 1/SNR 로 줄어든다.
    계통 바이어스는 모델 오설정에서 오며 SNR 에 의존하지 않는다.
    -> 측정이 정밀해질수록 계통 바이어스가 지배적이 된다.


=================================================================
원본 대비 변경 이력
=================================================================
  [FIX-1]  prior 상한 확대 (0.1 -> 0.5)
           원본에서 g=0.09623, kappa=0.10000 으로 상한 0.1 에 붙어
           있었다. 즉 140%·233% 라는 숫자는 바이어스가 아니라 상자의 벽.
  [FIX-2]  walker 초기 산포를 파라미터별 상대값으로
           g_guess=0.0015 인데 산포가 0.005 라 walker 절반이 하한
           0.001 밖에서 시작, log_prob=-inf 로 못 움직였다.
  [FIX-3]  딥 탐색을 detrend + prominence + distance 기준으로 교체
           원본은 절대 깊이 순 정렬이라, tilt 로 낮아진 구간의 잡음
           기복이 상위를 차지했다.
  [FIX-4]  MCMC 시드 고정 (재현성)
  [FIX-5]  median 대신 MAP 으로 최적 곡선 표시
  [FIX-6]  prior 경계 접촉 자동 검사
  [FIX-7]  tau vs N/50 수렴 기준 명시적 검사 + nsteps 확대
  [FIX-8]  moving average 길이를 입력과 동일하게 고정
  [FIX-9]  반교차 flux 지점 자동 탐색
  [FIX-10] 시나리오 개별 스윕 — 각 효과의 기여를 분리
  [FIX-11] 그림에 ideal / distorted / fit 세 곡선 모두 표시

전제: models_v2.py 를 쓴다. v1 의 fano_lineshape_correction 은
      /np.max 정규화로 신호 전체를 1/(1+q²) 배로 눌렀다(q=2 에서 1/5).
      베이스라인 1을 전제하는 표준 모델로는 어떤 파라미터로도 맞출 수
      없는 상태였고, 이것이 최초 시험의 140%/233% 오차의 원인이었다.
"""

import os
import inspect
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import models_v2 as models
import mock_data
import likelihood
import mcmc_pipeline
import diagnostics


# =========================================================
# STEP 0. 설정
# =========================================================
aa_output_dir = './outputs/'
os.makedirs(aa_output_dir, exist_ok=True)

# [FIX-4] 재현성. 잡음 실현과 MCMC 초기화를 모두 고정한다.
# 단일 시드는 한 실현일 뿐이므로, 결과를 인용하기 전에 5~10개로
# 반복해서 오차가 시드 산포 안에 묻히지 않는지 확인해야 한다.
aa_seed = 7

# --- 참값 ---
# 이 세 개가 추정 대상이고, 나머지는 사전 캘리브레이션 상수로 고정한다.
aa_f_r0, aa_g, aa_kappa = 5.000, 0.040, 0.030   # GHz
aa_fq_max, aa_EC, aa_tau_transmon = 5.150, 0.250, 0.12

aa_n_freq = 401                  # 격자 401점 -> 분해능 1 MHz
aa_freq_range = (4.8, 5.2)       # GHz

# 백색잡음(white noise). 잡음이 전혀 없는 측정은 비현실적이므로 항상 켠다.
aa_white_noise_level = 0.015


# --- 시나리오 스위치 (aa_run_sweep=False 일 때만 쓰임) ---
aa_enable_fano = True
aa_fano_q = 2.0            # Fano 비대칭 계수. 클수록 대칭에 가깝다.
                           # phi = -arctan(1/q) 로 위상에 들어간다.

aa_enable_tls = True
aa_tls_offset_from_fr = 0.005   # TLS 를 f_r 에서 얼마나 떨어뜨릴지 (GHz)
                                # 작을수록 진짜 반교차 봉우리와 가까워져
                                # 파이프라인이 헷갈리기 쉽다.
aa_tls_coupling = 0.4      # 딥의 깊이. baseline 대비 40%.
                           # [NOTE] 실제 소자에서 흔한 값은 0.02~0.15.
                           # 0.4 는 "현실적 지저분함"보다 "최악 조건"에
                           # 가깝다. 현실적 범위도 함께 시험할 것.
aa_tls_linewidth = 0.006   # 딥의 폭. 진짜 결합(2g=0.08)보다 훨씬 좁다.

aa_enable_tilt = True
aa_tilt_slope = 0.15       # 배선의 주파수 의존 손실 등에 의한 기울기


# --- 실행 모드 ---
aa_run_sweep = True        # [FIX-10] True: 효과를 하나씩 켜며 기여 분리.
                           # 동시에 켠 상태만 보면 상쇄로 개별 크기를
                           # 과소평가한다. 실제로 Fano(-7.3%)와
                           # TLS(+8.1%)가 ALL 에서 +3.3% 로 보였다.
aa_auto_find_flux = True   # [FIX-9] True: 반교차 flux 를 스캔해 자동 선택
aa_case_flux_val = 0.0     # auto_find_flux=False 일 때만 사용


# --- MCMC 설정 ---
aa_smoothing_window = 5
aa_nwalkers = 32
# [FIX-7] tau 기준 N > 50·tau 를 만족시키려면 충분히 길어야 한다.
# 8000 스텝에서 ALL 의 kappa 가 tau=427 -> 50·tau≈21300 필요했다.
# 25000 으로 올렸으나 ALL 은 여전히 tau=643 (N/50=500)으로 미달.
# tau 가 체인 길이와 함께 커진 것은, 긴 체인에서야 느린 모드가
# 드러났다는 신호다. 40000 으로 재시험 필요.
aa_nsteps = 25000
aa_burn_in_discard = 5000
aa_thin_by = 10

# [FIX-2] 파라미터별 상대 산포.
# 스칼라 산포를 쓰면 값의 크기가 다른 파라미터에서 문제가 생긴다.
# f_r~5.0 과 g~0.04 는 세 자릿수 차이가 난다.
aa_init_scatter_rel = 0.05     # 초기값의 5%
aa_init_scatter_floor = 1e-4   # 너무 작아지지 않게

# [FIX-1] prior 상한 확대.
# 원본은 g, kappa 모두 (0.001, 0.1) 이었고 추정값이 0.09623 / 0.10000
# 으로 벽에 닿아 있었다. 상자를 넓혀야 "데이터가 어디서 멈추는가"를
# 볼 수 있다.
aa_prior_bounds = {
    'f_r':   (4.80, 5.20),
    'g':     (0.001, 0.50),
    'kappa': (0.001, 0.50),
}

PARAM_NAMES = ['f_r', 'g', 'kappa']
TRUE_THETA = np.array([aa_f_r0, aa_g, aa_kappa])


# =========================================================
# 유틸
# =========================================================
def moving_average_same(x, window):
    """[FIX-8] 입력과 같은 길이를 반환하는 이동평균.

    diagnostics.moving_average 가 mode='valid' 로 길이를 줄여 반환하면
    f_grid 로 인덱싱할 때 window//2 만큼 주파수가 어긋난다. 5점 평활에서
    2칸(=2 MHz) 어긋나므로 딥 위치 추정에 직접 영향을 준다.
    edge padding 으로 길이를 맞춘다.
    """
    if window < 2:
        return np.asarray(x, dtype=float)
    k = int(window)
    pad = k // 2
    xp = np.pad(np.asarray(x, dtype=float), (pad, k - 1 - pad), mode='edge')
    return np.convolve(xp, np.ones(k) / k, mode='valid')


def detrended_magnitude(f_grid, s21_1d, window):
    """baseline tilt 를 1차 다항식으로 제거한 |S21|.

    [왜 필요한가]
    tilt 가 켜져 있으면 "절대적으로 가장 낮은 점"이 공진이 아니라
    baseline 이 낮은 구간이 된다. 딥 탐색 전에 반드시 제거해야
    공진 구조를 찾을 수 있다.

    Returns: (평활된 |S21|, 추정 baseline, detrend 된 |S21|)
    """
    mag = np.abs(s21_1d)
    mag_s = moving_average_same(mag, window)
    coef = np.polyfit(f_grid, mag_s, 1)      # 1차 = 직선 baseline
    baseline = np.polyval(coef, f_grid)
    return mag_s, baseline, mag_s - baseline


def find_dips(f_grid, s21_1d, window,
              prominence_frac=0.15, min_separation_ghz=0.010, verbose=True):
    """[FIX-3] detrend + prominence + 최소간격 기준 딥 탐색.

    [원본의 문제]
    단순 극소점을 모두 모아 '절대 깊이' 순으로 정렬했다. tilt 로
    낮아진 구간의 잡음 기복(깊이 0.012~0.019)이 상위를 차지했고,
    3 MHz 떨어진 이웃 두 점이 선택되어 g_guess=0.0015 가 나왔다.
    참값 0.04 의 1/27 이다. 격자 간격이 1 MHz 이므로 3칸 떨어진
    극소점 두 개는 물리 구조가 아니라 잡음이다.

    [해결]
    prominence(돌출도)는 "주변 대비 얼마나 두드러지는가"를 잰다.
    절대 깊이가 아니라 상대 돌출도로 정렬하면 잡음 기복이 걸러진다.
    distance 는 최소 간격을 강제해 이웃 픽셀 쌍을 배제한다.
    """
    mag_s, baseline, mag_d = detrended_magnitude(f_grid, s21_1d, window)

    df = float(np.median(np.diff(f_grid)))               # 격자 간격
    distance = max(1, int(round(min_separation_ghz / df)))
    span = float(np.ptp(mag_d))                          # peak-to-peak
    prominence = max(prominence_frac * span, 1e-6)

    # find_peaks 는 봉우리를 찾으므로 부호를 뒤집어 딥을 찾는다
    idx, props = find_peaks(-mag_d, prominence=prominence, distance=distance)

    order = np.argsort(props['prominences'])[::-1]       # 돌출도 큰 순
    idx_sorted = idx[order]
    prom_sorted = props['prominences'][order]

    if verbose:
        print(f"  [딥 탐색] prominence>={prominence:.4f}, "
              f"최소간격={min_separation_ghz*1000:.0f} MHz ({distance} pts)")
        if len(idx_sorted) == 0:
            print("  [딥 탐색] 조건을 만족하는 딥 없음")
        for i, p in list(zip(idx_sorted, prom_sorted))[:5]:
            print(f"    f={f_grid[i]:.4f} GHz  |S21|={mag_s[i]:.4f}  "
                  f"prominence={p:.4f}")

    return idx_sorted, prom_sorted, mag_s


def initial_guess_from_dips(f_grid, s21_1d, window, bounds, verbose=True):
    """딥 위치에서 초기값 산출. prior 안쪽으로 안전하게 clip.

    [물리] 반교차에서 두 딥은 f_r ± g 에 위치한다. 따라서
        f_r ≈ (딥1 + 딥2)/2,   g ≈ |딥2 - 딥1|/2
    이것이 g 를 측정하는 표준 방법이다.

    [주의] 딥이 하나뿐이면 이 관계가 성립하지 않는다. 관측량은
    딥 위치 하나인데 미지수는 (f_r, g) 둘이라 축퇴한다. 그 경우
    g 는 prior 로그중앙을 쓰고 그 사실을 명시한다.
    """
    idx, prom, _ = find_dips(f_grid, s21_1d, window, verbose=verbose)

    if len(idx) >= 2:
        two = np.sort(f_grid[idx[:2]])
        fr_g = 0.5 * (two[0] + two[1])
        g_g = abs(two[1] - two[0]) / 2.0
        src = f"딥 2개 {two[0]:.4f}, {two[1]:.4f} GHz"
    elif len(idx) == 1:
        fr_g = float(f_grid[idx[0]])
        # 로그 중앙: 자릿수 범위가 넓은 파라미터의 자연스러운 중앙값.
        # 산술 중앙(0.25)은 상한에 치우친다.
        g_g = float(np.sqrt(bounds['g'][0] * bounds['g'][1]))
        src = f"딥 1개 {fr_g:.4f} GHz (반교차 미분해 -> g 는 prior 로그중앙)"
    else:
        fr_g = float(np.mean(aa_freq_range))
        g_g = float(np.sqrt(bounds['g'][0] * bounds['g'][1]))
        src = "딥 미검출 -> 폴백"

    kappa_g = float(np.sqrt(bounds['kappa'][0] * bounds['kappa'][1]))
    guess = np.array([fr_g, g_g, kappa_g])

    # [FIX-2 보조] prior 경계에서 최소 10% 안쪽으로 밀어 넣는다.
    # 초기값이 경계에 가까우면 walker 산포가 prior 밖으로 나가
    # log_prob=-inf 가 되고, 그 walker 들은 처음부터 못 움직인다.
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = bounds[name]
        margin = 0.10 * (hi - lo)
        guess[i] = np.clip(guess[i], lo + margin, hi - margin)

    if verbose:
        print(f"  -> 초기값 근거: {src}")
        print(f"  -> initial_guess = f_r={guess[0]:.4f}, "
              f"g={guess[1]:.4f}, kappa={guess[2]:.4f}")
    return guess


def make_init_scatter(guess):
    """[FIX-2] 파라미터별 상대 산포 배열.

    f_r~5.0 과 g~0.04 는 크기가 세 자릿수 다르다. 스칼라 산포로는
    한쪽에 적절하면 다른 쪽에 부적절해진다.
    """
    return np.maximum(np.abs(guess) * aa_init_scatter_rel,
                      aa_init_scatter_floor)


def check_prior_edges(theta, bounds, tol_frac=0.02):
    """[FIX-6] 추정값이 prior 경계에 붙었는지 검사.

    [왜 중요한가]
    경계에 붙었다는 것은 최적화가 상자의 벽에서 멈췄다는 뜻이다.
    그 값은 데이터가 지시한 값이 아니라 탐색 범위의 끝이며,
    그 상태의 오차%는 바이어스의 크기로 해석할 수 없다.
    상자를 넓히면 어디까지 갈지 알 수 없기 때문이다.
    """
    hits = []
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = bounds[name]
        width = hi - lo
        if theta[i] <= lo + tol_frac * width:
            hits.append((name, 'lower', lo))
        elif theta[i] >= hi - tol_frac * width:
            hits.append((name, 'upper', hi))
    return hits


def check_tau(result, nsteps):
    """[FIX-7] tau < N/50 기준을 직접 검사.

    [왜 파이프라인의 converged 플래그를 믿지 않는가]
    실제로 모든 시나리오에서 converged=True 가 나왔지만 ALL 은
    tau 기준을 충족하지 못했다. 플래그의 판정 로직과 emcee 표준
    기준이 일치하지 않는다.

    [tau 가 체인 길이와 함께 커질 때]
    보통 긴 체인에서야 느린 모드가 드러났다는 신호다. 짧은 체인은
    그 모드를 몇 번 왕복하지 못해 tau 를 과소평가한다. 이 경우
    체인을 더 늘리거나 walker 수를 늘려야 한다.
    """
    tau = None
    for key in ('tau', 'autocorr_time', 'act'):
        if isinstance(result, dict) and key in result \
                and result[key] is not None:
            tau = np.atleast_1d(np.asarray(result[key], dtype=float))
            break
    if tau is None or not np.all(np.isfinite(tau)):
        return None, None, None
    threshold = nsteps / 50.0
    return tau, threshold, bool(np.all(tau < threshold))


def get_map_theta(result, fallback):
    """[FIX-5] posterior 최빈점(MAP). 없으면 fallback(median) 반환.

    [MAP vs median]
    MAP 은 사후분포가 최대인 점이며 실재하는 표본 중 하나다.
    median 은 성분별 중앙값이므로, 사후분포가 비대칭이거나
    파라미터 간 상관이 강하면 어느 표본과도 일치하지 않는 점이
    될 수 있다. 그 점으로 곡선을 그리면 데이터에서 벗어나 보인다.

    [현재 상태] result 딕셔너리에 표본/로그확률 키가 없어 median 으로
    fallback 되고 있다. 라벨에 그 사실이 표시된다. 실제 키 이름을
    확인해 이 목록에 추가해야 한다.
    """
    if not isinstance(result, dict):
        return np.asarray(fallback), 'median (fallback)'

    samples = None
    for key in ('flat_samples', 'samples', 'chain', 'flatchain'):
        if key in result and result[key] is not None:
            s = np.asarray(result[key])
            samples = s.reshape(-1, s.shape[-1])
            break

    logp = None
    for key in ('log_prob', 'flat_log_prob', 'lnprob', 'log_probability'):
        if key in result and result[key] is not None:
            logp = np.asarray(result[key]).ravel()
            break

    if samples is not None and logp is not None and len(logp) == len(samples):
        return samples[int(np.argmax(logp))], 'MAP'
    return np.asarray(fallback), 'median (MAP 계산 불가)'


def run_mcmc(log_prob, guess, f_grid, data, sigma, bounds, seed):
    """[FIX-4] 시드를 지원하면 넘기고, 아니면 전역 시드만 설정."""
    scatter = make_init_scatter(guess)
    kwargs = dict(
        nwalkers=aa_nwalkers, nsteps=aa_nsteps,
        burn_in_discard=aa_burn_in_discard, thin_by=aa_thin_by,
        bounds=bounds,
        suppress_warnings=False,   # emcee 의 tau 경고를 보이게 둔다.
                                   # 원본은 True 여서 초기화 문제가
                                   # 조용히 지나갔다.
    )
    # run_single_mcmc 의 시그니처를 확인해 시드 인자가 있으면 넘긴다
    try:
        params = inspect.signature(mcmc_pipeline.run_single_mcmc).parameters
        for cand in ('seed', 'random_seed', 'rng'):
            if cand in params:
                kwargs[cand] = seed
                break
    except (TypeError, ValueError):
        pass

    np.random.seed(seed)   # 시드 인자가 없을 때를 위한 보험

    try:
        return mcmc_pipeline.run_single_mcmc(
            log_prob, guess, f_grid, data, sigma,
            init_scatter=scatter, **kwargs)
    except (TypeError, ValueError):
        # init_scatter 가 배열을 안 받으면 스칼라로 후퇴.
        # 이 경우 FIX-2 가 절반만 적용되므로 메시지를 남긴다.
        print("  [주의] init_scatter 배열이 거부되어 스칼라로 후퇴합니다.")
        np.random.seed(seed)
        return mcmc_pipeline.run_single_mcmc(
            log_prob, guess, f_grid, data, sigma,
            init_scatter=float(np.min(scatter)), **kwargs)


# =========================================================
# 신호 생성
# =========================================================
def build_signal(flux_val, use_fano, use_tls, use_tilt, seed=aa_seed):
    """세 단계의 신호를 만든다.

    Returns
    -------
    f_grid        : 주파수 격자
    s21_ideal     : 왜곡 없음, 잡음 없음 (참값 모델 그대로)
    s21_distorted : 왜곡 적용, 잡음 없음
    s21_noisy     : 왜곡 + 잡음 (피팅 대상)
    effects       : 적용된 효과 이름 리스트

    [FIX-11] 세 개를 모두 반환하는 이유
    그림에서 회색(ideal) -> 초록(distorted) -> 빨강(fit) 을 겹쳐 보면
    "왜곡이 무엇을 바꿨고 모델이 어디까지 따라갔는가"가 한 장에
    드러난다. 원본은 s21_distorted 를 'Clean (no distortion)' 이라고
    라벨링해서 정반대로 읽힐 수 있었다.
    """
    f_grid = np.linspace(*aa_freq_range, aa_n_freq)

    # 참값으로 만든 이상적 신호. 이것이 "정답"이다.
    s21_ideal = models.s21_anticrossing_model(
        f_grid, flux_val, aa_f_r0, aa_g, aa_kappa,
        aa_fq_max, aa_EC, aa_tau_transmon)

    s21 = s21_ideal.copy()
    effects = []

    if use_fano:
        # models_v2 의 method='phase' 는 복소 결합 Q 방식.
        # 케이블 지연 D=exp(-2πifτ) 를 먼저 벗겨내야 한다.
        # s21 = (1-L)·D 이므로 (1-s21) 은 공진 성분 L 이 아니다.
        # 날개에서 L->0 이어도 (1-D) 가 남아 진동하고, 베이스라인이
        # 1.36 까지 뜬다. tau 를 반드시 넘길 것.
        s21 = models.fano_lineshape_correction(
            s21, f_grid, aa_f_r0, aa_fano_q,
            method='phase', tau=aa_tau_transmon)
        effects.append(f"Fano(q={aa_fano_q})")

    if use_tls:
        f_tls = aa_f_r0 + aa_tls_offset_from_fr
        s21 = models.add_spurious_tls_dip(
            s21, f_grid, f_tls, aa_tls_coupling, aa_tls_linewidth)
        effects.append(f"TLS(off={aa_tls_offset_from_fr})")

    if use_tilt:
        # add_baseline_tilt 는 2D 를 기대하므로 임시로 차원 추가
        s21 = mock_data.add_baseline_tilt(
            s21[np.newaxis, :], f_grid, aa_tilt_slope)[0, :]
        effects.append(f"Tilt(slope={aa_tilt_slope})")

    # 백색잡음은 항상 마지막에. 시드로 고정해 재현 가능하게 한다.
    s21_noisy = mock_data.add_white_noise(
        s21, aa_white_noise_level, rng=np.random.default_rng(seed))

    return f_grid, s21_ideal, s21, s21_noisy, effects


# =========================================================
# [FIX-9] 반교차 flux 지점 탐색
# =========================================================
def find_anticrossing_flux(n_scan=161, flux_range=(-0.5, 0.5)):
    """왜곡 없는 신호로 flux 를 훑어, 딥이 2개로 갈라지고 그 간격이
    최소가 되는 지점을 찾는다. 그곳이 반교차이며 간격 ≈ 2g.

    [왜 필요한가 — 원본의 Φ=0 문제]
    표준 트랜스몬이라면
        f_q(0) = (5.150+0.250)·sqrt(1) - 0.250 = 5.150
        delta  = 5.150 - 5.000 = 0.150 = 3.75·g
    즉 Φ=0 은 반교차가 아니라 분산 영역이다.
    호프필드 계수가 0.941/0.059 로 갈려 딥이 사실상 하나만 보이고,
    관측량 1개에 미지수 2개(f_r, g)라 축퇴한다.
    그 상태에서 g 는 데이터가 아니라 prior 가 정한다.

    실제 반교차는 f_q(Φ)=f_r 을 푼
        sqrt(|cos(πΦ)|) = (f_r+E_C)/(f_q_max+E_C) = 5.25/5.40
        -> Φ = ±0.1059
    이다. 이 함수는 그것을 수치적으로 찾는다.

    [간격이 2g 와 정확히 같지 않은 이유]
    관측되는 딥은 |S21| 의 극소점이며, 유한 선폭 kappa 때문에 두
    로렌츠가 겹치면서 서로 안쪽으로 당긴다. 격자 분해능(1 MHz)도
    더해진다. 실측 0.0780 vs 2g=0.0800 은 2.5% 차이로 정상 범위.
    """
    print("\n[반교차 flux 탐색] 왜곡 없는 신호로 스캔합니다...")
    f_grid = np.linspace(*aa_freq_range, aa_n_freq)
    best = None
    two_dip_fluxes = []

    for flux in np.linspace(*flux_range, n_scan):
        try:
            s21 = models.s21_anticrossing_model(
                f_grid, flux, aa_f_r0, aa_g, aa_kappa,
                aa_fq_max, aa_EC, aa_tau_transmon)
        except Exception:
            continue
        idx, prom, _ = find_dips(f_grid, s21, aa_smoothing_window,
                                 prominence_frac=0.10,
                                 min_separation_ghz=0.005, verbose=False)
        if len(idx) >= 2:
            sep = abs(f_grid[idx[0]] - f_grid[idx[1]])
            two_dip_fluxes.append(flux)
            if best is None or sep < best[1]:
                best = (flux, sep)

    if best is None:
        print("  딥이 2개로 갈라지는 flux 를 찾지 못했습니다.")
        print("  -> g 가 선폭 대비 너무 작아 반교차가 분해되지 않거나,")
        print("     스캔 범위 밖일 수 있습니다. flux_range 를 넓혀 보세요.")
        return None

    flux_best, sep_best = best
    print(f"  딥 2개가 보이는 flux 구간: "
          f"[{min(two_dip_fluxes):+.3f}, {max(two_dip_fluxes):+.3f}]")
    print(f"  최소 간격 지점: flux={flux_best:+.4f}, "
          f"간격={sep_best:.4f} GHz (2g={2*aa_g:.4f} 와 비교)")
    if abs(sep_best - 2 * aa_g) / (2 * aa_g) > 0.5:
        print("  [주의] 간격이 2g 와 크게 다릅니다. 모델 파라미터를 확인하세요.")
    return flux_best


# =========================================================
# 단일 시나리오 실행
# =========================================================
def run_case(flux_val, use_fano, use_tls, use_tilt, tag, make_plot=True):
    print("\n" + "=" * 62)
    print(f"시나리오: {tag}  (flux={flux_val:+.4f})")
    print("=" * 62)

    f_grid, s21_ideal, s21_distorted, s21_noisy, effects = build_signal(
        flux_val, use_fano, use_tls, use_tilt)
    print(f"  적용된 효과: {effects if effects else ['(백색잡음만)']}")

    guess = initial_guess_from_dips(
        f_grid, s21_noisy, aa_smoothing_window, aa_prior_bounds)

    # --- 잡음 수준 추정 ---
    # [통계] 가우스 우도는 sigma 를 요구한다. sigma 를 과대평가하면
    # 우도가 넓어져서 잔차가 커도 통계적으로 허용된다. 즉 오설정을
    # 숨기는 방향으로 작동한다.
    # 실측: baseline 에서도 2.42배, ALL 에서 6.06배 부풀었다.
    # adaptive 추정이 공진 구조와 왜곡 잔차를 잡음으로 흡수한다.
    sigma_est = diagnostics.estimate_noise_sigma(s21_noisy, method='adaptive')
    ratio = sigma_est / aa_white_noise_level
    print(f"\n  noise_sigma 추정={sigma_est:.5f}  "
          f"참값={aa_white_noise_level}  비율={ratio:.2f}x")
    if ratio > 1.5:
        print("  [주의] 잡음을 과대평가하면 우도가 느슨해져, 잔차가 커도")
        print("        통계적으로 허용됩니다. 오설정이 지표에 안 드러나는 원인.")

    # --- 사후분포 구성: posterior ∝ likelihood × prior ---
    log_prior = likelihood.make_uniform_log_prior(aa_prior_bounds)
    model_kwargs = {'flux_val': flux_val, 'fq_max': aa_fq_max,
                    'EC': aa_EC, 'tau': aa_tau_transmon}
    log_prob = likelihood.make_log_probability(
        models.s21_anticrossing_model, model_kwargs, log_prior,
        likelihood_type='gaussian')

    print("\n  MCMC 실행 중...")
    result = run_mcmc(log_prob, guess, f_grid, s21_noisy,
                      sigma_est, aa_prior_bounds, aa_seed)

    median = np.asarray(result['median'])
    theta_best, best_label = get_map_theta(result, median)

    # ---- 결과 ----
    print("\n  " + "-" * 58)
    print(f"  {'param':8s} {'median':>10s} {'MAP':>10s} "
          f"{'true':>10s} {'err(med)':>10s}")
    print("  " + "-" * 58)
    errs = {}
    for i, name in enumerate(PARAM_NAMES):
        err = abs(median[i] - TRUE_THETA[i]) / TRUE_THETA[i] * 100
        errs[name] = err
        flag = " <-- 10%↑" if err > 10 else ""
        print(f"  {name:8s} {median[i]:10.5f} {theta_best[i]:10.5f} "
              f"{TRUE_THETA[i]:10.5f} {err:9.1f}%{flag}")

    # ---- prior 경계 검사 ----
    hits = check_prior_edges(median, aa_prior_bounds)
    if hits:
        print("\n  [경고] prior 경계 접촉:")
        for name, side, val in hits:
            print(f"    {name} 이(가) {side} 경계 {val} 에 붙었습니다.")
        print("    -> 이 추정값은 '데이터가 말한 값' 이 아니라 "
              "'탐색 범위의 끝' 입니다.")
        print("    -> 이 상태의 오차%는 바이어스 측정값으로 쓸 수 없습니다.")
    else:
        print("\n  [OK] prior 경계 접촉 없음 — 오차는 해석 가능합니다.")

    # ---- 수렴 검사 ----
    tau, thr, ok = check_tau(result, aa_nsteps)
    print(f"\n  수렴 플래그(converged): {result.get('converged')}")
    if tau is not None:
        print(f"  tau = {np.round(tau, 1)},  N/50 = {thr:.1f}")
        if ok:
            print("  [OK] tau < N/50 기준 충족")
        else:
            need = int(np.ceil(50 * np.max(tau)))
            print(f"  [경고] tau >= N/50 — 기준 미충족. 필요 스텝 ~{need}")
    else:
        print("  tau 를 결과에서 찾지 못했습니다 (result 키 확인 필요).")

    # ---- 그림 ----
    if make_plot:
        model_curve = models.s21_anticrossing_model(
            f_grid, flux_val, *theta_best, aa_fq_max, aa_EC, aa_tau_transmon)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # --- Magnitude ---
        # [FIX-11] 회색(이상적) -> 초록(왜곡) -> 빨강(피팅) 순으로 겹쳐
        # 보면, 왜곡이 무엇을 바꿨고 모델이 어디까지 따라갔는지가
        # 한 장에 드러난다.
        #
        # [읽는 법] 빨간 곡선은 양 끝에서 반드시 1.0 으로 수렴한다.
        # 표준 모델에 진폭 스케일 a 나 baseline 기울기 파라미터가 없어
        # 구조적으로 |S21| -> 1 이기 때문이다. 데이터가 그렇지 않으면
        # 그 차이는 잔차로 남을 수밖에 없다.
        # 역설적으로 이 "따라갈 수 없음"이 f_r 과 g 의 정확도를 지켰다.
        # 흡수할 자유도가 없으니 파라미터 쪽으로 새어 들어가지 않았다.
        axes[0].plot(f_grid, np.abs(s21_noisy), '.', ms=3, alpha=0.5,
                     label='Measured (distorted + noise)')
        axes[0].plot(f_grid, np.abs(s21_ideal), '-', lw=1, alpha=0.5,
                     color='gray', label='Ideal (no distortion)')
        axes[0].plot(f_grid, np.abs(s21_distorted), '-', lw=1, alpha=0.7,
                     color='green', label='Distorted (noise-free)')
        axes[0].plot(f_grid, np.abs(model_curve), '-', color='red',
                     label=f'Fit ({best_label})')
        axes[0].axvline(guess[0], color='gray', ls='--', lw=1,
                        label='Initial f_r guess')
        axes[0].set_xlabel('Frequency (GHz)')
        axes[0].set_ylabel('|S21|')
        axes[0].legend(fontsize=8)
        axes[0].set_title(f'Magnitude — {tag}')

        # --- Phase ---
        # unwrap: -pi/+pi 경계에서 생기는 인위적 점프를 제거해
        # 물리적으로 연속인 위상 곡선으로 펼친다. 이를 하지 않으면
        # 공진 근처에서 가짜 불연속이 보인다.
        axes[1].plot(f_grid, np.unwrap(np.angle(s21_noisy)), '.', ms=3,
                     alpha=0.5, label='Measured')
        axes[1].plot(f_grid, np.unwrap(np.angle(s21_ideal)), '-', lw=1,
                     alpha=0.5, color='gray', label='Ideal')
        axes[1].plot(f_grid, np.unwrap(np.angle(model_curve)), '-',
                     color='red', label=f'Fit ({best_label})')
        axes[1].set_xlabel('Frequency (GHz)')
        axes[1].set_ylabel('Phase (rad, unwrapped)')
        axes[1].legend(fontsize=8)
        axes[1].set_title('Phase')

        plt.tight_layout()
        safe = tag.replace(' ', '_').replace('+', '_').replace('/', '_')
        path = os.path.join(aa_output_dir, f'stress_{safe}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  그림 저장: {path}")

    return {'tag': tag, 'median': median, 'map': theta_best,
            'errs': errs, 'prior_hits': hits, 'tau_ok': ok,
            'sigma_ratio': ratio}


# =========================================================
# 메인
# =========================================================
if __name__ == '__main__':
    print("=" * 62)
    print("설정")
    print("=" * 62)
    print(f"  참값: f_r={aa_f_r0}, g={aa_g}, kappa={aa_kappa}")
    print(f"  prior: {aa_prior_bounds}")
    print(f"  nsteps={aa_nsteps}, nwalkers={aa_nwalkers}, seed={aa_seed}")

    # 반교차 flux 결정
    if aa_auto_find_flux:
        flux = find_anticrossing_flux()
        if flux is None:
            flux = aa_case_flux_val
            print(f"  -> 자동 탐색 실패. flux={flux} 로 진행합니다.")
    else:
        flux = aa_case_flux_val

    if aa_run_sweep:
        # [FIX-10] 효과를 하나씩 켜서 기여도를 분리.
        #
        # [왜 필수인가 — 상쇄]
        # 실측 결과 Fano 는 kappa 를 -7.3%(과소), TLS 는 +8.1%(과대)
        # 로 움직였는데, 셋을 동시에 켠 ALL 에서는 +3.3% 로 보였다.
        # 단순 합(+0.8%)도 아닌 비선형 혼합이다.
        # ALL 만 보았다면 개별 효과를 절반 이하로 과소평가했을 것이다.
        #
        # baseline 행은 파이프라인 자체의 검증이자 오차 바닥이다.
        # 이 행이 깨끗하지 않으면 나머지 비교가 성립하지 않는다.
        cases = [
            (False, False, False, 'baseline'),
            (True,  False, False, 'Fano'),
            (False, True,  False, 'TLS'),
            (False, False, True,  'Tilt'),
            (True,  True,  True,  'ALL'),
        ]
    else:
        cases = [(aa_enable_fano, aa_enable_tls, aa_enable_tilt, 'ALL')]

    summary = []
    for fano, tls, tilt, tag in cases:
        summary.append(run_case(flux, fano, tls, tilt, tag))

    # ---- 요약표 ----
    print("\n" + "=" * 62)
    print("요약 — 시나리오별 복원 오차 (median 기준)")
    print("=" * 62)
    print(f"  {'scenario':10s} {'f_r':>9s} {'g':>9s} {'kappa':>9s}  "
          f"{'sigma':>7s} {'prior':>6s} {'tau':>5s}")
    print("  " + "-" * 60)
    for r in summary:
        pri = 'HIT' if r['prior_hits'] else 'ok'
        tt = {True: 'ok', False: 'FAIL', None: '?'}[r['tau_ok']]
        print(f"  {r['tag']:10s} {r['errs']['f_r']:8.1f}% "
              f"{r['errs']['g']:8.1f}% {r['errs']['kappa']:8.1f}%  "
              f"{r['sigma_ratio']:6.2f}x {pri:>6s} {tt:>5s}")

    print("\n  prior=HIT 또는 tau=FAIL 인 행의 오차%는")
    print("  모델 오설정의 크기로 해석할 수 없습니다.")
    print("  baseline 행이 깨끗해야 나머지 비교가 의미를 가집니다.")
