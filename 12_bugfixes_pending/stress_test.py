"""
Fano / TLS / baseline-tilt 주입 스트레스 테스트

참값이 알려진 합성 데이터에 계통 효과를 주입하고, 그것들을 포함하지
않는 표준 모델로 피팅했을 때 파라미터 복원이 얼마나 무너지는지 측정.

원본 대비 변경점
  [FIX-1]  prior 상한 확대 — g, kappa 가 0.1 벽에 붙어 있었음
  [FIX-2]  walker 초기 산포를 파라미터별 상대값으로
  [FIX-3]  딥 탐색을 detrend + prominence + distance 기준으로 교체
  [FIX-4]  MCMC 시드 고정
  [FIX-5]  median 대신 MAP 으로 최적 곡선 표시
  [FIX-6]  prior 경계 접촉 자동 검사
  [FIX-7]  tau vs N/50 수렴 기준 명시적 검사 + nsteps 확대
  [FIX-8]  moving average 길이를 입력과 동일하게 고정
  [FIX-9]  반교차 flux 지점 자동 탐색
  [FIX-10] 시나리오 개별 스윕 — 각 효과의 기여를 분리
  [FIX-11] 그림에 ideal / distorted / fit 세 곡선을 모두 표시
           (기존 'Clean (no distortion)' 라벨은 실제로는 왜곡이
            적용된 무잡음 신호였음 — 정반대로 읽힐 수 있었다)

전제: models_v2.py 를 쓴다. v1 의 fano_lineshape_correction 은
      /np.max 정규화로 신호 전체를 1/(1+q^2) 배로 눌렀다.
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

aa_seed = 7
aa_f_r0, aa_g, aa_kappa = 5.000, 0.040, 0.030
aa_fq_max, aa_EC, aa_tau_transmon = 5.150, 0.250, 0.12
aa_n_freq = 401
aa_freq_range = (4.8, 5.2)

aa_white_noise_level = 0.015

# --- 시나리오 스위치 (aa_run_sweep=False 일 때만 쓰임) ---
aa_enable_fano = True
aa_fano_q = 2.0

aa_enable_tls = True
aa_tls_offset_from_fr = 0.005
aa_tls_coupling = 0.4      # NOTE: baseline 대비 40% 딥. 실제 TLS 보다 크다.
aa_tls_linewidth = 0.006   #       현실적 강도는 0.02~0.15 정도. 함께 시험할 것.

aa_enable_tilt = True
aa_tilt_slope = 0.15

# --- 실행 모드 ---
aa_run_sweep = True        # True: 시나리오를 하나씩 켜며 기여도 분리
aa_auto_find_flux = True   # True: 반교차 flux 를 스캔해서 자동 선택
aa_case_flux_val = 0.0     # auto_find_flux=False 일 때만 사용

# --- MCMC 설정 ---
aa_smoothing_window = 5
aa_nwalkers = 32
aa_nsteps = 25000          # [FIX-7] ALL 의 kappa 는 tau=427 -> 50*tau≈21300 필요
aa_burn_in_discard = 5000
aa_thin_by = 10

# [FIX-2] 파라미터별 상대 산포. 초기값이 작아도 prior 밖으로 안 나감
aa_init_scatter_rel = 0.05     # 초기값의 5%
aa_init_scatter_floor = 1e-4

# [FIX-1] 상한 확대. 원본은 g, kappa 모두 (0.001, 0.1) 이었고
#         추정값이 0.09623 / 0.10000 으로 벽에 닿아 있었다.
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
    f_grid 로 인덱싱할 때 window//2 만큼 주파수가 어긋난다.
    """
    if window < 2:
        return np.asarray(x, dtype=float)
    k = int(window)
    pad = k // 2
    xp = np.pad(np.asarray(x, dtype=float), (pad, k - 1 - pad), mode='edge')
    return np.convolve(xp, np.ones(k) / k, mode='valid')


def detrended_magnitude(f_grid, s21_1d, window):
    """baseline tilt 를 1차 다항식으로 제거한 |S21|.

    tilt 가 켜져 있으면 '절대적으로 가장 낮은 점' 이 공진이 아니라
    baseline 이 낮은 구간이 된다. 딥 탐색 전에 반드시 제거해야 한다.
    """
    mag = np.abs(s21_1d)
    mag_s = moving_average_same(mag, window)
    coef = np.polyfit(f_grid, mag_s, 1)
    baseline = np.polyval(coef, f_grid)
    return mag_s, baseline, mag_s - baseline


def find_dips(f_grid, s21_1d, window,
              prominence_frac=0.15, min_separation_ghz=0.010, verbose=True):
    """[FIX-3] detrend + prominence + 최소간격 기준 딥 탐색.

    원본은 단순 극소점을 모두 모아 '절대 깊이' 순으로 정렬했다.
    그 결과 tilt 로 낮아진 구간의 잡음 기복(깊이 0.012~0.019)이
    상위를 차지하고, 3 MHz 떨어진 이웃 두 점이 선택되어
    g_guess=0.0015 (참값의 1/27) 가 나왔다.
    """
    mag_s, baseline, mag_d = detrended_magnitude(f_grid, s21_1d, window)

    df = float(np.median(np.diff(f_grid)))
    distance = max(1, int(round(min_separation_ghz / df)))
    span = float(np.ptp(mag_d))
    prominence = max(prominence_frac * span, 1e-6)

    idx, props = find_peaks(-mag_d, prominence=prominence, distance=distance)

    order = np.argsort(props['prominences'])[::-1]   # 돌출도 큰 순
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
    """딥 위치에서 초기값 산출. prior 안쪽으로 안전하게 clip."""
    idx, prom, _ = find_dips(f_grid, s21_1d, window, verbose=verbose)

    if len(idx) >= 2:
        two = np.sort(f_grid[idx[:2]])
        fr_g = 0.5 * (two[0] + two[1])
        g_g = abs(two[1] - two[0]) / 2.0
        src = f"딥 2개 {two[0]:.4f}, {two[1]:.4f} GHz"
    elif len(idx) == 1:
        # 딥이 하나면 반교차가 분해되지 않은 것. 폭에서 g 를 짐작할 수
        # 없으므로 prior 로그중앙을 쓰고 그 사실을 명시한다.
        fr_g = float(f_grid[idx[0]])
        g_g = float(np.sqrt(bounds['g'][0] * bounds['g'][1]))
        src = f"딥 1개 {fr_g:.4f} GHz (반교차 미분해 -> g 는 prior 로그중앙)"
    else:
        fr_g = float(np.mean(aa_freq_range))
        g_g = float(np.sqrt(bounds['g'][0] * bounds['g'][1]))
        src = "딥 미검출 -> 폴백"

    kappa_g = float(np.sqrt(bounds['kappa'][0] * bounds['kappa'][1]))
    guess = np.array([fr_g, g_g, kappa_g])

    # prior 경계에서 최소 10% 안쪽으로 밀어 넣는다
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
    """[FIX-2] 파라미터별 상대 산포 배열."""
    return np.maximum(np.abs(guess) * aa_init_scatter_rel,
                      aa_init_scatter_floor)


def check_prior_edges(theta, bounds, tol_frac=0.02):
    """[FIX-6] 추정값이 prior 경계에 붙었는지 검사."""
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
    """[FIX-7] tau < N/50 기준을 직접 검사."""
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

    median 은 성분별 중앙값이라, posterior 가 비대칭이면 실제로는
    어느 표본과도 일치하지 않는 점이 될 수 있다.
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
        bounds=bounds, suppress_warnings=False,   # 경고를 보이게 둔다
    )
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
        # init_scatter 가 배열을 안 받으면 스칼라로 후퇴
        print("  [주의] init_scatter 배열이 거부되어 스칼라로 후퇴합니다.")
        np.random.seed(seed)
        return mcmc_pipeline.run_single_mcmc(
            log_prob, guess, f_grid, data, sigma,
            init_scatter=float(np.min(scatter)), **kwargs)


# =========================================================
# 신호 생성
# =========================================================
def build_signal(flux_val, use_fano, use_tls, use_tilt, seed=aa_seed):
    """
    Returns
    -------
    f_grid        : 주파수 격자
    s21_ideal     : 왜곡 없음, 잡음 없음 (참값 모델 그대로)
    s21_distorted : 왜곡 적용, 잡음 없음
    s21_noisy     : 왜곡 + 잡음 (피팅 대상)
    effects       : 적용된 효과 이름 리스트
    """
    f_grid = np.linspace(*aa_freq_range, aa_n_freq)

    s21_ideal = models.s21_anticrossing_model(
        f_grid, flux_val, aa_f_r0, aa_g, aa_kappa,
        aa_fq_max, aa_EC, aa_tau_transmon)

    s21 = s21_ideal.copy()
    effects = []

    if use_fano:
        # models_v2: method='phase' 는 케이블 지연을 벗겨낸 뒤 공진
        # 성분만 위상 회전시킨다. tau 를 반드시 넘겨야 한다.
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
        s21 = mock_data.add_baseline_tilt(
            s21[np.newaxis, :], f_grid, aa_tilt_slope)[0, :]
        effects.append(f"Tilt(slope={aa_tilt_slope})")

    s21_noisy = mock_data.add_white_noise(
        s21, aa_white_noise_level, rng=np.random.default_rng(seed))

    return f_grid, s21_ideal, s21, s21_noisy, effects


# =========================================================
# [FIX-9] 반교차 flux 지점 탐색
# =========================================================
def find_anticrossing_flux(n_scan=161, flux_range=(-0.5, 0.5)):
    """왜곡 없는 신호로 flux 를 훑어, 딥이 2개로 갈라지고 그 간격이
    최소가 되는 지점을 찾는다. 그곳이 반교차이며 간격 ~ 2g.

    Φ=0 은 표준 transmon 이라면 fq(0)=fq_max=5.150, f_r0=5.000 이므로
    detuning=0.150=3.75g 인 dispersive 영역이다. 그 경우 혼성 모드의
    가중치가 0.941/0.059 로 갈려 딥이 사실상 하나만 보이고,
    관측량 1개에 미지수 2개(f_r, g)라 축퇴한다.
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

    sigma_est = diagnostics.estimate_noise_sigma(s21_noisy, method='adaptive')
    ratio = sigma_est / aa_white_noise_level
    print(f"\n  noise_sigma 추정={sigma_est:.5f}  "
          f"참값={aa_white_noise_level}  비율={ratio:.2f}x")
    if ratio > 1.5:
        print("  [주의] 잡음을 과대평가하면 우도가 느슨해져, 잔차가 커도")
        print("        통계적으로 허용됩니다. 오설정이 지표에 안 드러나는 원인.")

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
        # 동시에 켠 상태만 보면 상쇄로 개별 크기를 과소평가한다.
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
