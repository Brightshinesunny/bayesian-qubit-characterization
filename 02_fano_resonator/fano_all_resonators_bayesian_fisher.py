"""
fano_all_resonators_bayesian_fisher.py
=================================================================
[목적] 오늘 field sweep 데이터 하나(resonator_7)에서 확인한
"베이지안/피셔 접근이 circle fit보다 안정적으로 답을 낸다"는 결론이,
우연히 그 데이터 하나에서만 그런 것인지, 아니면 다른 resonator(1~9,
overcoupled)에서도 일반적으로 성립하는지 확인합니다.

[방법론 - 어제 field sweep 스크립트와 동일한 두 방법을 그대로 재사용]
  방법 A. circle fit (fano_models.autofit) - 대수적, 반복 없음
  방법 B. 베이지안/피셔 (curve_fit + fano_fisher_matrix) - 최소제곱
          피팅 후, 피셔행렬 기반 공분산에서 몬테카를로 샘플링

[이번에 새로 하는 것 - "노이즈를 적당히 걷어내고"]
지금까지는 curve_fit에 잡음 크기(sigma)를 넘기지 않고 균등 가중치로
피팅했습니다(암묵적으로 모든 점을 똑같이 신뢰). 이번엔 baseline
구간에서 추정한 noise_sigma를 curve_fit의 sigma 인자로 명시적으로
전달합니다 - 이러면 (1) 잡음이 큰 데이터 포인트의 영향력을 적절히
낮추고, (2) 피셔행렬 계산에 쓰는 sigma와 curve_fit이 내부적으로
가정하는 잡음 수준이 서로 일치해서, 오차 추정치(공분산)가 더
정확해집니다. "잡음을 걷어낸다"는 건 데이터 자체를 지우거나
스무딩하는 게 아니라, "이 정도 잡음이 있다는 걸 피팅 과정에
명시적으로 알려줘서 그 잡음의 영향을 통계적으로 올바르게 반영한다"
는 뜻입니다.

[각 resonator마다 어떤 slice를 볼 것인가]
어제 원저자 예제 노트북과 일관되게, 최저 전력(index=0) slice를
사용합니다 - 원저자가 "가장 깨끗하고 전형적인 선형 공진 형태"로
이 slice를 골랐던 것과 같은 이유입니다.
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')   # 화면 없는 서버 환경에서도 그래프를 파일로 저장하기 위한 설정
import matplotlib.pyplot as plt
import os
import sys
import json

sys.path.insert(0, os.getcwd())
import fano_models
import fano_fisher_matrix as fm


# =========================================================
# STEP 0. 분석자 설정값 (aa_ 접두사)
# =========================================================
aa_upload_dir = '/mnt/user-data/uploads/'
    # 사용자가 업로드한 9개 resonator 파일이 있는 경로. Colab의
    # Google Drive 경로가 아니라, 지금은 여기(이 세션의 업로드 폴더)
    # 파일을 직접 읽음 - 이번엔 분석자(당신)가 Colab에서 돌리는 게
    # 아니라, 제가 여기서 직접 실행해서 결과를 바로 보여드리는 구조.

aa_resonator_files = [
    'resonator_1_powersweep_overcoupled.npz',
    'resonator_2_powersweep_overcoupled.npz',
    'resonator_3_powersweep_overcoupled.npz',
    'resonator_4_powersweep_overcoupled.npz',
    'resonator_5_powersweep_overcoupled.npz',
    'resonator_6_powersweep_overcoupled.npz',
    'resonator_7_powersweep_overcoupled.npz',
    'resonator_8_powersweep_overcoupled.npz',
    'resonator_9_powersweep_overcoupled.npz',
]
aa_power_slice_index = 0   # 최저 전력 slice (원저자 예제와 일관되게)
aa_n_ports = 1.0            # reflection_port
aa_isolation_db = 15        # circle fit의 Fano 불확실성 범위 계산에 쓸 가정값
aa_output_dir = '/home/claude/qubit_analysis_template/outputs/'
os.makedirs(aa_output_dir, exist_ok=True)

print("=" * 60)
print("분석자 설정값(aa_*) 요약:")
for k, v in list(globals().items()):
    if k.startswith("aa_") and not isinstance(v, list):
        print(f"  {k:24s} = {v}")
print("=" * 60)


# =========================================================
# STEP 1. .npz 파일 로더 (fano_loader.py와 동일한 로직을 이 파일
# 안에 직접 넣어둠 - 독립 실행 가능하게)
# =========================================================
def load_fano_npz(filepath):
    """
    resonator_*_powersweep_overcoupled.npz 파일 하나를 읽어 복소수
    S21과 cable_delay 등을 반환. fano_loader.py의 load_fano_npz와
    동일한 로직.
    """
    raw = np.load(filepath, allow_pickle=True)
    amplitude = raw['amplitude']
    phase = raw['phase']
    freq_hz = raw['frequency']
    power_dbm = raw['power']
    s21 = amplitude * np.exp(1j * phase)
    settings_dict = json.loads(raw['settings'][0])
    cable_delay = settings_dict.get('vna', {}).get('port1_edel', None)
    return {
        'power_dbm': power_dbm, 'freq_hz': freq_hz,
        's21': s21, 'cable_delay': cable_delay,
    }


def estimate_noise_sigma(s21_1d, mag):
    """
    baseline(공진 딥에서 먼, 진폭이 가장 큰 구간)의 실수부 흔들림으로
    잡음 크기(표준편차)를 추정. reflection 모델은 공진에서 진폭이
    "작아지는" 딥 형태이므로, 진폭 상위 20% 구간을 baseline으로 봄.
    """
    mask = mag > np.percentile(mag, 80)
    return np.std(np.real(s21_1d[mask]))


# =========================================================
# STEP 2. 파라미터/모델 정의 (fano_models.Sij를 curve_fit 형태로 감싸는 함수)
# =========================================================
param_names = ['fr', 'Ql', 'Qc', 'phi', 'a', 'alpha']


def make_model_wrapper(delay, n_ports):
    """
    delay, n_ports를 고정한 채로, curve_fit이 요구하는 형태
    "model(f, p1, p2, ...) -> 1차원 실수 배열"에 맞춰 Sij를 감싸는
    함수를 만들어 반환. 복소수 S21을 실수부+허수부를 이어붙인 하나의
    긴 실수 배열로 바꿔서 curve_fit이 다룰 수 있게 함(curve_fit은
    기본적으로 복소수 출력을 직접 지원하지 않으므로, 이렇게 "실수
    배열로 펼치는" 트릭이 표준적인 우회법).
    """
    def model_wrapper(f, fr, Ql, Qc, phi, a, alpha):
        m = fano_models.Sij(f, fr, Ql, Qc, phi, a, alpha, delay, n_ports=n_ports)
        return np.concatenate([np.real(m), np.imag(m)])
    return model_wrapper


# =========================================================
# STEP 3. 9개 resonator 전부 반복 처리
# =========================================================
results_summary = []

for filename in aa_resonator_files:
    full_path = os.path.join(aa_upload_dir, filename)
    if not os.path.exists(full_path):
        print(f"[건너뜀] 파일 없음: {filename}")
        continue

    data = load_fano_npz(full_path)
    f_grid = data['freq_hz']
    s21 = data['s21'][aa_power_slice_index, :]
    delay = data['cable_delay']
    power_val = data['power_dbm'][aa_power_slice_index]
    mag = np.abs(s21)

    # --- 방법 A: circle fit ---
    try:
        cf_result = fano_models.autofit(f_grid, s21, n_ports=aa_n_ports,
                                          fixed_delay=delay, isolation=aa_isolation_db)
        cf_ok = True
    except Exception as e:
        cf_result = None
        cf_ok = False

    # --- 방법 B: 베이지안/피셔 (잡음 sigma를 curve_fit에 명시적으로 전달) ---
    noise_sigma = estimate_noise_sigma(s21, mag)
        # "노이즈를 적당히 걷어낸다"의 핵심: 이 sigma를 아래 curve_fit의
        # sigma 인자로 넘겨서, 각 데이터 포인트의 신뢰도를 잡음 수준에
        # 맞게 반영함(균등 가중치가 아니라, 잡음 크기의 역수에 비례하는
        # 가중치를 암묵적으로 적용하는 것과 같은 효과 - 최소제곱법에서
        # sigma를 주는 것의 표준적인 의미).

    fr_guess = f_grid[np.argmin(mag)]
    half_level = (mag.max() + mag.min()) / 2
    above_half = np.where(mag > half_level)[0]
    fwhm_guess = f_grid[above_half[-1]] - f_grid[above_half[0]] if len(above_half) > 1 else 1e6
    Ql_guess = fr_guess / max(fwhm_guess, 1e3)
    p0 = [fr_guess, Ql_guess, Ql_guess * 1.1, 0.0, mag.max() - mag.min(), 0.0]

    model_wrapper = make_model_wrapper(delay, aa_n_ports)
    y_data = np.concatenate([np.real(s21), np.imag(s21)])
    sigma_for_fit = np.full_like(y_data, noise_sigma)

    # [버그 수정] curve_fit에 물리적 제약(bounds)을 안 줬더니, 일부
    # resonator(1, 3번)에서 Ql/Qc가 "음수"라는 물리적으로 불가능한
    # 값으로 수렴해버리는 문제를 발견했습니다 (진단 과정에서 popt가
    # Ql=-0.0057, Qc=-984.5처럼 나온 걸 확인). 최소제곱법은 "수학적으로
    # 그럴듯한" 지점이면 물리 법칙을 모르고도 아무 곳에나 수렴할 수
    # 있으므로, 아래처럼 상식적인 물리 범위를 명시적으로 강제해야
    # 이런 엉뚱한 국소최적점을 피할 수 있습니다.
    lower_bounds = [f_grid.min(), 1.0, 1.0, -np.pi, 1e-8, -np.pi]
    upper_bounds = [f_grid.max(), 1e8, 1e8, np.pi, 1.0, np.pi]
        # fr: 반드시 스캔 범위 안 / Ql,Qc: 반드시 양수(물리적으로 음의
        # 품질계수는 존재할 수 없음) / phi,alpha: 위상은 -pi~pi 범위가
        # 자연스러운 한계 / a: 진폭 스케일도 양수여야 함.

    # [추가 수정] resonator_1을 chi-square로 직접 검증해본 결과, 단일
    # 초기값(p0)에서 시작한 curve_fit이 진짜 최적점이 아니라 훨씬 나쁜
    # 국소최적점(chi2=2426, 진짜 최적점은 chi2=409)에 빠지는 걸
    # 확인했습니다. 원인은 phi, alpha가 "주기적(periodic, -pi~pi를
    # 도는 각도)" 파라미터라서, 초기 위상을 잘못 잡으면 전혀 다른(그러나
    # 그럴듯해 보이는) 지점에 갇히기 쉽기 때문입니다 - circle fit이
    # 반복탐색이 없어 이런 함정 자체가 없는 것과 대조적으로, curve_fit
    # 같은 반복적 최소제곱법의 근본적인 약점입니다.
    #
    # 해결책(multi-start, 다중시작 최적화): phi, alpha의 초기값을 여러
    # 후보로 바꿔가며 curve_fit을 여러 번 시도하고, 그중 데이터에
    # 가장 잘 맞는(chi-square가 가장 작은) 결과를 최종 채택합니다.
    # 이렇게 하면 국소최적점 하나에 운명을 걸지 않고, 여러 출발점 중
    # "가장 좋은 답"을 고르는 방식으로 안정성을 높일 수 있습니다.
    phi_candidates = [-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5]
        # -pi~pi 범위를 넉넉히 커버하는 7개의 서로 다른 phi 시작점.

    best_popt, best_chi2 = None, np.inf
    for phi0 in phi_candidates:
        p0_trial = [fr_guess, Ql_guess, Ql_guess * 1.1, phi0, mag.max() - mag.min(), phi0]
            # alpha 초기값도 phi0와 같이 바꿔서 시도(회전 각도들이 서로
            # 연관되어 있을 가능성이 높으므로, 같이 스캔하는 것이 효율적).
        try:
            popt_trial, _ = curve_fit(
                model_wrapper, f_grid, y_data, p0=p0_trial,
                sigma=sigma_for_fit, absolute_sigma=True,
                bounds=(lower_bounds, upper_bounds), maxfev=30000
            )
            model_trial = fano_models.Sij(f_grid, *popt_trial, delay, n_ports=aa_n_ports)
            residual = s21 - model_trial
            chi2_trial = np.sum((np.real(residual)/noise_sigma)**2
                                  + (np.imag(residual)/noise_sigma)**2)
                # chi-square(카이제곱): 모델과 데이터의 차이를 잡음
                # 크기로 정규화해 제곱합한 값 - 작을수록 데이터를 더
                # 잘 설명하는 파라미터라는 뜻(오늘 계속 써온 최소제곱
                # 원리와 동일한 지표를 "최적점 후보 비교"에도 재사용).
            if chi2_trial < best_chi2:
                best_chi2 = chi2_trial
                best_popt = popt_trial
        except Exception:
            continue

    popt = best_popt
    bf_ok = popt is not None
        # multi-start 반복문에서 여러 phi 초기값 후보 중 chi2가 가장
        # 작았던(=데이터에 가장 잘 맞았던) 결과를 최종 popt로 채택.
        # absolute_sigma=True를 계속 사용한 이유: sigma 값을 이미
        # baseline에서 측정한 절대적인 잡음 크기로 취급하기 위함
        # (기본값 False로 두면 curve_fit이 내부적으로 잔차 크기에
        # 맞춰 sigma를 재조정해버려, 우리가 직접 측정한 noise_sigma
        # 정보가 무시됨).

    if bf_ok:
        fixed_kwargs = {'delay': delay, 'n_ports': aa_n_ports}
        F = fm.fisher_information_matrix(fano_models.Sij, popt, param_names,
                                           f_grid, noise_sigma, fixed_kwargs,
                                           step_fraction=1e-6)
            # [수정] step_fraction 기본값(1e-4) 대신 1e-6 사용. 어제
            # Zenodo 데이터 분석에서 fr처럼 절대 스케일이 매우 큰
            # 파라미터(~1e9~1e10)는 기본 step_fraction으로 수치미분을
            # 하면 미분 스텝이 공진 폭보다 커져서 부정확해지는 문제를
            # 겪었던 것과 동일한 이유 - 더 작은 스텝으로 정밀하게 미분.
        cov = fm.covariance_from_fisher(F)
        mc_samples = np.random.default_rng(0).multivariate_normal(popt, cov, size=20000)
        Ql_mc, Qc_mc = mc_samples[:, 1], mc_samples[:, 2]
        valid = (Ql_mc > 0) & (Qc_mc > 0)
        Qi_mc = 1.0 / (1.0 / Ql_mc[valid] - 1.0 / Qc_mc[valid])
        Qi_mc_positive = Qi_mc[(Qi_mc > 0) & (Qi_mc < 1e7)]
        Qi_bf_median = np.median(Qi_mc_positive) if len(Qi_mc_positive) > 0 else np.nan
        Qi_bf_width = (np.percentile(Qi_mc_positive, 84) - np.percentile(Qi_mc_positive, 16)
                        if len(Qi_mc_positive) > 0 else np.nan)
            # "폭"을 68% 신뢰구간(16~84 백분위수 차이)으로 정의 - 표준
            # 정규분포의 ±1-sigma에 해당하는 폭과 같은 개념이라, 좁을수록
            # 더 확신 있는 추정이라는 뜻.
    else:
        Qi_bf_median, Qi_bf_width = np.nan, np.nan

    results_summary.append({
        'resonator': filename.split('_powersweep')[0],
        'power_dbm': power_val,
        'noise_sigma': noise_sigma,
        'cf_Qi_min': cf_result['Qi_min'] if cf_ok else np.nan,
        'cf_Qi_max': cf_result['Qi_max'] if cf_ok else np.nan,
        'cf_is_inf': np.isinf(cf_result['Qi_max']) if cf_ok else True,
        'bf_Qi_median': Qi_bf_median,
        'bf_Qi_width_68pct': Qi_bf_width,
    })

    if cf_ok:
        cf_qi_str = "inf" if np.isinf(cf_result['Qi_max']) else f"{cf_result['Qi_max']:.0f}"
        print(f"{filename}: circle fit Qi=[{cf_result['Qi_min']:.0f}, {cf_qi_str}], "
              f"베이지안 Qi 중앙값={Qi_bf_median:.0f}, 68%폭={Qi_bf_width:.0f}")
    else:
        print(f"{filename}: circle fit 실패")


# =========================================================
# STEP 4. 전체 요약표 및 시각화
# =========================================================
print("\n" + "=" * 100)
print(f"{'Resonator':<14} {'전력(dBm)':>10} {'circle fit Qi_max=inf?':>22} "
      f"{'베이지안 Qi 중앙값':>18} {'베이지안 68%폭':>16}")
print("=" * 100)
n_cf_inf = 0
for r in results_summary:
    inf_flag = "예 (상한없음)" if r['cf_is_inf'] else "아니오"
    if r['cf_is_inf']:
        n_cf_inf += 1
    print(f"{r['resonator']:<14} {r['power_dbm']:>10.1f} {inf_flag:>22} "
          f"{r['bf_Qi_median']:>18.0f} {r['bf_Qi_width_68pct']:>16.0f}")

print("-" * 100)
print(f"\n9개 resonator 중 circle fit이 Qi_max=inf를 낸 개수: {n_cf_inf}/9")
print(f"베이지안/피셔는 {sum(1 for r in results_summary if not np.isnan(r['bf_Qi_median']))}/9 에서 유한한 답을 제공")


# 시각화: 9개 resonator의 베이지안 Qi 중앙값과 68% 신뢰폭을 errorbar로 한눈에 비교
fig, ax = plt.subplots(figsize=(10, 5.5))
names = [r['resonator'].replace('resonator_', 'R') for r in results_summary]
medians = [r['bf_Qi_median'] for r in results_summary]
widths = [r['bf_Qi_width_68pct'] for r in results_summary]
colors = ['tab:red' if r['cf_is_inf'] else 'tab:blue' for r in results_summary]

ax.errorbar(range(len(names)), medians, yerr=[np.array(widths)/2]*1, fmt='o',
            ecolor='gray', capsize=4, ms=0)
    # yerr에 68%폭의 절반을 줘서, 중앙값 기준 위아래로 대칭적인 오차
    # 막대를 그림(정확히는 16~84 백분위수가 중앙값 기준 대칭이 아닐 수도
    # 있지만, 시각적 요약을 위한 근사).
for i, (m, c) in enumerate(zip(medians, colors)):
    ax.plot(i, m, 'o', color=c, ms=10)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names)
ax.set_ylabel('Qi (베이지안/피셔 추정, 중앙값 ± 68%)')
ax.set_yscale('log')
ax.set_title('9개 overcoupled resonator - 베이지안/피셔로 얻은 Qi 비교\n'
              '(빨간 점 = circle fit이 이 slice에서 Qi_max=inf를 낸 경우)')
plt.tight_layout()
plt.savefig(os.path.join(aa_output_dir, 'fano_all_resonators_comparison.png'), dpi=140, bbox_inches='tight')
print(f"\n비교 그래프 저장 완료: "
      f"{os.path.join(aa_output_dir, 'fano_all_resonators_comparison.png')}")
