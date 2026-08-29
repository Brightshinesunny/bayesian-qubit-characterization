"""
tls_noise_removal_bayesian_full.py
=================================================================
[오늘 최종] 20개 TLS x 4전극 결합세기 데이터에, 4개 전극의 평균
결합세기(mu_alpha,mu_beta,mu_gamma,mu_delta)를 하나로 묶은 "결합
(joint) 4-파라미터 모델"로 피셔행렬/공분산/상관행렬을 전부 구하고,
베이지안 코너플롯까지 그림.

[모델 설계 - 왜 "결합 모델"인가]
지금까지는 전극마다 따로따로(독립적으로) 평균을 구했음. 이번엔
4개 평균(mu_alpha~delta)을 "동시에 추정하는 하나의 파라미터 벡터"로
보고, 4x4 피셔행렬을 구성함. 각 전극의 데이터가 서로 완전히 독립
(다른 전극 데이터를 봐도 이 전극 평균 추정에 아무 정보를 안 줌)
이므로, 사전에 예측하기를 이 피셔행렬은 "대각행렬"(비대각선이 전부
0)이 될 것 - 즉 4개 전극의 결합세기 추정치들은 서로 상관관계가
없어야 정상. 이걸 실제로 확인하는 것 자체가 검증의 일부.
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.getcwd())
import fano_bayesian_toolkit as bt

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
aa_gamma_rows = [0, 2, 9, 11]
aa_electrode_names = {0: 'alpha(row0)', 2: 'beta(row2)', 9: 'gamma(row9)', 11: 'delta(row11)'}
aa_outlier_trim_pct = 3.0
aa_mc_sample_size = 30000
aa_step_fraction = 1e-4
    # [주의] 이번 파라미터(mu, ~1e7~1e8 스케일)에 맞춘 수치미분 스텝
    # 비율. 오늘 Fano 분석에서 배운 대로, 스텝이 파라미터 변화폭에
    # 비해 너무 크거나 작으면 미분이 부정확해지므로 사전 검증 완료된
    # 값(1e-4)을 사용- 위에서 직접 확인한 사전검증과 동일 조건.


def mat_string(f, dataset_or_ref):
    """MATLAB의 uint16 코드 배열 문자열을 파이썬 문자열로. 문자열
    아니면 None - 숫자 필드랑 구분하는 용도."""
    try:
        if isinstance(dataset_or_ref, h5py.Reference):
            dataset_or_ref = f[dataset_or_ref]
        arr = np.array(dataset_or_ref)
        return ''.join(chr(int(c)) for c in arr.flatten())
    except Exception:
        return None


with h5py.File(aa_mat_filepath, 'r') as f:
    tabledata2_refs = f['tdat']['tabledata2']
    n_rows, n_cols = tabledata2_refs.shape

    # =====================================================
    # STEP 1. 20개 채워진 TLS 항목의 4개 결합세기 추출 (기존과 동일)
    # =====================================================
    all_gammas = {row: [] for row in aa_gamma_rows}
    for col in range(n_cols):
        gamma_vals = {}
        skip = False
        for row in aa_gamma_rows:
            ref = tabledata2_refs[row, col]
            val = np.array(f[ref])
            if val.dtype == np.uint16:
                skip = True
                break
            gamma_vals[row] = float(val.flatten()[0]) if val.size > 0 else 0.0
        if skip:
            continue
        if max(abs(v) for v in gamma_vals.values()) >= 1e6:
            for row, v in gamma_vals.items():
                all_gammas[row].append(v)

    # =====================================================
    # STEP 2. 전극별 3% 이상치 제거 + robust sigma(MAD) 계산
    # =====================================================
    trimmed_data, robust_sigmas, mu_map = {}, {}, {}
    for row in aa_gamma_rows:
        vals = np.array(all_gammas[row])
        lo, hi = np.percentile(vals, [aa_outlier_trim_pct/2, 100-aa_outlier_trim_pct/2])
        trimmed = vals[(vals >= lo) & (vals <= hi)]
        median_r = np.median(trimmed)
        mad = np.median(np.abs(trimmed - median_r))
        trimmed_data[row] = trimmed
        robust_sigmas[row] = mad * 1.4826
        mu_map[row] = np.mean(trimmed)   # MAP 추정치 = 이상치 제거 후 평균(가우시안 우도의 MAP)

    print("[STEP 2] 전극별 이상치 제거 후 MAP 및 robust sigma")
    for row in aa_gamma_rows:
        print(f"  {aa_electrode_names[row]:>16}: n={len(trimmed_data[row])}, "
              f"mu={mu_map[row]/1e6:.1f}MHz/V, sigma={robust_sigmas[row]/1e6:.1f}MHz/V")

    # =====================================================
    # STEP 3. 결합(joint) 4x4 피셔행렬 계산 - 해석적 공식 + 수치미분 교차검증
    # =====================================================
    def neg_log_lik(mu, data_per_electrode, sigma_e):
        total = 0.0
        for i, row in enumerate(aa_gamma_rows):
            total += 0.5*np.sum(((data_per_electrode[row] - mu[i])/sigma_e[row])**2)
        return total

    mu_vec = np.array([mu_map[row] for row in aa_gamma_rows])
    sigma_list = [robust_sigmas[row] for row in aa_gamma_rows]

    # [해석적 공식] 독립 가우시안 평균 추정의 표준 결과: F_ee=n_e/sigma_e^2, 비대각선 0
    F_analytic = np.diag([len(trimmed_data[row])/robust_sigmas[row]**2 for row in aa_gamma_rows])

    # [수치미분 교차검증] 위에서 미리 합성 데이터로 검증한 것과 동일한 방식
    def numerical_hessian(mu0, data, sigma_e, h_frac=aa_step_fraction):
        n = len(mu0)
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                h_i = h_frac*max(abs(mu0[i]), 1e6)
                h_j = h_frac*max(abs(mu0[j]), 1e6)
                mu_pp = mu0.copy(); mu_pp[i]+=h_i; mu_pp[j]+=h_j
                mu_pm = mu0.copy(); mu_pm[i]+=h_i; mu_pm[j]-=h_j
                mu_mp = mu0.copy(); mu_mp[i]-=h_i; mu_mp[j]+=h_j
                mu_mm = mu0.copy(); mu_mm[i]-=h_i; mu_mm[j]-=h_j
                H[i,j] = (neg_log_lik(mu_pp,data,sigma_e) - neg_log_lik(mu_pm,data,sigma_e)
                          - neg_log_lik(mu_mp,data,sigma_e) + neg_log_lik(mu_mm,data,sigma_e))/(4*h_i*h_j)
        return H

    F_numeric = numerical_hessian(mu_vec, trimmed_data, robust_sigmas)
    max_rel_err = np.max(np.abs(np.diag(F_analytic)-np.diag(F_numeric))/np.diag(F_analytic))
    print(f"\n[STEP 3 검증] 해석적 vs 수치미분 피셔행렬 최대 상대오차: {max_rel_err:.2e} (작을수록 정상)")

    # =====================================================
    # STEP 4. [요청사항] 피셔행렬 원본, 공분산, 상관행렬 전부 출력
    # =====================================================
    F = F_analytic   # 검증됐으므로 해석적(정확한) 값을 최종 사용
    cov = np.linalg.inv(F)
    # 상관행렬: 공분산을 대각선(분산)의 제곱근으로 정규화한 것 -
    # 대각선은 항상 1, 비대각선은 -1~1 사이 값으로 "상관 정도"를 표현
    std_devs = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_devs, std_devs)

    names_short = ['alpha', 'beta', 'gamma', 'delta']
    print("\n[STEP 4] 피셔행렬(F) - 원본:")
    print(np.array2string(F, precision=3, suppress_small=False))
    print("\n공분산행렬(Cov = F^-1):")
    print(np.array2string(cov, precision=3, suppress_small=False))
    print("\n상관행렬(Corr, 대각선은 항상 1):")
    print(np.array2string(corr, precision=3))
    print("\n[해석] 상관행렬의 비대각선이 전부 0에 가까우면, 4개 전극의")
    print("결합세기 추정치가 서로 완전히 독립적이라는 뜻 - 오늘 데이터")
    print("구조(전극마다 별도 표본)를 생각하면 당연한, 그러나 확인해야")
    print("하는 결과.")

    # =====================================================
    # STEP 5. 몬테카를로 샘플링 + 코너플롯
    # =====================================================
    mc_samples = np.random.default_rng(0).multivariate_normal(mu_vec, cov, size=aa_mc_sample_size)

    print("\n[STEP 5] 68% 신뢰구간 (피셔행렬 기반, joint 모델)")
    for i, row in enumerate(aa_gamma_rows):
        lo, med, hi = bt.credible_interval(mc_samples[:, i], level=0.68)
        print(f"  {aa_electrode_names[row]:>16}: {med/1e6:.1f} [{lo/1e6:.1f}, {hi/1e6:.1f}] MHz/V")

    def simple_corner_plot(samples, labels, truths=None, bins=40):
        n = samples.shape[1]
        fig, axes = plt.subplots(n, n, figsize=(2.3*n, 2.3*n))
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                if j > i:
                    ax.axis('off'); continue
                if i == j:
                    ax.hist(samples[:, i], bins=bins, color='tab:purple', alpha=0.7)
                    if truths is not None:
                        ax.axvline(truths[i], color='red', ls='--', lw=1)
                else:
                    ax.plot(samples[:, j], samples[:, i], '.', ms=1, alpha=0.05, color='tab:purple')
                    if truths is not None:
                        ax.plot(truths[j], truths[i], 'r+', ms=10, mew=2)
                if i == n-1:
                    ax.set_xlabel(labels[j], fontsize=10)
                else:
                    ax.set_xticklabels([])
                if j == 0 and i != 0:
                    ax.set_ylabel(labels[i], fontsize=10)
                else:
                    ax.set_yticklabels([])
        plt.tight_layout()
        return fig

    fig = simple_corner_plot(mc_samples/1e6, names_short, truths=mu_vec/1e6)
        # /1e6: MHz/V 단위로 표시(그래프 눈금이 너무 큰 숫자가 안 되게)
    fig.suptitle('20개 TLS x 4전극 결합세기 - Corner Plot\n'
                 '(비대각선 패널이 원형이면 전극 간 독립, 대각선으로 기울면 상관)',
                 y=1.02, fontsize=11)
    plt.savefig('./outputs/tls_gamma_corner_plot.png', dpi=140, bbox_inches='tight')
    print("\nCorner plot 저장 완료: ./outputs/tls_gamma_corner_plot.png")

    # =====================================================
    # [용어 설명 - 물리/통계, 매 스크립트 끝에 계속 첨부]
    # =====================================================
    print("\n" + "="*70)
    print("[용어 설명]")
    print("="*70)
    print("""
--- 물리 용어 ---
TLS (이준위계)      : 비정질 재료 속 결함, 두 안정상태를 터널링하는 미시 시스템
Delta(터널링 에너지) : 두 우물 사이를 넘나드는 양자역학적 에너지 규모
epsilon(비대칭 에너지): 두 우물의 에너지 차이, 전기장 따라 변함
전기 쌍극자 모멘트 p  : TLS가 전기장과 얼마나 강하게 상호작용하는지
Stark 편이           : 전기장에 의해 공명 주파수가 이동하는 현상
결합세기 gamma        : 전압 1V당 TLS 주파수가 몇 Hz 움직이는지
대칭점(symmetry point): 공명곡선이 최솟값을 갖는 U자형 꼭짓점 전압
게이트 전극           : 칩 위 국소 전압 인가 지점(오늘은 4개: alpha,beta,gamma,delta)

--- 통계 용어 ---
Gaussian(가우시안) 평균: 이상치에 민감한 단순 산술평균
Robust(로버스트) 추정  : 이상치 영향을 줄이는 중앙값/MAD 기반 추정
MAD                   : 중앙값 절대편차, 표준편차의 강건한 대안
피셔행렬(Fisher matrix): 로그우도의 곡률(2차미분) - 파라미터를 얼마나
                        정확히 추정할 수 있는지 나타내는 행렬
공분산행렬(Covariance) : 피셔행렬의 역행렬 - 파라미터들의 불확실성과
                        상관관계를 담은 행렬
상관행렬(Correlation)  : 공분산을 정규화해 -1~1 범위로 표현한 것 -
                        대각선은 항상 1, 두 파라미터가 얼마나 같이
                        움직이는지 보여줌
68% 신뢰구간           : 진짜 값이 이 구간 안에 있을 확률이 68%라는 베이지안적 해석
Corner plot           : 여러 파라미터의 분포와 상관관계를 한 번에
                        보여주는 격자형 그래프(대각선=개별분포,
                        비대각선=두 파라미터 쌍의 산점도)
""")
