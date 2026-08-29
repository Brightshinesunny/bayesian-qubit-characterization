"""
compare_gaussian_vs_robust.py
=================================================================
같은 (outlier가 섞인) 데이터에 대해 gaussian 우도와 robust 우도로
각각 돌린 배치 MCMC 결과를 불러와서, "어느 쪽이 참값에 더 가깝게
추정했는가(정확도, accuracy)"를 정량적으로 비교합니다.

[왜 이 비교가 필요한가]
지금까지 본 'ok/unstable' 개수나 posterior 오차폭(불확실성, precision)만
비교해서는 robust 우도의 진짜 장점을 확인할 수 없습니다. 오차폭이
비슷하더라도, robust가 outlier의 영향을 덜 받아 "중심값 자체가 참값에
더 가깝게" 나왔다면 그게 진짜 강건함(robustness)의 증거입니다.
즉 여기서 비교하는 건 정밀도(precision, 얼마나 좁게 추정했는가)가
아니라 정확도(accuracy, 얼마나 참값에 가깝게 추정했는가)입니다.

사용법: example_run.py를 aa_likelihood_type='gaussian'으로 한 번,
       'robust'로 한 번 실행한 뒤(같은 aa_outlier_probability로),
       이 스크립트를 실행하면 두 결과 파일을 자동으로 찾아 비교합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# =========================================================
# STEP 0. 분석자 설정값
# =========================================================
aa_output_dir = './outputs/'
aa_outlier_probability = 0.01
    # example_run.py에서 실제로 썼던 값과 반드시 일치해야 같은 실험 조건의
    # 두 결과 파일(gaussian, robust)을 정확히 찾아낼 수 있음.

# =========================================================
# STEP 1. 두 결과 파일 로드
# =========================================================
gauss_path = os.path.join(aa_output_dir, f'results_gaussian_outlier{aa_outlier_probability}.npz')
robust_path = os.path.join(aa_output_dir, f'results_robust_outlier{aa_outlier_probability}.npz')

if not (os.path.exists(gauss_path) and os.path.exists(robust_path)):
    # glob: 와일드카드(*) 패턴으로 파일 목록을 찾는 함수. 정확한 파일명을
    # 못 찾았을 때, 어떤 결과 파일들이 실제로 있는지 힌트를 보여주기 위함.
    found = glob.glob(os.path.join(aa_output_dir, 'results_*.npz'))
    raise FileNotFoundError(
        f"필요한 두 파일을 찾을 수 없습니다.\n"
        f"  찾던 경로: {gauss_path}\n"
        f"           {robust_path}\n"
        f"  실제로 존재하는 결과 파일들: {found}\n"
        f"  -> example_run.py의 aa_likelihood_type을 'gaussian'과 'robust'로\n"
        f"     각각 한 번씩 실행했는지, aa_outlier_probability 값이 일치하는지 확인하세요."
    )

gauss_data = np.load(gauss_path, allow_pickle=True)
robust_data = np.load(robust_path, allow_pickle=True)

true_fr = float(gauss_data['true_fr'])
true_g = float(gauss_data['true_g'])
true_kappa = float(gauss_data['true_kappa'])
true_values = np.array([true_fr, true_g, true_kappa])
labels = [r'$f_r$', r'$g$', r'$\kappa$']

print(f"참값: fr={true_fr}, g={true_g}, kappa={true_kappa}")

# =========================================================
# STEP 2. 정확도(참값과의 오차) 계산
# =========================================================
def compute_accuracy_metrics(data, true_values):
    """
    각 파라미터(fr, g, kappa)별로, 모든 flux slice에 걸친 추정치가
    참값에서 평균적으로 얼마나 벗어났는지 계산.

    - bias(편향): (추정값 - 참값)의 평균. 부호가 있어서 "어느 방향으로
      치우쳤는지"까지 알 수 있음. 0에 가까울수록 좋음.
    - rms_error(제곱평균제곱근 오차): sqrt(mean((추정값-참값)^2)).
      부호와 무관하게 "평균적으로 얼마나 크게 틀렸는지"를 하나의 양수로
      요약. bias보다 큰 이상치(outlier)에 더 민감하게 반응하는 지표
      (제곱을 취하기 때문 - 큰 오차가 있으면 그 값이 두드러지게 커짐).
    """
    median = data['median']   # shape: (n_flux, 3)
    errors = median - true_values[np.newaxis, :]
        # true_values[np.newaxis,:] : (3,) -> (1,3)로 차원을 늘려서
        # (n_flux, 3) 배열의 모든 행에서 동시에 참값을 뺄 수 있게 브로드캐스팅

    bias = np.mean(errors, axis=0)          # axis=0: flux 방향(행)으로 평균 -> 파라미터별 결과 (3,)
    rms_error = np.sqrt(np.mean(errors ** 2, axis=0))

    return bias, rms_error, errors


gauss_bias, gauss_rms, gauss_errors = compute_accuracy_metrics(gauss_data, true_values)
robust_bias, robust_rms, robust_errors = compute_accuracy_metrics(robust_data, true_values)

# =========================================================
# STEP 3. 결과 출력
# =========================================================
print("\n" + "=" * 70)
print(f"{'파라미터':<10} {'Gaussian bias':>15} {'Robust bias':>15} "
      f"{'Gaussian RMS':>15} {'Robust RMS':>15}")
print("=" * 70)
for i, label in enumerate(labels):
    print(f"{label:<10} {gauss_bias[i]:>15.6f} {robust_bias[i]:>15.6f} "
          f"{gauss_rms[i]:>15.6f} {robust_rms[i]:>15.6f}")
print("=" * 70)

for i, label in enumerate(labels):
    winner = 'robust' if robust_rms[i] < gauss_rms[i] else 'gaussian'
    improvement = abs(gauss_rms[i] - robust_rms[i]) / gauss_rms[i] * 100
    print(f"{label}: RMS 오차 기준 {winner}가 더 정확함 "
          f"(차이 {improvement:.1f}%)")

# =========================================================
# STEP 4. 시각화 - flux별 오차 비교
# =========================================================
fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

flux = gauss_data['x']   # gaussian, robust 둘 다 같은 flux_grid를 썼다고 가정

for i, label in enumerate(labels):
    axes[i].axhline(0, color='black', lw=0.5)
        # 오차=0인 기준선. 이 선에 가까울수록 정확한 추정.
    axes[i].plot(flux, gauss_errors[:, i], 'o', ms=3, alpha=0.6,
                 color='tab:red', label='gaussian error')
    axes[i].plot(flux, robust_errors[:, i], 'o', ms=3, alpha=0.6,
                 color='tab:green', label='robust error')
    axes[i].set_ylabel(f'{label} 오차\n(추정값-참값)')

axes[0].legend()
axes[0].set_title(f'Gaussian vs Robust 우도: 참값 대비 오차 비교 '
                   f'(outlier_probability={aa_outlier_probability})')
axes[2].set_xlabel(r'Flux $\Phi/\Phi_0$')

plt.tight_layout()
plt.savefig(os.path.join(aa_output_dir, 'gaussian_vs_robust_accuracy.png'),
            dpi=150, bbox_inches='tight')
plt.show()

print(f"\n비교 그래프 저장 완료: "
      f"{os.path.join(aa_output_dir, 'gaussian_vs_robust_accuracy.png')}")
