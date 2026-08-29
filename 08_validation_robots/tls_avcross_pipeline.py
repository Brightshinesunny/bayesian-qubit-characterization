"""
tls_avcross_pipeline.py
=================================================================
[D단계 - 오늘 정리 + 복습] 오늘 여러 스크립트(fit_seg12.py 초안들)에
흩어져 있던 로직을, 다른 segment에도 바로 재사용 가능한 "함수 모음"
으로 정리했습니다. 오늘 발견한 버그 수정(V0 재중심화, 단위통일,
prior 범위, MCMC 어댑터, 물리적 bounds)이 전부 반영되어 있습니다.

[오늘 최종 결론 - 반드시 먼저 읽을 것]
seg12(raw_index=12)의 실측 히트맵을 육안으로 확인한 결과, 어두운
구조는 "전압에 따라 이동하는 대각선"이 아니라 "전압과 거의 무관한
수평 띠 여러 개"(대략 5.09, 5.14, 5.21GHz 근처)였습니다. 즉 오늘
설계한 avoided-crossing 2D 모델(TLS가 Stark 편이로 대각선을 그린다는
가정)이 이 segment에는 애초에 안 맞는 가설이었을 가능성이 높습니다.
이 파일의 함수들은 그래도 "다른 segment에서는 진짜 대각선이 보일
수도 있다"는 가능성 때문에 계속 재사용 가치가 있고, 파일 끝에는
"수평 다중선 모델"이라는 대안 가설도 확장 지점으로 남겨뒀습니다.
"""

import h5py
import numpy as np
from scipy.optimize import curve_fit
import sys, os
sys.path.insert(0, os.getcwd())
import tls_avcross_models as models
import tls_avcross_likelihood as likelihood


# =========================================================
# 함수 1. 데이터 로드
# =========================================================
def mat_string(f, dataset_or_ref):
    """MATLAB uint16 코드 배열을 파이썬 문자열로 변환."""
    try:
        if isinstance(dataset_or_ref, h5py.Reference):
            dataset_or_ref = f[dataset_or_ref]
        arr = np.array(dataset_or_ref)
        return ''.join(chr(int(c)) for c in arr.flatten())
    except Exception:
        return None


def load_tls_segment_2d_map(mat_filepath, raw_index, observable='dispamp'):
    """
    지정한 raw_index(segment)의 2D 관측 지도를 로드하고, calibdat로
    플럭스를 실제 큐빗 주파수(GHz)로 변환해서 반환.

    [오늘 확립한 단위 규칙] calibdat.qubitfreq는 Hz 단위로 저장되어
    있으므로, 1e9로 나눠 GHz로 통일 - 모델/prior가 전부 GHz 기준으로
    설계되어 있음.

    Returns
    -------
    dict: {sweep1vals(전압,21개), fq_axis(주파수GHz,401개), data_2d,
           electrode(전극이름)}
    """
    with h5py.File(mat_filepath, 'r') as f:
        qdat_group = f[f['qdats']['qdat'][raw_index, 0]]
        sweep1vals = np.array(qdat_group['sweep1vals']).flatten()
        sweep2vals = np.array(qdat_group['sweep2vals']).flatten()
        electrode = mat_string(f, qdat_group['sweep1name'])

        calib_field = np.array(f['qset']['calibdat']['field']).flatten()
        calib_freq_hz = np.array(f['qset']['calibdat']['qubitfreq']).flatten()
        order = np.argsort(calib_field)
        fq_axis = np.interp(sweep2vals, calib_field[order], calib_freq_hz[order]) / 1e9

        obs_val_refs = qdat_group['obs']['vals']
        obs_name_refs = qdat_group['observables']
        target_col = None
        for i in range(obs_name_refs.shape[0]):
            if mat_string(f, obs_name_refs[i, 0]) == observable:
                target_col = i
                break
        data_2d = np.array(f[obs_val_refs[target_col, 0]])

    return {'sweep1vals': sweep1vals, 'sweep2vals': sweep2vals,
            'fq_axis': fq_axis, 'data_2d': data_2d, 'electrode': electrode}


# =========================================================
# 함수 2. 계통오차 마스킹
# =========================================================
def mask_known_artifacts(sweep2vals, fq_axis,
                            flux_centers=(0.045, -0.030, -0.065), flux_halfwidth=0.006,
                            freq_center=5.168, freq_halfwidth=0.002,
                            freq_window=None):
    """
    오늘 확립한 3단계 마스킹(플럭스 기준 3개 + 주파수 기준 1개 +
    선택적 창 축소)을 적용해 "탐색 가능한 행" 불리언 마스크를 반환.

    freq_window : (lo, hi) 또는 None. 특정 범위로 추가 제한하고 싶을
                  때 사용(오늘은 육안 확인용으로 5.13~5.19를 썼지만,
                  이번 결론(수평 띠 구조) 이후로는 굳이 좁힐 필요가
                  없을 수도 있음 - 상황에 맞게 판단).
    """
    row_ok = np.ones(len(sweep2vals), dtype=bool)
    for center in flux_centers:
        row_ok &= (np.abs(sweep2vals - center) > flux_halfwidth)
    row_ok &= (np.abs(fq_axis - freq_center) > freq_halfwidth)
    if freq_window is not None:
        row_ok &= (fq_axis >= freq_window[0]) & (fq_axis <= freq_window[1])
    return row_ok


# =========================================================
# 함수 3. multi-start curve_fit (avoided-crossing 2D 모델)
# =========================================================
def fit_avoided_crossing_multistart(V_grid, fq_grid, data_2d, V0,
                                       gamma_candidates=(-0.109, -0.05, 0.0, 0.05, 0.109),
                                       f_TLS0_candidates=(5.10, 5.15, 5.20),
                                       g_upper_bound=0.03, verbose=True):
    """
    오늘 확립한 multi-start curve_fit. 여러 (f_TLS0, gamma) 초기값
    조합을 전부 시도하고 chi2가 가장 낮은 걸 채택.

    [오늘 교훈] gamma_stark 부호를 모르므로 +/- 후보를 다 넣고,
    g 상한은 물리적으로 타당한 값(30MHz 근처)으로 조여야 국소최적점
    문제가 줄어듦.

    Returns
    -------
    (best_popt, best_chi2)
    """
    def model_wrapper(xy_flat, f_TLS0, gamma_stark, g, baseline, contrast):
        n = V_grid.size
        V_flat, fq_flat = xy_flat[:n], xy_flat[n:]
        V_2d = V_flat.reshape(V_grid.shape)
        fq_2d = fq_flat.reshape(fq_grid.shape)
        return models.tls_swap_spectroscopy_2d_model(
            V_2d, fq_2d, f_TLS0, gamma_stark, g, baseline, contrast, V0=V0
        ).flatten()

    xy_flat = np.concatenate([V_grid.flatten(), fq_grid.flatten()])
    lower_bounds = [fq_grid.min(), -0.2, 1e-6, data_2d.min()-1, -5.0]
    upper_bounds = [fq_grid.max(), 0.2, g_upper_bound, data_2d.max()+1, 5.0]

    best_popt, best_chi2 = None, np.inf
    for f0 in f_TLS0_candidates:
        for gc in gamma_candidates:
            p0 = [f0, gc, 0.001, np.median(data_2d), (data_2d.max()-data_2d.min())/2]
            try:
                popt, _ = curve_fit(model_wrapper, xy_flat, data_2d.flatten(), p0=p0,
                                      bounds=(lower_bounds, upper_bounds), maxfev=30000)
                pred = model_wrapper(xy_flat, *popt).reshape(data_2d.shape)
                chi2 = np.sum((data_2d - pred)**2)
                if verbose:
                    print(f"  p0(f_TLS0={f0:.2f}, gamma={gc:+.3f}) -> chi2={chi2:.4f}, "
                          f"수렴: gamma={popt[1]:.5f}, g={popt[2]:.5f}")
                if chi2 < best_chi2:
                    best_chi2, best_popt = chi2, popt
            except Exception as e:
                if verbose:
                    print(f"  p0(f_TLS0={f0:.2f}, gamma={gc:+.3f}) -> 실패({e})")

    return best_popt, best_chi2


# =========================================================
# 함수 4. MCMC 실행 (emcee 필요 - Colab 전용)
# =========================================================
def make_mcmc_ready_log_probability(V0, prior_bounds, param_names=None):
    """
    2D avoided-crossing 모델용 log_probability를 만들고, emcee의
    "f_grid 인자 하나" 관례에 맞춰 튜플 언패킹 어댑터까지 포함해서
    반환. mcmc_pipeline.run_single_mcmc에 바로 넘길 수 있음.
    """
    if param_names is None:
        param_names = ['f_TLS0', 'gamma_stark', 'g', 'baseline', 'contrast']
    log_prior = likelihood.make_uniform_log_prior(prior_bounds)
    log_probability_2d = likelihood.make_log_probability_2d_real(
        models.tls_swap_spectroscopy_2d_model, {'V0': V0}, param_names, log_prior,
        likelihood_type='gaussian'
    )

    def log_probability(theta, xy_tuple, data_2d_arg, sigma):
        x_grid, y_grid = xy_tuple
        return log_probability_2d(theta, x_grid, y_grid, data_2d_arg, sigma)

    return log_probability


# =========================================================
# [확장 지점 - 오늘 육안 검증에서 나온 새 가설]
# =========================================================
def multi_horizontal_lines_model(V_grid, fq_grid, freq_centers, widths, depths, baseline):
    """
    [신규 후보 모델 - 아직 검증 안 됨, 다음 세션 과제]
    오늘 seg12를 육안으로 보니, 어두운 구조가 "전압에 따라 이동하는
    대각선"이 아니라 "전압과 거의 무관한 여러 개의 수평 띠"였습니다
    (대략 5.09, 5.14, 5.21GHz 근처). 이 관측과 훨씬 잘 맞을 대안
    모델을 미리 만들어둡니다: 각 띠를 "전압과 무관한 로렌츠 딥"으로
    보고, 여러 개를 단순히 더하는 형태.

    freq_centers, widths, depths : 각각 길이 N(띠 개수)짜리 배열.
      한 띠의 중심주파수/폭/깊이.

    [주의] 아직 실측 데이터로 검증 안 됨 - 다음 세션에서 이 함수로
    fit_avoided_crossing_multistart와 비슷한 절차를 다시 밟아야 함.
    """
    signal = np.full(V_grid.shape, baseline, dtype=float)
    for fc, w, d in zip(freq_centers, widths, depths):
        signal -= d / (1.0 + ((fq_grid - fc) / (w/2.0))**2)
            # 표준 로렌츠(Lorentzian) 형태: fc에서 최대로 어두워지고
            # (d만큼), fc에서 멀어질수록 baseline으로 돌아옴. 전압
            # V_grid는 이 식에 아예 등장하지 않음 - "전압과 무관"이라는
            # 오늘 관측을 그대로 식에 반영한 것.
    return signal


# =========================================================
# 사용 예시 (seg12 재현) - Colab에서 emcee 있어야 STEP 4까지 실행 가능
# =========================================================
if __name__ == "__main__":
    aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
    aa_raw_index = 12

    seg = load_tls_segment_2d_map(aa_mat_filepath, aa_raw_index)
    print(f"전극={seg['electrode']}, 전압범위=[{seg['sweep1vals'].min():.2f},"
          f"{seg['sweep1vals'].max():.2f}]V")

    row_ok = mask_known_artifacts(seg['sweep2vals'], seg['fq_axis'])
    V_grid, fq_grid = np.meshgrid(seg['sweep1vals'], seg['fq_axis'])
    V_grid, fq_grid = V_grid[row_ok, :], fq_grid[row_ok, :]
    data_2d = seg['data_2d'][row_ok, :]
    V0 = float(np.median(seg['sweep1vals']))

    popt, chi2 = fit_avoided_crossing_multistart(V_grid, fq_grid, data_2d, V0)
    print(f"\n최종 채택 (chi2={chi2:.4f}):", dict(zip(
        ['f_TLS0','gamma_stark','g','baseline','contrast'], popt)))

    print("\n[오늘 결론] 이 결과를 MCMC로 검증한 결과와 육안 검증(실측")
    print("히트맵 위에 겹쳐그리기)을 반드시 함께 봐야 함 - 숫자만으로는")
    print("이 모델이 실제 구조를 설명하는지 판단 불가능하다는 게 오늘의 교훈.")


# =========================================================
# [용어 정리 - 물리 + 통계, 매번 첨부]
# =========================================================
"""
--- 물리 용어 ---
avoided-crossing(회피교차) : 두 시스템의 에너지 준위가 직접 만나지
    않고 밀어내며 갈라지는 현상.
Stark 편이                 : 전압(전기장)에 의해 공명주파수가 이동.
결합세기 gamma_stark(MHz/V) : 전압 1V당 TLS 주파수 이동량.
결합강도 g(GHz)             : 큐빗-TLS 에너지 교환 속도.
계통오차(systematic error) : 조건(전극/전압/관측량)을 바꿔도 반복
    적으로 같은 자리에 나타나는 오차. 통계적 잡음과 달리 평균 내도
    안 사라짐.

--- 통계 용어 ---
V0 재중심화       : 절편 파라미터의 기준점을 측정구간 중앙으로 옮겨
    다른 파라미터와의 상관관계를 없애는 표준 기법.
multi-start       : 여러 초기값에서 최적화해 국소최적점 문제를
    완화하는 기법.
posterior 다중봉우리(multi-modal) : 그럴듯한 답이 여러 개 동시에
    존재하는 상태 - 점추정으로는 발견 불가, MCMC로 전체 분포를
    봐야 판단 가능.
prior 범위 버그   : 사전분포 설정이 실제 관심 파라미터 값을 배제하면,
    MCMC가 "수렴했다"고 나와도 그 결론 자체가 무효가 될 수 있음.
육안 검증(visual overlay check) : 숫자만으로 판단하기 어려운 모델의
    타당성을, 실측 데이터 위에 겹쳐 그려 직접 확인하는 절차 - 오늘
    최종 판정에 결정적 역할을 함.
"""
