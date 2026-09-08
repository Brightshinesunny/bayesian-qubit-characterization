"""
zenodo_mcmc_pipeline.py  (mcmc_pipeline.py를 Zenodo 실측 데이터 워크플로우 전용으로 복제한 버전)
=================================================================
emcee를 이용한 MCMC 샘플링 실행부. "거의 안 바뀌는 뼈대" 부분입니다.
새 물리계를 다루더라도, log_probability 함수만 새로 만들어(likelihood.py
참고) 이 파이프라인에 꽂으면 그대로 재사용할 수 있습니다.

포함된 기능:
  - 단일 데이터셋에 대한 MCMC 실행 + 수렴 진단
  - 여러 데이터셋(예: flux slice들)에 대한 warm-start 배치 실행
"""

import numpy as np    # 수치 배열 연산
import emcee          # 베이지안 MCMC 샘플링 전용 라이브러리
                       # ("affine-invariant ensemble sampler" 알고리즘 구현:
                       #  여러 개의 "walker"가 파라미터 공간을 동시에 탐색)
import warnings       # 파이썬의 경고 메시지 제어 (반복적인 수렴 경고를 숨기는 데 사용)


def run_single_mcmc(log_probability, initial_guess, f_grid, data_1d, sigma,
                      nwalkers=32, nsteps=2000, init_scatter=1e-3,
                      burn_in_discard=200, thin_by=15, bounds=None,
                      suppress_warnings=False, init_scatter_floor=0.0):
    """
    단일 데이터셋(예: 하나의 flux slice)에 대해 MCMC를 실행하고
    posterior 샘플과 수렴 진단 정보를 반환.

    Parameters
    ----------
    log_probability : likelihood.py의 make_log_probability로 만든 함수.
                       시그니처: log_probability(theta, f_grid, data_1d, sigma)
    initial_guess : 파라미터 초기 추정치 (예: [fr_guess, g_guess, kappa_guess])
    bounds : {'파라미터이름': (min,max)} 형태. 워커 초기 위치를 prior 범위
             안으로 clip(강제 이동)시킬 때 사용. None이면 clip 생략.

    Returns
    -------
    dict: {
      'flat_samples': posterior 샘플 (n_samples, ndim),
      'median': 각 파라미터의 중앙값,
      'err_lo', 'err_hi': 비대칭 오차,
      'autocorr_time': 자기상관 시간 추정치 (수렴 진단용),
      'converged': bool - 체인이 충분히 길었는지 여부
    }
    """
    ndim = len(initial_guess)   # 추정할 파라미터 개수 (예: fr,g,kappa면 3)

    scatter_amplitude = init_scatter * np.abs(initial_guess)
        # 기존 방식: "각 파라미터 값의 크기에 비례해서" 흩뿌림.
        # [버그 위험] 어떤 파라미터의 초기값이 정확히(또는 거의) 0이면
        # (예: 위상 오프셋 phi를 몰라서 0.0으로 시작하는 경우),
        # abs(0)=0 이라 그 파라미터에 대해서는 모든 워커가 "완전히
        # 동일한 값"에서 출발하게 됨. 이러면 emcee가 "워커들이 서로
        # 선형독립하지 않다(Initial state has a large condition
        # number)"는 에러를 내며 실행을 거부함 - 실제로 이 문제 때문에
        # 에러가 났던 사례가 있었음(phi 초기값을 0.0으로 뒀던 경우).
    scatter_amplitude = np.maximum(scatter_amplitude, init_scatter_floor)
        # init_scatter_floor: 파라미터 값과 무관하게 보장되는 "최소
        # 흩뿌림 폭"(절대적 스케일). 기본값 0.0이면 기존 동작과 완전히
        # 동일(하위 호환성 유지) - 값이 0이 아닐 위험이 있는 파라미터를
        # 다룰 때만 분석자가 명시적으로 이 값을 지정해 안전장치로 사용.
    pos = np.array(initial_guess) + scatter_amplitude * np.random.randn(nwalkers, ndim)
        # np.random.randn(nwalkers, ndim): 표준정규분포(평균0, 표준편차1)에서
        # (nwalkers x ndim) 크기의 난수 행렬을 뽑음.
        # 이렇게 만들면 nwalkers(예:32)개의 "탐사 에이전트"가 initial_guess
        # 주변에 살짝씩 무작위로 흩어진 시작점을 갖게 됨 (완전히 같은 지점에서
        # 출발하면 워커들이 서로 구분이 안 되어 탐색 다양성이 사라지므로 필요).

    if bounds is not None:
        for i, (lo, hi) in enumerate(bounds.values()):
            # enumerate: (순번, 값) 쌍을 동시에 꺼내는 파이썬 내장 함수.
            # bounds.values(): 딕셔너리에서 값들만(키는 빼고) 순서대로 꺼냄.
            pos[:, i] = np.clip(pos[:, i], lo + 1e-6, hi - 1e-6)
                # pos[:, i] : 2차원 배열에서 "모든 행, i번째 열"만 선택하는 슬라이싱
                # (즉 i번째 파라미터에 대한 모든 워커의 값들)
                # np.clip(값, 최소, 최대): 값이 범위를 벗어나면 경계값으로
                # 강제로 잘라내는(clip) 함수. 여기선 prior 범위 밖으로 워커가
                # 튀어나가 log_prior=-inf가 되어 "시작부터 죽는" 상황을 방지.
                # +1e-6/-1e-6: 정확히 경계값 위에 놓여서 부동소수점 오차로
                # 범위 밖 취급되는 것을 막기 위한 아주 작은 여유(margin).

    with warnings.catch_warnings():
        # with 구문: 이 블록 안에서만 특정 설정(여기선 경고 필터)이 적용되고,
        # 블록을 벗어나면 자동으로 원래 상태로 복원됨.
        if suppress_warnings:
            warnings.simplefilter("ignore")   # 이 블록 안의 모든 경고를 무시
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                          args=(f_grid, data_1d, sigma))
            # EnsembleSampler: emcee의 핵심 클래스. nwalkers개의 워커가
            # log_probability 함수(높을수록 "그럴듯한" 파라미터)를 따라
            # 파라미터 공간을 탐색하도록 설정.
            # args: log_probability에 theta 외에 추가로 넘길 고정 인자들
            # (f_grid, data_1d, sigma는 매 걸음마다 바뀌지 않는 데이터이므로
            #  여기서 한 번만 지정해두면 emcee가 알아서 매번 넘겨줌)
        sampler.run_mcmc(pos, nsteps, progress=not suppress_warnings)
            # run_mcmc: 실제로 워커들을 nsteps(걸음 수)만큼 움직이며 샘플링 수행.
            # progress: 진행 상황을 프로그레스바로 표시할지 여부.

    flat_samples = sampler.get_chain(discard=burn_in_discard, thin=thin_by, flat=True)
        # get_chain: MCMC로 뽑힌 모든 샘플(파라미터 후보값들)을 가져옴.
        # discard: burn-in(체인이 아직 초기값 영향을 강하게 받는 초반 구간) 제거.
        # thin: 연속 샘플끼리의 자기상관(autocorrelation, 서로 닮음)을 줄이기 위해
        #        몇 개마다 하나씩만 골라 쓰는 간격.
        # flat=True: (걸음수, 워커수, 파라미터수)인 3차원 배열을
        #            (샘플수, 파라미터수)인 2차원으로 평평하게 펼침.

    result = {'flat_samples': flat_samples}

    if flat_samples.shape[0] < 10:
        # 남은 샘플이 너무 적다는 건, 대부분의 워커가 prior 범위 밖으로
        # 나가 죽었거나(log_prob=-inf) 체인이 발산했다는 신호.
        result['median'] = np.array(initial_guess)
        result['err_lo'] = np.zeros(ndim)
        result['err_hi'] = np.zeros(ndim)
        result['converged'] = False
    else:
        pct = np.percentile(flat_samples, [16, 50, 84], axis=0)
            # np.percentile: 데이터의 특정 백분위수 값을 계산하는 함수.
            # 16/50/84 백분위수는 가우시안 분포의 "평균 ± 1 표준편차" 구간에
            # 해당하는 값으로, posterior의 중앙값과 비대칭 오차를 표현할 때
            # 관례적으로 널리 쓰이는 조합.
            # axis=0: 각 파라미터(열)별로 따로 계산 (행 방향으로 통계 집계)
        result['median'] = pct[1]                 # 중앙값(median) - 대표 추정치
        result['err_lo'] = pct[1] - pct[0]         # 하한 오차 (median - 16th)
        result['err_hi'] = pct[2] - pct[1]         # 상한 오차 (84th - median)
        result['converged'] = True

    try:
        result['autocorr_time'] = sampler.get_autocorr_time(quiet=True)
            # get_autocorr_time: 체인이 "서로 독립적인 정보"를 얼마나
            # 자주 만들어내는지 추정하는 값(자기상관 시간). 이 값의 약 50배
            # 이상 체인 길이가 있어야 신뢰할 만한 pos터리어로 여겨짐(경험적 기준).
            # quiet=True: 체인이 너무 짧아 추정이 부정확할 때도 에러 대신
            # 그냥 추정치를 반환하게 함 (기본값은 에러를 냄).
    except Exception:
        result['autocorr_time'] = None

    # [베이지안 도구 연동용 추가 정보]
    # bayesian_toolkit.py의 진단/모델비교 함수들을 쓰려면 posterior 요약값
    # (median, err) 외에 "원본 체인"과 "최대 로그우도"가 추가로 필요함.
    result['raw_chain'] = sampler.get_chain(discard=burn_in_discard, thin=thin_by, flat=False)
        # flat=False: (걸음수, 워커수, 파라미터수) 형태의 3차원 배열을 그대로 유지.
        # bayesian_toolkit.gelman_rubin_rhat()은 "여러 개의 독립된 체인"이 필요한데,
        # emcee의 각 워커를 "하나의 독립 체인"으로 간주해 쓸 수 있음
        # (엄밀히는 워커들이 서로 완전히 독립은 아니지만, 실전에서 흔히 쓰는 근사).
    try:
        log_prob_values = sampler.get_log_prob(discard=burn_in_discard, thin=thin_by, flat=True)
            # get_log_prob: 각 샘플에서의 log_probability(posterior) 값을 가져옴.
            # AIC/BIC는 원래 log_likelihood(우도)의 최댓값이 필요하지만, prior가
            # 균일분포(uniform)라면 prior 범위 안에서 log_prior가 상수이므로
            # log_probability의 최댓값이 곧 log_likelihood 최댓값과 사실상 같음.
        result['max_log_prob'] = np.max(log_prob_values)
    except Exception:
        result['max_log_prob'] = None

    return result


def run_batch_mcmc_warmstart(log_probability_factory, x_values, data_2d, sigma_estimator,
                                initial_guess_func, f_grid,
                                nwalkers=32, nsteps_per_point=800,
                                init_scatter=5e-3, burn_in_discard=200, thin_by=10,
                                bounds=None,
                                unstable_threshold=None,
                                progress_every=10):
    """
    여러 데이터셋(예: 101개 flux slice)에 대해 순차적으로 MCMC를 실행하며,
    이전 지점의 결과를 다음 지점의 초기값(warm-start)으로 재사용.

    warm-start를 쓰는 이유: 물리적으로 인접한 조건(예: 비슷한 flux 값)
    끼리는 파라미터가 급격히 안 바뀐다는 사실을 이용해, 매번 처음부터
    데이터를 훑어 초기값을 찾는 대신 이전 결과를 재사용함으로써 수렴을
    빠르고 안정적으로 만드는 기법.

    Parameters
    ----------
    log_probability_factory : x_val을 받아 그 지점 전용 log_probability
        함수를 반환하는 함수. (avoided-crossing 모델처럼 flux_val이 모델
        자체에 들어가는 경우, 각 slice마다 다른 log_probability가 필요하기
        때문에 "factory 함수"로 설계함)
    x_values : 스캔할 조건들의 배열 (예: flux 값들)
    data_2d : (len(x_values), n_freq) 형태의 복소수 데이터
    sigma_estimator : 1D 데이터를 받아 noise_sigma를 추정하는 함수
    initial_guess_func : 첫 데이터 포인트에서만 쓰이는, 데이터를 보고
        초기값을 추정하는 함수. 이후 지점들은 warm-start로 대체됨.
    unstable_threshold : {'파라미터인덱스': 임계값} 형태. posterior 오차폭이
        이 값을 넘으면 status='unstable'로 표시. None이면 판정 생략.

    Returns
    -------
    dict: 각 x_value에 대한 median, err_lo, err_hi, status 등을 담은 배열들
    """
    n_points = len(x_values)
    results = {'x': [], 'median': [], 'err_lo': [], 'err_hi': [], 'status': []}
    current_guess = None   # warm-start용 변수: 직전 지점의 피팅 결과를 저장해두는 곳
                            # (첫 지점은 이 값이 없으므로 데이터 기반 추정으로 시작)

    for count, x_val in enumerate(x_values):
        # enumerate: (순번, 값) 쌍을 동시에 꺼내는 파이썬 내장 함수
        data_1d = data_2d[count, :]   # 2D 배열에서 count번째 행(row) 전체를 슬라이싱
        sigma = sigma_estimator(data_1d)

        # 초기값 결정: 첫 지점은 데이터에서 직접 추정, 이후는 warm-start(이전 결과 재사용)
        if current_guess is None:
            initial_guess = initial_guess_func(f_grid, data_1d)
        else:
            initial_guess = current_guess.copy()
                # .copy(): 배열을 복사해서 새 객체를 만듦. 복사하지 않고 그대로
                # 참조하면, 이후 run_single_mcmc 내부에서 이 배열이 의도치
                # 않게 변형될 경우 current_guess까지 같이 바뀌는 버그가 생길 수 있음.

        log_prob = log_probability_factory(x_val)
            # 이 x_val(예: 이번 flux 값) 전용 log_probability 함수를 새로 만듦.
            # avoided-crossing 모델처럼 flux_val이 모델식 안에 직접 들어가는
            # 경우, slice마다 "고정 인자가 다른" 별도의 log_probability가 필요함.

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
                # 101번 반복 실행하는 배치 작업이라, emcee의 반복적인 수렴
                # 경고가 출력을 어지럽히지 않도록 배치 처리 중에는 숨김.
            single_result = run_single_mcmc(
                log_prob, initial_guess, f_grid, data_1d, sigma,
                nwalkers=nwalkers, nsteps=nsteps_per_point,
                init_scatter=init_scatter, burn_in_discard=burn_in_discard,
                thin_by=thin_by, bounds=bounds, suppress_warnings=True
            )

        status = 'ok' if single_result['converged'] else 'failed'
        if unstable_threshold is not None and single_result['converged']:
            for idx, thresh in unstable_threshold.items():
                # .items(): 딕셔너리에서 (키, 값) 쌍을 동시에 꺼내는 메서드.
                # 여기선 {파라미터인덱스: 임계값} 딕셔너리를 순회.
                if single_result['err_hi'][idx] > thresh:
                    # posterior 오차폭이 임계값을 넘으면 "불안정"으로 판정.
                    # 물리적으로: 디튜닝이 커서 avoided-crossing 봉우리 하나가
                    # 거의 사라지는 slice 등에서 이런 상황이 생김.
                    status = 'unstable'
                    break   # break: 여러 파라미터 중 하나라도 불안정하면
                            # 더 볼 것 없이 반복문을 즉시 빠져나옴

        results['x'].append(x_val)
        results['median'].append(single_result['median'])
        results['err_lo'].append(single_result['err_lo'])
        results['err_hi'].append(single_result['err_hi'])
        results['status'].append(status)

        # 다음 지점의 warm-start 초기값 갱신. 실패(failed)한 결과를 다음
        # 시작점으로 물려주면 연쇄적으로 계속 실패할 위험이 있으므로,
        # 수렴에 성공했을 때만 갱신함.
        if single_result['converged']:
            current_guess = single_result['median']

        if (count + 1) % progress_every == 0 or count == 0:
            # % (나머지 연산자): progress_every(예:10)번째마다 한 번씩만
            # 진행상황을 출력해 로그가 너무 길어지지 않게 함.
            print(f"  [{count+1:3d}/{n_points}] x={x_val:+.4f}  "
                  f"median={np.round(single_result['median'], 4)}  status={status}")

    for k in ['x', 'median', 'err_lo', 'err_hi']:
        results[k] = np.array(results[k])
            # 파이썬 리스트로 쌓아온 결과들을 numpy 배열로 일괄 변환
            # (numpy 배열이어야 이후 슬라이싱, 통계 계산 등이 편해짐)
    results['status'] = np.array(results['status'], dtype=object)
        # dtype=object: 'ok'/'unstable'/'failed' 같은 문자열(숫자가 아닌 값)을
        # 담기 위한 배열 타입 지정. 기본 dtype으로는 문자열을 담을 수 없음.

    return results


if __name__ == "__main__":
    # models.py, mock_data.py와 동일한 패턴의 자가진단 코드.
    # 특히 이 파일은 파라미터(함수 시그니처)가 추가되는 식으로 자주
    # 바뀌므로, "함수가 있는지"뿐 아니라 "새로 추가된 인자까지 있는지"도
    # 확인함 - 예: init_scatter_floor 인자가 없다면 구버전이 아직도
    # import되어 캐시되고 있다는 뜻 (Colab에서 파일만 새로 올리고
    # 런타임을 재시작하지 않으면 이런 상황이 흔히 발생함).
    import inspect
    print("mcmc_pipeline.py 자가진단")
    print("-" * 55)
    sig = inspect.signature(run_single_mcmc)
        # inspect.signature: 함수가 어떤 인자들을 받는지(이름, 기본값 등)
        # 실행 없이 코드 구조만 보고 알아내는 도구.
    has_floor = 'init_scatter_floor' in sig.parameters
    print(f"  run_single_mcmc 존재            : OK")
    print(f"  init_scatter_floor 인자 존재     : "
          f"{'OK' if has_floor else '누락!! -> 구버전 파일일 수 있음'}")
    print(f"  run_batch_mcmc_warmstart 존재    : "
          f"{'OK' if 'run_batch_mcmc_warmstart' in dir() else '누락!!'}")
    print("-" * 55)
    if has_floor:
        print("전체 상태: 정상 (최신 버전)")
    else:
        print("전체 상태: 구버전으로 보입니다.")
        print("  -> Colab에서: 이 파일을 새로 업로드한 뒤 '런타임 재시작'을")
        print("     하거나, `import importlib; importlib.reload(mcmc_pipeline)`")
        print("     을 실행해 최신 버전을 다시 불러오세요.")
