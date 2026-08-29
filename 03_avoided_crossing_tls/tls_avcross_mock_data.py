"""
mock_data.py
=================================================================
Mock(모사) 데이터를 만드는 모듈. 두 부분으로 나뉩니다:

  1. clean 신호 생성 (물리 모델을 그대로 계산 - models.py 사용)
  2. 잡음 주입 (백색잡음 / 1/f 잡음 / 캘리브레이션 드리프트 / outlier /
     baseline 기울어짐 / κ 시간요동)

[이 파일에 포함된 함수 목록 - 새로 업로드할 때 아래 목록으로 버전 확인]
  1. add_white_noise()            - 백색잡음 추가
  2. generate_1f_noise_1d()       - 1/f(플리커) 잡음 1차원 생성
  3. add_flicker_noise()          - 1/f 잡음을 2D 데이터에 적용
  4. add_calibration_drift()      - 캘리브레이션 드리프트(flux 방향)
  5. add_baseline_tilt()          - baseline 기울어짐(frequency 방향)
  6. generate_fluctuating_kappa() - κ의 시간적 요동 시계열 생성
  7. add_outliers()               - outlier(이상치) 주입
  8. make_realistic_2d_dataset()  - 위 잡음들을 한 번에 조합하는 통합 함수
  (Colab에서 `import mock_data; print(dir(mock_data))`로 이 8개 함수가
   모두 보이는지 확인하면, 구버전 파일을 잘못 쓰고 있는 실수를 예방할 수 있음)

[템플릿 설계 원칙]
잡음 주입 함수들은 "어떤 물리 모델을 쓰든 상관없이" 재사용 가능하도록
설계했습니다. 즉 clean_signal이라는 복소수 배열만 주면, 그게 avoided-crossing
모델이든 다른 모델이든 똑같이 잡음을 입힐 수 있습니다.

왜 mock 데이터를 계속 만드는가 (실전에서의 위치):
  - 실측 데이터는 참값을 모르기 때문에, "내 분석 코드가 맞는지" 검증할
    유일한 방법이 "참값을 미리 정해놓고 만든 mock 데이터로 그 참값이
    복원되는지 확인하는 것"입니다.
  - 새로운 모델/방법(robust likelihood, 계층적 모델 등)을 시도할 때마다
    그에 맞는 mock 데이터를 새로 만들어 검증하는 것이 표준적인 워크플로우입니다.
"""

import numpy as np


# =========================================================
# 1. 백색잡음 (white noise)
# =========================================================
def add_white_noise(clean_signal, level, rng=None):
    """
    가우시안 백색잡음(white Gaussian noise)을 더함.
    "백색"이라는 이름은 모든 주파수 성분의 잡음 세기가 균일하다는 뜻
    (빛의 스펙트럼이 모든 파장에서 고르게 섞이면 흰색으로 보이는 것에서 유래).
    가장 기본적이고 이상적인 잡음 모델 (열잡음, thermal noise를 근사).

    Parameters
    ----------
    clean_signal : 복소수 배열. 잡음을 더할 원본 신호.
    level : float. 잡음의 표준편차 크기.
    rng : np.random.Generator 또는 None. 재현성 있는 난수 생성을 위해
          외부에서 시드가 고정된 generator를 넘길 수 있음. None이면
          numpy 전역 난수 상태를 사용.
    """
    if rng is None:
        rng = np.random
    noise_real = rng.normal(0, level, clean_signal.shape)
    noise_imag = rng.normal(0, level, clean_signal.shape)
    return clean_signal + noise_real + 1j * noise_imag


# =========================================================
# 2. 1/f 잡음 (flicker noise)
# =========================================================
def generate_1f_noise_1d(n_points, level, rng=None):
    """
    1/f 잡음(플리커 노이즈) 1차원 배열을 생성.

    원리: 백색잡음을 FFT(고속푸리에변환)로 주파수 영역에 옮긴 뒤, 각
    주파수 성분의 진폭을 1/sqrt(frequency)로 스케일링하면 (파워 스펙트럼
    기준 1/f), 저주파(천천히 변하는) 성분이 강조된 "출렁이는" 잡음이 됨.
    HEMT 저잡음 증폭기 등 실제 전자회로에서 흔히 나타나는 잡음 특성.
    """
    if rng is None:
        rng = np.random
    white = rng.normal(0, 1, n_points)
    freqs = np.fft.rfftfreq(n_points)   # 실수 신호에 대응하는 FFT 주파수 축
    freqs[0] = freqs[1]                 # DC(0주파수) 성분의 0-나누기 방지
    spectrum = np.fft.rfft(white) / np.sqrt(freqs)
    colored = np.fft.irfft(spectrum, n=n_points)   # 역FFT로 다시 원래 축으로
    colored = colored / np.std(colored) * level     # 목표 세기로 정규화
    return colored


def add_flicker_noise(clean_signal_2d, level, rng=None):
    """
    2D 배열(예: flux x frequency)의 각 행(row)마다 독립적으로 1/f 잡음을
    생성해 더함. clean_signal_2d의 shape이 (n_rows, n_freq)라고 가정.
    """
    n_rows, n_freq = clean_signal_2d.shape
    noise_real = np.array([generate_1f_noise_1d(n_freq, level, rng) for _ in range(n_rows)])
    noise_imag = np.array([generate_1f_noise_1d(n_freq, level, rng) for _ in range(n_rows)])
    return clean_signal_2d + noise_real + 1j * noise_imag


# =========================================================
# 3. 캘리브레이션 드리프트
# =========================================================
def add_calibration_drift(clean_signal_2d, amplitude, period_fraction):
    """
    측정이 진행되는 동안(2D 데이터의 행 방향, 예: flux slice 순서) 배경
    신호가 서서히 흔들리는 현상을 시뮬레이션. 실제로는 냉동기 온도 변화,
    증폭기 이득 변화 등이 원인.

    amplitude : 드리프트의 최대 진폭
    period_fraction : 전체 스캔 동안 sine 곡선이 몇 주기 도는지 (0.5~1
                       정도면 "서서히 한 방향으로 흘러갔다 오는" 형태)
    """
    n_rows = clean_signal_2d.shape[0]
        # .shape: numpy 배열의 각 차원 크기를 담은 튜플. [0]은 첫 번째 차원
        # (여기서는 flux slice 개수, 즉 "행"의 개수)
    progress = np.arange(n_rows) / n_rows   # 0~1로 정규화된 진행률
        # np.arange(n): 0,1,2,...,n-1의 정수 배열을 만드는 함수.
        # n_rows로 나누면 0~1 사이 값이 되어, "몇 번째 slice인지"를
        # "전체 스캔의 몇 %가 진행됐는지"로 바꿔줌.
    drift_curve = amplitude * np.sin(2 * np.pi * period_fraction * progress)
        # sine 곡선: 시간이 지남에 따라 부드럽게 오르내리는 형태를 표현.
        # period_fraction이 작을수록(예:0.3) 전체 스캔 동안 절반 주기도 못
        # 돌아 "한쪽으로 서서히 흘러가는" 모양이 되고, 클수록(예:2) 여러 번
        # 오르내리는 형태가 됨.
    drift_2d = drift_curve[:, np.newaxis]   # (n_rows,) -> (n_rows, 1) 브로드캐스팅용
        # np.newaxis: 배열에 새로운 축(차원)을 추가하는 인덱싱 트릭.
        # 1차원 배열(n_rows,)을 2차원(n_rows, 1)으로 바꿔서, 이후
        # (n_rows, n_freq) 크기의 2D 신호 배열과 "브로드캐스팅"(크기가
        # 다른 배열끼리 numpy가 자동으로 크기를 맞춰 연산해주는 규칙)이
        # 되도록 함 -> 각 행(slice) 전체에 같은 드리프트 값을 더할 수 있게 됨.
    return clean_signal_2d + drift_2d, drift_2d


# =========================================================
# 3.5 배경(baseline) 기울어짐 - 케이블/증폭기 대역 특성으로 인한 왜곡
# =========================================================
def add_baseline_tilt(signal_2d, f_grid, tilt_slope):
    """
    스캔 주파수에 따라 배경(baseline) 진폭이 서서히 기울어지는 현상을
    반영. 이론상 |S21|의 baseline(공진에서 먼 구간)은 정확히 1이어야
    하지만, 실제로는 증폭기 이득이 주파수에 따라 완전히 평평하지 않고
    (증폭기의 대역폭 특성), 케이블 손실도 주파수에 따라 조금씩 달라지므로
    baseline이 완전한 수평선이 아니라 살짝 기울어진 형태로 나타나는
    경우가 흔합니다.

    이게 왜 문제가 되는가 (diagnostics.py와의 연결):
    diagnostics.estimate_noise_sigma의 'adaptive' 방식은 baseline이
    "평평하다"는 암묵적 가정 위에서 표준편차를 계산합니다. 만약 baseline
    자체가 기울어져 있다면, 그 "기울기로 인한 변화"까지 표준편차에
    섞여 들어가 noise_sigma가 실제보다 과대추정될 수 있습니다. 즉 이
    함수로 만든 데이터를 기존 파이프라인에 넣어보면, "잡음 추정이
    부정확해지는 상황"을 직접 겪어볼 수 있습니다.

    tilt_slope : 스캔 주파수 전체 폭에 걸쳐 baseline이 얼마나
                 기울어지는지 (예: 0.05면 스캔 시작~끝 사이에
                 baseline 진폭이 대략 5% 정도 선형으로 변함)
    """
    f_normalized = (f_grid - f_grid.mean()) / (f_grid[-1] - f_grid[0])
        # 주파수 축을 "중심을 0으로, 전체 폭을 1로" 정규화.
        # 이렇게 하면 tilt_slope 값이 "스캔 범위 대비 상대적인 기울기"로
        # 해석되어, 다른 실험(다른 주파수 범위)에도 같은 tilt_slope
        # 값을 재사용하기 쉬워짐.
    tilt_curve = 1.0 + tilt_slope * f_normalized
        # 선형 기울기: 중심에서는 1.0(원래 진폭), 양 끝으로 갈수록
        # +-tilt_slope/2 만큼 진폭이 증가/감소.
    return signal_2d * tilt_curve[np.newaxis, :]
        # tilt_curve[np.newaxis, :] : (n_freq,) -> (1, n_freq)로 차원을 늘려서
        # signal_2d(shape: n_flux x n_freq)의 "모든 행에 같은 주파수별
        # 기울기를 곱하는" 브로드캐스팅이 되도록 함.
        # (앞서 add_calibration_drift에서 [:, np.newaxis]로 "행 방향"
        #  브로드캐스팅을 했다면, 이번엔 [np.newaxis, :]로 "열 방향"
        #  브로드캐스팅을 하는 것 - 축의 위치가 다름에 유의)


# =========================================================
# 3.6 κ(감쇠율)의 시간적 요동 - T1/T2 흔들림 시뮬레이션
# =========================================================
def generate_fluctuating_kappa(n_flux, kappa_base, fluctuation_level, correlation_length, seed=None):
    """
    측정이 진행되는 동안 κ(공진기 감쇠율, 결맞음과 반비례하는 양) 자체가
    slice마다 조금씩 요동치는 시계열을 생성.

    물리적 배경: 실제 초전도 큐빗/공진기의 결맞음 시간(T1, T2)은 하루
    안에서도, 심지어 한 번의 flux 스캔이 진행되는 동안에도 미세하게
    변동합니다 (주변 TLS 결함들의 상태 변화, 온도 요동 등이 원인으로
    추정됨). 이 함수는 그런 "물리량 자체가 시간에 따라 흔들리는" 상황을
    표현합니다 (지금까지의 add_white_noise 등이 "측정 과정의 잡음"이었다면,
    이건 "측정 대상 자체의 준정적(quasi-static) 변화"라는 점이 다름).

    구현 방법: 랜덤워크(random walk)에 이동평균을 적용해 "천천히,
    부드럽게 흔들리는" 시계열을 만듦 (매 slice마다 완전히 독립적으로
    확 바뀌는 게 아니라, 인접한 slice끼리는 비슷한 값을 갖도록).

    correlation_length : 값이 얼마나 "부드럽게" 흔들리는지 (클수록
                          더 매끈하게, 즉 더 천천히 변함)
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random
        # np.random.default_rng: numpy가 권장하는 최신 난수 생성기
        # (예전 방식인 np.random.seed()보다 더 안전하게 독립된 난수 스트림을
        # 만들 수 있어서, 여러 함수에서 각자 다른 rng를 써도 서로 간섭이 없음)

    raw_walk = np.cumsum(rng.normal(0, 1, n_flux))
        # np.cumsum: 누적합(cumulative sum)을 계산하는 함수.
        # 매 스텝마다 무작위로 +-값을 더해나가면 "랜덤워크"(취한 사람의
        # 걸음걸이처럼 무작위로 오락가락하며 이동하는 경로)가 만들어짐.

    # 이동평균으로 스무딩해서 "너무 들쭉날쭉하지 않고 부드럽게 흔들리는" 형태로 조정
    kernel = np.ones(correlation_length) / correlation_length
    smoothed_walk = np.convolve(raw_walk, kernel, mode='same')

    # 원하는 요동 크기(fluctuation_level)에 맞게 정규화한 뒤, kappa_base를 중심으로 더함
    smoothed_walk = smoothed_walk / np.std(smoothed_walk) * fluctuation_level
    kappa_series = kappa_base + smoothed_walk

    # 물리적으로 kappa는 반드시 양수여야 하므로, 혹시 요동이 너무 커서
    # 음수가 되는 경우를 대비해 작은 양수로 하한을 둠 (안전장치)
    kappa_series = np.clip(kappa_series, kappa_base * 0.1, None)
        # np.clip(값, 최소, 최대): 여기선 최대는 제한 안 함(None)

    return kappa_series


# =========================================================
# 4. Outlier (이상치)
# =========================================================
def add_outliers(signal_2d, probability, scale, rng=None):
    """
    무작위로 일부 데이터 포인트를 정상 범위를 크게 벗어나는 값으로 만듦.
    우주선(cosmic ray) 충돌, 준입자 파열(quasiparticle burst) 등
    가끔 발생하는 돌발적 이상 신호를 흉내냄. 통계적으로는 "긴 꼬리
    (heavy-tailed)" 분포를 만드는 요소.

    probability : 각 포인트가 outlier가 될 확률 (예: 0.01 = 1%)
    scale : outlier로 튈 때의 크기 (표준편차)
    """
    if rng is None:
        rng = np.random
    mask = rng.random(signal_2d.shape) < probability
        # rng.random(shape): 0~1 사이 균일분포(uniform distribution)에서
        # signal_2d와 같은 크기의 난수 배열을 뽑음.
        # 그 값이 probability(예:0.01)보다 작은 위치만 True가 되므로,
        # 전체 포인트 중 대략 probability 비율만큼 무작위로 선택됨.
    bump_real = rng.normal(0, scale, signal_2d.shape)
    bump_imag = rng.normal(0, scale, signal_2d.shape)
    noisy = np.where(mask, signal_2d + bump_real + 1j * bump_imag, signal_2d)
        # np.where(조건, A, B): 조건이 True인 위치는 A, False인 위치는 B의
        # 값을 쓰는 함수. 여기서는 mask가 True인(outlier로 뽑힌) 지점에만
        # 큰 값(bump)을 추가로 더하고, 나머지는 원래 신호 그대로 둠.
    return noisy, mask


# =========================================================
# 통합 함수: 여러 잡음을 한 번에 조합해서 적용
# =========================================================
def make_realistic_2d_dataset(clean_signal_2d,
                                white_level=0.015,
                                flicker_level=0.0,
                                drift_amplitude=0.0,
                                drift_period_fraction=0.6,
                                outlier_probability=0.0,
                                outlier_scale=0.4,
                                seed=None):
    """
    clean 신호에 원하는 잡음 성분들을 선택적으로 조합해 적용.
    각 레벨을 0으로 두면 해당 잡음이 꺼짐 -> "깨끗한 것부터 아주 지저분한
    것까지" 같은 함수 하나로 단계적으로 만들 수 있음.

    이렇게 설계해두면, 새 물리 모델(models.py에 추가한 다른 함수)로
    만든 clean 신호에도 그대로 재사용할 수 있습니다.
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random

    signal = clean_signal_2d.copy()

    if drift_amplitude > 0:
        signal, drift_2d = add_calibration_drift(signal, drift_amplitude, drift_period_fraction)
    else:
        drift_2d = np.zeros((signal.shape[0], 1))

    if white_level > 0:
        # 드리프트가 있으면 잡음 크기 자체도 같이 흔들리도록(실제 증폭기 이득
        # 변화와 유사하게) (1+drift) 배율을 곱해서 백색잡음을 적용
        white_scaled = (1 + drift_2d) * (
            rng.normal(0, white_level, signal.shape) + 1j * rng.normal(0, white_level, signal.shape)
        )
        signal = signal + white_scaled

    if flicker_level > 0:
        signal = add_flicker_noise(signal, flicker_level, rng)

    outlier_mask = np.zeros(signal.shape, dtype=bool)
    if outlier_probability > 0:
        signal, outlier_mask = add_outliers(signal, outlier_probability, outlier_scale, rng)

    return signal, outlier_mask


if __name__ == "__main__":
    # 이 파일을 직접 실행했을 때만 동작하는 자가진단 코드
    # (models.py와 동일한 패턴 - "구버전 파일을 잘못 쓰고 있는" 실수를
    #  빠르게 잡아내기 위한 장치)
    #
    # 사용법: Colab에서 `!python mock_data.py`로 실행하면, 지금 로드된
    # mock_data.py에 어떤 함수들이 들어있는지 즉시 확인 가능.
    expected_functions = [
        'add_white_noise',
        'generate_1f_noise_1d',
        'add_flicker_noise',
        'add_calibration_drift',
        'add_baseline_tilt',
        'generate_fluctuating_kappa',
        'add_outliers',
        'make_realistic_2d_dataset',
    ]
    print("mock_data.py 자가진단: 기대되는 함수들이 모두 있는지 확인")
    print("-" * 55)
    all_ok = True
    for name in expected_functions:
        exists = name in dir()
        status = "OK" if exists else "누락!! -> 구버전 파일일 수 있음"
        print(f"  {name:28s} : {status}")
        all_ok = all_ok and exists
    print("-" * 55)
    print("전체 상태:", "정상 (최신 버전)" if all_ok else "일부 함수 누락 - 파일을 새로 업로드하세요")
