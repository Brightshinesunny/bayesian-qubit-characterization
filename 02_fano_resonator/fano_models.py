"""
fano_models.py
=================================================================
원저자 circuit.py(Rieger & Guenzler et al., KIT, based on Sebastian
Probst의 resonator_tools)의 "대수적(algebraic) circle fit" 방법을
그대로 이식한 모듈입니다.

[전체 그림 - 이 파일이 하는 일을 한 문장으로]
주파수별로 측정된 복소수 S21 데이터(어제/오늘 계속 다룬 그 값)를
복소평면에 점으로 찍으면 "원(circle)"을 그리는데, 이 원의 기하학적
성질(중심, 반지름, 기울어진 각도)만 알면 공진기의 물리 파라미터
(공진주파수 fr, 품질계수 Ql/Qc/Qi, Fano 위상 phi)를 반복 탐색(MCMC)
없이 대수 공식으로 "한 번에" 계산해낼 수 있습니다.

[어제 배운 MCMC 방식과의 근본적 차이 - 다시 한번 확실히]
어제(Zenodo 데이터): 파라미터 후보를 계속 바꿔가며(워커들이 돌아다니며)
  "이 후보가 얼마나 데이터를 잘 설명하는지"를 반복 평가해서 점점
  정답에 접근(수렴)하는 방식. 시간이 걸리고, 축퇴가 있으면 수렴이
  잘 안 될 수 있음(R-hat 문제).
오늘(이 파일): 데이터가 "원"이라는 기하학적 사실을 이용해서, 행렬
  연산 한 번으로 원의 방정식을 직접 풀어버림. 반복도 수렴도 필요
  없이 항상 정해진 시간 안에 답이 나옴. 다만 "데이터가 정말 원을
  그리는 상황"에서만 쓸 수 있는 방법(모든 물리 모델에 적용 가능한
  것은 아님).

[이 파일 맨 끝에 물리/코드 용어 참고표가 있습니다]
낯선 용어가 나오면 그때그때 주석으로 설명하되, 자주 등장하는 핵심
용어들은 파일 맨 아래 "용어 참고표" 섹션에 모아서 한 번에 찾아볼 수
있게 정리해두었습니다.
"""

import numpy as np
import scipy.optimize as spopt
    # scipy.optimize: 최적화(원하는 값을 최소/최대로 만드는 파라미터
    # 찾기) 관련 함수들을 모아둔 scipy의 하위 모듈. 오늘은 이 중
    # newton(뉴턴법, 방정식의 근을 찾는 반복법)과 leastsq(비선형
    # 최소제곱법)를 씀. "반복이 전혀 없다"고 위에서 말했지만, 정확히는
    # "전체 파라미터를 한꺼번에 반복 탐색하지 않는다"는 뜻이고, 원의
    # 중심을 구하는 특성방정식의 근이나 delay 보정치처럼 아주 작은
    # 부분에서는 여전히 짧은 반복이 쓰임 - 이것도 아래에서 설명함.


# =========================================================
# 핵심 물리 모델: Sij (원저자 circuit.py의 Sij를 그대로 이식)
# =========================================================
def Sij(f, fr, Ql, Qc, phi=0., a=1., alpha=0., delay=0., n_ports=2.):
    """
    공진기의 산란 파라미터(scattering parameter) S를 계산하는 모델식.

    n_ports=1 이면 단일 포트 반사(reflection) 측정의 S11,
    n_ports=2 이면 2포트 투과(notch/transmission) 측정의 S21에 해당.
    (오늘 우리 데이터는 파일 안 배열 이름이 amplitude/phase로만 되어
    있어 명시적으로 "S21이다/S11이다"라고 적혀있지는 않지만, KIT 논문의
    측정 세팅(notch-type resonator)을 감안하면 n_ports=2가 기본값으로
    타당함 - 필요하면 나중에 n_ports=1로 바꿔 결과를 비교해볼 수 있음)

    수식:
        complexQc = Qc * cos(phi) * exp(-i*phi)
        S(f) = a * exp(i*(alpha - 2*pi*f*delay))
               * [1 - (2*Ql) / (complexQc * n_ports * (1 + 2i*Ql*(f/fr - 1)))]

    파라미터별 물리적 의미:
    ----------------------------------------------------------------
    fr    : 공진 주파수 (resonance frequency). 공진기가 가장 강하게
            반응하는 주파수. 어제의 f0와 동일한 개념.

    Ql    : loaded Q(로드된 품질계수). "이 공진기가 에너지를 잃기 전에
            몇 번이나 진동하는가"를 나타내는 무차원(단위 없는) 숫자.
            클수록 좋은(오래 버티는) 공진기. 내부손실과 외부결합
            손실을 모두 합친 "총 손실"에 대응하는 개념 - 어제의
            "ke1+ke2+ki"(총 kappa)와 정확히 반대 방향의 표현
            (Ql이 클수록 오히려 손실은 작다는 점에 주의: Ql = fr/kappa_total
            관계이므로, kappa가 작을수록(잘 안 새어나갈수록) Ql은 커짐)

    Qc    : coupling Q(결합 품질계수). 공진기가 "외부(케이블)로"
            에너지를 내보내는 정도만 따로 뽑은 값. 어제의 ke1,ke2를
            합친 개념과 대응. diameter correction method(원저자 docstring
            표현)로 계산 - 이는 원의 "지름(diameter)"이 Qc와 직접
            연결된다는 사실을 이용해 Qc를 구하는 방법이라는 뜻(아래
            _extract_Qs에서 실제로 이 방식이 등장함).

    phi   : Fano 위상(라디안). 이 값이 0이면 대칭적인 순수 로렌츠
            공진, 0이 아니면 배경 신호와의 간섭 때문에 딥/피크 모양이
            비대칭으로 기울어짐. 오늘 fano_visualize.py에서 복소평면
            궤적이 완벽한 원이 아니라 살짝 "찌그러져" 보였던 게 바로
            이 phi 때문. complexQc 수식에서 Qc를 복소수로 만드는
            역할을 하며, 이게 곧 "원이 원점을 중심으로 phi만큼
            회전된 것"과 기하학적으로 동일한 효과를 냄.

    a, alpha : 전체 신호의 진폭 스케일과 위상 오프셋. 어제의 A0, phi
            (참고: 어제 스크립트의 'phi'와 오늘 이 모델의 'phi'는
            같은 그리스 문자를 쓰지만 물리적 의미가 다른 별개의
            변수이니 헷갈리지 않도록 주의 - 어제는 "케이블 위상
            오프셋", 오늘은 "Fano 비대칭 각도")에 해당하는 장비
            배경 보정용 파라미터.

    delay : 케이블 지연(초 단위). 오늘 fano_loader.py에서 자동으로
            추출한 cable_delay(=port1_edel)가 바로 이 값. 어제는
            MCMC로 추정해야 했지만, 오늘은 장비가 직접 알려주므로
            추정할 필요 없이 고정값으로 바로 쓸 수 있음(다만 아래
            _fit_delay 함수처럼, 데이터 자체에서 잔여 delay를 한 번
            더 미세 보정하는 절차도 원저자 코드에 포함되어 있음 -
            장비가 알려준 값이 항상 완벽하게 정확하지는 않기 때문).

    n_ports : 위에서 설명한 대로, 측정 방식(반사 1 / 투과 2)에 따른 계수.
    """
    complexQc = Qc * np.cos(phi) * np.exp(-1j * phi)
    return a * np.exp(1j * (alpha - 2 * np.pi * f * delay)) * (
        1. - 2. * Ql / (complexQc * n_ports * (1. + 2j * Ql * (f / fr - 1.)))
    )


# =========================================================
# 원 피팅(circle fit)의 핵심 - 대수적 방법으로 원의 중심/반지름 구하기
# =========================================================
def fit_circle_algebraic(z_data):
    """
    복소수 데이터 z_data(예: S21 측정값들)에 "가장 잘 맞는 원"의
    중심(xc, yc)과 반지름(r0)을, 반복 탐색 없이 대수적으로(행렬 연산
    한 번으로) 계산합니다.

    [원리 - 왜 "원 맞추기"가 대수 문제로 바뀌는가]
    원의 방정식은 (x-xc)^2 + (y-yc)^2 = r0^2 인데, 이걸 그대로 두면
    미지수(xc, yc, r0)가 제곱(2차) 형태로 얽혀 있어 일반적인
    직선-최소제곱법을 바로 쓸 수 없습니다. 그런데 이 식을 펼쳐서
    정리하면:
        (x^2+y^2) = 2*xc*x + 2*yc*y + (r0^2 - xc^2 - yc^2)
    이 되는데, 여기서 z = x^2+y^2 라는 "새로운 변수"를 도입하면,
    z가 x, y, 1의 "선형 결합"으로 표현됩니다. 즉 겉보기에 2차였던
    문제가 z라는 보조 변수를 통해 다시 "선형 최소제곱 문제"로
    바뀌는 것 - 이게 이 함수 전체의 핵심 트릭입니다(Sebastian
    Probst 논문, arXiv:1410.3365에서 제안된 방법).

    [구현 절차 요약]
    1. 데이터를 정규화(스케일 통일)
    2. "모멘트 행렬" M을 구성 (x,y,z들의 다양한 곱과 합으로 이루어진
       4x4 행렬 - 통계학의 "이차 모멘트"와 비슷한 개념으로, 데이터
       분포의 퍼짐/치우침 정보를 담고 있음)
    3. 특성방정식(4차 방정식)의 근을 뉴턴법으로 찾아 "제약 조건"을 보정
    4. SVD(특이값분해)로 최종 원의 파라미터를 추출

    inputs:
    - z_data: 복소수 배열 (예: S21 측정값들). 실수부가 x좌표,
              허수부가 y좌표에 해당.

    outputs:
    - xc, yc: 원의 중심 좌표
    - r0: 원의 반지름
    """
    # --- STEP A. 정규화: 숫자 크기를 다루기 편한 범위로 통일 ---
    # S21 값 자체는 보통 0.01 근처의 작은 수인데, 이런 작은 숫자로
    # 그대로 행렬 연산을 하면 반올림 오차(부동소수점 오차)가 커질 수
    # 있음. 그래서 먼저 중심을 대략 원점 근처로 옮기고, 크기를
    # 1 근처로 맞춰서 수치적으로 안정된 계산이 되도록 함.
    x_norm = 0.5 * (np.max(z_data.real) + np.min(z_data.real))
    y_norm = 0.5 * (np.max(z_data.imag) + np.min(z_data.imag))
    z_data = z_data[:] - (x_norm + 1j * y_norm)
    amp_norm = np.max(np.abs(z_data))
    z_data = z_data / amp_norm

    # --- STEP B. 모멘트 행렬(matrix of moments) 구성 ---
    xi = z_data.real
    xi_sqr = xi * xi
    yi = z_data.imag
    yi_sqr = yi * yi
    zi = xi_sqr + yi_sqr
        # zi = x^2+y^2 : 위에서 설명한 "선형화를 가능하게 하는 보조 변수"
    Nd = float(len(xi))
        # Nd: 데이터 포인트 개수 (Number of Data의 줄임, 원저자 표기 그대로 유지)
    xi_sum = xi.sum()
    yi_sum = yi.sum()
    zi_sum = zi.sum()
    xiyi_sum = (xi * yi).sum()
    xizi_sum = (xi * zi).sum()
    yizi_sum = (yi * zi).sum()
    M = np.array([
        [(zi * zi).sum(), xizi_sum, yizi_sum, zi_sum],
        [xizi_sum, xi_sqr.sum(), xiyi_sum, xi_sum],
        [yizi_sum, xiyi_sum, yi_sqr.sum(), yi_sum],
        [zi_sum, xi_sum, yi_sum, Nd]
    ])
        # M: 4x4 "모멘트 행렬". 이 행렬의 각 성분은 데이터 점들의
        # x,y,z 값들을 짝지어 곱하고 합한 값들로 이루어져 있음 -
        # 마치 공분산 행렬을 4차원(x,y,z,1)으로 확장한 것과 비슷한
        # 역할. 원의 방정식을 최소제곱으로 풀 때 필요한 모든 정보가
        # 이 행렬 하나에 압축되어 담김.

    # --- STEP C. 특성방정식(4차 다항식)의 근을 찾아 제약 조건 보정 ---
    # 원의 방정식에는 "이게 실제로 원이 되려면 만족해야 하는 제약
    # 조건"이 있는데(타원이나 다른 이차곡선이 아니라 정확히 원이
    # 되려면), 이를 만족시키기 위해 라그랑주 승수법과 비슷한 방식으로
    # eta라는 보정값을 구함. 이 부분만 뉴턴법(spopt.newton)으로 근을
    # 찾는 짧은 반복이 들어가는데, 이는 "전체 6개 물리 파라미터를
    # MCMC로 반복 탐색하는 것"과는 완전히 다른 차원의 아주 작고
    # 국소적인 계산 (4차 다항식 하나의 근을 찾는 것뿐).
    a0 = ((M[2][0]*M[3][2]-M[2][2]*M[3][0])*M[1][1]-M[1][2]*M[2][0]*M[3][1]-M[1][0]*M[2][1]*M[3][2]+M[1][0]*M[2][2]*M[3][1]+M[1][2]*M[2][1]*M[3][0])*M[0][3]+(M[0][2]*M[2][3]*M[3][0]-M[0][2]*M[2][0]*M[3][3]+M[0][0]*M[2][2]*M[3][3]-M[0][0]*M[2][3]*M[3][2])*M[1][1]+(M[0][1]*M[1][3]*M[3][0]-M[0][1]*M[1][0]*M[3][3]-M[0][0]*M[1][3]*M[3][1])*M[2][2]+(-M[0][1]*M[1][2]*M[2][3]-M[0][2]*M[1][3]*M[2][1])*M[3][0]+((M[2][3]*M[3][1]-M[2][1]*M[3][3])*M[1][2]+M[2][1]*M[3][2]*M[1][3])*M[0][0]+(M[1][0]*M[2][3]*M[3][2]+M[2][0]*(M[1][2]*M[3][3]-M[1][3]*M[3][2]))*M[0][1]+((M[2][1]*M[3][3]-M[2][3]*M[3][1])*M[1][0]+M[1][3]*M[2][0]*M[3][1])*M[0][2]
        # 위 4개(a0~a3, 아래 a4)는 특성방정식의 계수. 4x4 행렬 M의
        # 원소들을 조합한 아주 긴 대수식인데, 이건 손으로 유도된
        # 수학 공식을 그대로 코드로 옮긴 것이라 "왜 이렇게 생겼는지"를
        # 항마다 이해할 필요는 없음(원 논문의 부록에 유도 과정이
        # 있음) - 다만 "M 행렬로부터 이 계수들을 계산해, 그 근을
        # 찾는다"는 전체 흐름만 이해하면 충분함.
    a1 = (((M[3][0]-2.*M[2][2])*M[1][1]-M[1][0]*M[3][1]+M[2][2]*M[3][0]+2.*M[1][2]*M[2][1]-M[2][0]*M[3][2])*M[0][3]+(2.*M[2][0]*M[3][2]-M[0][0]*M[3][3]-2.*M[2][2]*M[3][0]+2.*M[0][2]*M[2][3])*M[1][1]+(-M[0][0]*M[3][3]+2.*M[0][1]*M[1][3]+2.*M[1][0]*M[3][1])*M[2][2]+(-M[0][1]*M[1][3]+2.*M[1][2]*M[2][1]-M[0][2]*M[2][3])*M[3][0]+(M[1][3]*M[3][1]+M[2][3]*M[3][2])*M[0][0]+(M[1][0]*M[3][3]-2.*M[1][2]*M[2][3])*M[0][1]+(M[2][0]*M[3][3]-2.*M[1][3]*M[2][1])*M[0][2]-2.*M[1][2]*M[2][0]*M[3][1]-2.*M[1][0]*M[2][1]*M[3][2])
    a2 = ((2.*M[1][1]-M[3][0]+2.*M[2][2])*M[0][3]+(2.*M[3][0]-4.*M[2][2])*M[1][1]-2.*M[2][0]*M[3][2]+2.*M[2][2]*M[3][0]+M[0][0]*M[3][3]+4.*M[1][2]*M[2][1]-2.*M[0][1]*M[1][3]-2.*M[1][0]*M[3][1]-2.*M[0][2]*M[2][3])
    a3 = (-2.*M[3][0]+4.*M[1][1]+4.*M[2][2]-2.*M[0][3])
    a4 = -4.

    def char_pol(x):
        return a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4

    def d_char_pol(x):
        # 특성다항식의 도함수 (뉴턴법이 근을 빨리 찾도록 도와주는
        # "기울기" 정보 - 뉴턴법 자체가 "현재 위치의 기울기를 보고
        # 근이 있을 법한 방향으로 이동"하는 방식이라 도함수가 필요함)
        return a1 + 2*a2*x + 3*a3*x**2 + 4*a4*x**3

    eta = spopt.newton(char_pol, 0., fprime=d_char_pol)
        # spopt.newton: 뉴턴-랩슨법으로 방정식 char_pol(x)=0의 근을 찾는
        # scipy 함수. 초기 추정값 0에서 시작해서, 도함수(기울기) 정보를
        # 이용해 빠르게(보통 몇 번 안에) 정확한 근으로 수렴함. 이건
        # "MCMC처럼 수만 번 반복"하는 것과는 전혀 다른, 아주 적은
        # 횟수(보통 5~10번 이내)로 끝나는 국소적 계산.

    M[3][0] = M[3][0] + 2*eta
    M[0][3] = M[0][3] + 2*eta
    M[1][1] = M[1][1] - eta
    M[2][2] = M[2][2] - eta

    # --- STEP D. SVD(특이값분해)로 최종 원의 파라미터 추출 ---
    U, s, Vt = np.linalg.svd(M)
        # np.linalg.svd: 특이값분해(Singular Value Decomposition).
        # 임의의 행렬 M을 U, s(특이값들), Vt(V의 전치) 세 부분으로
        # 분해하는 선형대수의 표준 도구. 여기서는 "M을 0에 가장
        # 가깝게 만드는 방향(=제약을 가장 잘 만족하는 해)"을 찾는
        # 용도로 씀 - 가장 작은 특이값에 대응하는 벡터가 바로 그 해.
    A_vec = Vt[np.argmin(s), :]
        # np.argmin(s): 특이값들(s) 중 가장 작은 값의 위치를 찾음.
        # 가장 작은 특이값에 대응하는 벡터가 우리가 원하는 원의
        # 방정식 계수(A_vec)가 됨 - "M과 곱했을 때 결과가 0에 가장
        # 가까워지는 방향"이 곧 "M으로 표현된 제약 조건을 가장 잘
        # 만족하는 해"이기 때문.

    xc = -A_vec[1] / (2. * A_vec[0])
    yc = -A_vec[2] / (2. * A_vec[0])
    r0 = 1. / (2. * np.absolute(A_vec[0])) * np.sqrt(
        A_vec[1]*A_vec[1] + A_vec[2]*A_vec[2] - 4.*A_vec[0]*A_vec[3]
    )
        # 원의 방정식 계수(A_vec)로부터 중심(xc,yc)과 반지름(r0)을
        # 역산하는 표준 대수 공식. sqrt 안의 항은 수치오차로 아주
        # 살짝 음수가 될 수 있어(제약이 완벽히 만족되지 않는 경우),
        # 원저자 주석에서도 "제약 조건이 수치 계산 중 미세하게
        # 어긋날 수 있어 이 sqrt 항으로 보정한다"고 명시함.

    # 정규화를 되돌려서 원래 스케일의 좌표로 복원
    return xc*amp_norm + x_norm, yc*amp_norm + y_norm, r0*amp_norm


# =========================================================
# 위상 응답 피팅 (delay, fr, Ql의 초기 추정에 사용)
# =========================================================
def phase_centered(f, fr, Ql, theta, delay=0.):
    """
    "원점에 중심이 맞춰진" 공진기(강하게 과대결합된 경우에 해당)의
    이론적 위상 응답 공식.

    theta: offset phase(오프셋 위상). 공진에서 아주 멀리 떨어진 지점의
           위상 값 - 배경 위상의 기준점 역할.

    2*arctan(2*Ql*(1-f/fr)) 항이 핵심: 공진 주파수(f=fr)를 지나가면서
    arctan 함수가 -pi/2에서 +pi/2로 급격히 변하는데, 이게 바로
    fano_visualize.py에서 봤던 "위상이 공진 근처에서 급격히 튀어
    오르는" 현상의 수학적 근원.
    """
    return theta - 2*np.pi*delay*(f-fr) + 2.*np.arctan(2.*Ql*(1. - f/fr))


def periodic_boundary(angle):
    """
    임의의 각도를 [-pi, pi) 구간으로 접어넣는 함수.
    위상은 원형(circular)이라 2*pi를 더하거나 빼도 물리적으로 같은
    값이므로, 항상 이 "대표 구간" 안으로 정리해두는 것이 이후 계산의
    일관성을 위해 중요함 (예: 나중에 두 위상값을 비교할 때, 정리가
    안 되어 있으면 3.1과 -3.1이 "거의 같은 값"인데도 큰 차이로
    잘못 계산될 수 있음 - 어제 zenodo_joint_fit.py에서 phi 초기값을
    "원형평균"으로 계산했던 것과 정확히 같은 문제의식).
    """
    return (angle + np.pi) % (2*np.pi) - np.pi


def phase_distance(angle):
    """
    두 각도 사이의 "원 위에서의 거리"를 [0, pi] 범위로 반환.
    예를 들어 각도 차이가 350도든 -10도든, 실제로는 "10도만큼
    떨어져 있다"는 같은 의미이므로, 이 함수로 그 진짜 거리를 구함.
    잔차(residual) 계산 시 이 거리를 써야 위상이 감기는 지점에서
    엉뚱하게 큰 오차로 계산되는 것을 방지할 수 있음.
    """
    return np.pi - np.abs(np.pi - np.abs(angle))


def fit_phase(f_data, z_data, guesses=None):
    """
    (이미 원점 근처로 옮겨진) 데이터의 위상 응답에 phase_centered
    모델을 피팅해서 (fr, Ql, theta, delay)의 초기 추정치를 구함.

    [단계적 피팅 전략 - 왜 한 번에 4개를 다 피팅하지 않는가]
    4개 파라미터를 처음부터 동시에 피팅하면 국소최적점에 빠지기
    쉬워서(어제 MCMC에서 겪었던 것과 비슷한 문제), 원저자는 "적은
    파라미터부터 순차적으로 피팅해서 점점 정교화"하는 전략을 씀:
      1. Ql만 피팅 (나머지 고정)
      2. fr, theta 피팅
      3. delay만 피팅
      4. fr, Ql 피팅
      5. 마지막으로 4개 전부 함께 피팅 (지금까지의 좋은 초기값 덕분에
         이 마지막 전체 피팅은 안정적으로 수렴함)
    이것도 여전히 "국소적인 짧은 최적화"이지, 어제처럼 넓은 파라미터
    공간을 확률적으로 탐색하는 MCMC와는 성격이 다름.
    """
    phase = np.unwrap(np.angle(z_data))
        # np.angle: 복소수의 위상(각도)을 계산.
        # np.unwrap: 위상이 -pi/+pi 경계를 넘을 때 생기는 "인위적인
        # 점프"를 제거하고, 실제 물리적으로 연속적인 위상 곡선으로
        # 펼쳐주는 함수 (예: ...,3.0,3.1,-3.1,-3.0,... 처럼 보이는
        # 데이터를 ...,3.0,3.1,3.2,3.3,... 처럼 자연스럽게 이어지게 함)

    if np.max(phase) - np.min(phase) <= 0.8*2*np.pi:
        roll_off = np.max(phase) - np.min(phase)
    else:
        roll_off = 2*np.pi
        # roll_off: 위상이 전체적으로 얼마나 회전했는지(2*pi=한바퀴).
        # 완전히 원점에 중심이 맞춰진 원이면 정확히 한 바퀴(2*pi)를
        # 돌아야 하는데, 실제 데이터가 그에 못 미치면(즉 원이
        # 완전히 안 닫혀 있으면) 그 실제 값을 사용.

    if guesses is None:
        # gaussian_filter1d 없이 간단한 이동평균으로 대체 (scipy의
        # gaussian_filter1d 대신 numpy만으로 구현해 의존성을 줄임).
        #
        # [버그 수정] np.convolve(mode='same')은 배열 경계 밖을 암묵적으로
        # 0으로 채우는(zero-padding) 방식이라, phase 값이 0 근처가 아닐
        # 때(우리 데이터는 대개 그러함) 배열 양 끝에서 "진짜 신호가 아닌"
        # 인위적으로 큰 미분값이 생김. 이 가짜 봉우리를 argmax가 집어서
        # fr_guess가 진짜 공진점이 아니라 스캔 경계로 잘못 뽑히는 심각한
        # 버그로 이어졌음 (실제로 합성 데이터로 검증하다가 발견 - Ql
        # 결과가 -10^17 수준의 완전히 비물리적인 값으로 나와서 역추적함).
        # np.pad(mode='edge')로 먼저 양 끝을 "가장자리 값으로" 늘려준 뒤
        # convolve하면, 경계에서 인위적인 미분 폭증이 사라짐.
        pad_width = 5   # kernel 크기(11)의 절반 정도
        phase_padded = np.pad(phase, pad_width, mode='edge')
            # mode='edge': 배열의 첫/마지막 값을 그대로 반복해서 늘림
            # (0으로 채우는 것과 달리, 실제 데이터 경향을 자연스럽게
            # 이어가므로 인위적인 미분 폭증이 생기지 않음).
        kernel = np.ones(11) / 11
        phase_smooth_padded = np.convolve(phase_padded, kernel, mode='same')
        phase_smooth = phase_smooth_padded[pad_width:-pad_width]
            # 패딩으로 늘렸던 양 끝을 다시 잘라내어 원래 길이로 복원.
        phase_derivative = np.gradient(phase_smooth)
        fr_guess = f_data[np.argmax(np.abs(phase_derivative))]
            # 위상이 "가장 빠르게 변하는 지점"을 공진 주파수 초기값으로
            # 추정. phase_centered 공식을 보면 f=fr 근처에서 arctan
            # 함수의 기울기가 최대가 되므로, 이 지점을 찾으면 fr의
            # 좋은 초기 추정치가 됨.
        Ql_guess = 2 * fr_guess / (f_data[-1] - f_data[0])
        slope = phase[-1] - phase[0] + roll_off
        delay_guess = -slope / (2*np.pi*(f_data[-1]-f_data[0]))
    else:
        fr_guess, Ql_guess, delay_guess = guesses

    theta_guess = 0.5 * (np.mean(phase[:5]) + np.mean(phase[-5:]))
        # 스캔 양 끝(공진에서 가장 먼 지점들) 위상의 평균을 오프셋
        # 위상(theta) 초기값으로 사용.

    def residuals_full(params):
        return phase_distance(phase - phase_centered(f_data, *params))

    def residuals_Ql(params):
        return residuals_full((fr_guess, params[0], theta_guess, delay_guess))
    def residuals_fr_theta(params):
        return residuals_full((params[0], Ql_guess, params[1], delay_guess))
    def residuals_delay(params):
        return residuals_full((fr_guess, Ql_guess, theta_guess, params[0]))
    def residuals_fr_Ql(params):
        return residuals_full((params[0], params[1], theta_guess, delay_guess))

    # 단계적 피팅 (leastsq: Levenberg-Marquardt 알고리즘 기반 비선형
    # 최소제곱법 - scipy의 표준적인 국소 최적화 함수)
    Ql_guess, = spopt.leastsq(residuals_Ql, [Ql_guess])[0]
    fr_guess, theta_guess = spopt.leastsq(residuals_fr_theta, [fr_guess, theta_guess])[0]
    delay_guess, = spopt.leastsq(residuals_delay, [delay_guess])[0]
    fr_guess, Ql_guess = spopt.leastsq(residuals_fr_Ql, [fr_guess, Ql_guess])[0]
    final = spopt.leastsq(residuals_full, [fr_guess, Ql_guess, theta_guess, delay_guess])[0]

    return final  # (fr, Ql, theta, delay)


# =========================================================
# 전체 파이프라인: autofit - 위 함수들을 조합해 최종 결과 산출
# =========================================================
def autofit(f_data, z_data_raw, n_ports=2., fixed_delay=None,
             isolation=15, calc_errors=True):
    """
    circuit.py의 autofit() 메소드를 함수형으로 이식한 버전.
    측정 데이터(f_data, z_data_raw)로부터 공진기의 모든 물리
    파라미터를 자동으로 추출합니다.

    처리 순서 (STEP A~E):
      A. delay 결정 (fixed_delay가 주어지면 그걸 쓰고, 아니면 데이터에서 추정)
      B. calibrate: 원을 이용해 fr, Ql, phi, a, alpha를 대수적으로 계산
      C. normalize: 데이터를 "표준 위치"(원점 기준)로 정규화
      D. extract_Qs: 반지름으로부터 Qc, Qi 계산
      E. calc_fano_range: Fano 간섭에 의한 Qi 불확실성 범위 계산

    Returns: fitresults 딕셔너리 (모든 결과가 이 안에 담김)
    """
    fitresults = {}

    # --- STEP A. Delay 결정 ---
    if fixed_delay is not None:
        delay = fixed_delay
            # fano_loader.py가 자동으로 뽑아준 cable_delay(port1_edel)를
            # 여기에 바로 넣을 수 있음 - 어제 MCMC에서 tau를 추정해야
            # 했던 것과 달리, 오늘은 장비가 이미 알려준 값을 그대로 사용.
    else:
        # 데이터 자체에서 delay를 반복적으로 추정 (circuit.py의
        # _fit_delay 로직을 단순화한 버전 - 원의 중심을 원점으로
        # 옮겨가며 위상 기울기를 보정하는 절차를 5회 반복)
        xc, yc, r0 = fit_circle_algebraic(z_data_raw)
        z_centered = z_data_raw - complex(xc, yc)
        fr, Ql, theta, delay = fit_phase(f_data, z_centered)
        delay *= 0.05
            # "과잉반응하지 않기(do not overreact)": 첫 추정치를 그대로
            # 다 반영하지 않고 5%만 반영해서, 다음 반복에서 서서히
            # 정확한 값으로 접근하도록 함(발산 방지를 위한 감쇠 계수).
        for _ in range(5):
            z_data = z_data_raw * np.exp(2j*np.pi*delay*f_data)
            xc, yc, r0 = fit_circle_algebraic(z_data)
            z_data = z_data - complex(xc, yc)
            fr, Ql, theta, delay_corr = fit_phase(
                f_data, z_data, guesses=(fr, Ql, 5e-11)
            )
            delay += 0.1 * delay_corr
                # 매 반복마다 보정치의 10%만 반영 (역시 발산 방지).
                # 이 반복은 최대 5번으로 고정되어 있어, "언제까지고
                # 계속 도는" MCMC와 달리 항상 정해진 횟수 안에 끝남.
    fitresults['delay'] = delay

    # --- STEP B. Calibrate: 원의 기하학으로부터 물리 파라미터 추출 ---
    z_data = z_data_raw * np.exp(2j*np.pi*delay*f_data)
    xc, yc, r0 = fit_circle_algebraic(z_data)
    zc = complex(xc, yc)
    z_centered = z_data - zc

    fr, Ql, theta, delay_remaining = fit_phase(f_data, z_centered)
    theta = periodic_boundary(theta)
    beta = periodic_boundary(theta - np.pi)
        # beta: "off-resonant point"(공진에서 무한히 먼 지점에 해당하는
        # 각도) - 원 위에서 공진점의 정반대편 위치.
    offrespoint = zc + r0*np.cos(beta) + 1j*r0*np.sin(beta)
    a = np.absolute(offrespoint)
    alpha = np.angle(offrespoint)
    phi = periodic_boundary(beta - alpha)
        # phi: 드디어 여기서 Fano 위상(원이 얼마나 회전되어 있는지)이
        # 기하학적으로 직접 계산됨 - "off-resonant point의 각도"와
        # "실제 원 위의 대응점 각도"의 차이가 곧 phi.
    r0_relative = r0 / a
        # 이후 계산에 필요한, a로 정규화된 상대적 반지름.

    fitresults.update({
        'fr': fr, 'Ql': Ql, 'theta': theta, 'phi': phi,
        'a': a, 'alpha': alpha, 'delay_remaining': delay_remaining,
    })

    # --- STEP D. Qc, Qi 추출 (반지름으로부터) ---
    absQc = Ql / (n_ports * r0_relative)
        # "diameter correction method": 원의 반지름(정확히는 지름)이
        # Qc와 직접 비례한다는 기하학적 사실을 이용해 Qc를 구함.
        # 반지름이 클수록(원이 클수록) 결합이 강하다(Qc가 작다) -
        # 실제로는 반비례 관계이므로 이 공식에 r0_relative가 분모에 있음.
    Qc = absQc / np.cos(phi)
        # phi(Fano 위상)를 반영해 "진짜" Qc로 보정. phi가 0이면
        # Qc = absQc 그대로, phi가 커질수록 보정량이 커짐.
    Qi = 1. / (1./Ql - 1./Qc)
        # Qi(내부 품질계수) = Ql과 Qc로부터 역산. 이게 바로 어제
        # 우리가 MCMC로 애먹었던 "ki를 어떻게 구할 것인가"의 오늘
        # 버전 답 - 축퇴 없이 대수적으로 바로 나옴 (다만 이건 phi를
        # 통해 Fano 효과가 이미 모델 안에서 분리되어 있기 때문에
        # 가능한 것 - 공짜로 얻어지는 게 아니라, 원의 기하학이라는
        # "추가 정보"를 활용한 결과임)

    fitresults.update({'Qc': Qc, 'Qc_no_dia_corr': absQc, 'Qi': Qi})

    # --- STEP E. Fano 간섭에 의한 Qi 불확실성 범위 계산 ---
    b = 10**(-isolation/20)
        # isolation(격리도, dB단위)을 선형 스케일 진폭비(b)로 환산.
        # dB에서 선형으로 바꾸는 표준 공식: 진폭비 = 10^(dB/20)
        # (전력비라면 10^(dB/10)을 쓰지만, 여기서는 "진폭"의 비율이므로 /20)
    b = b / (1 - b)

    R_mid = r0_relative * np.cos(phi)
    R_err = r0_relative * np.sqrt(max(b**2 - np.sin(phi)**2, 0))
        # [수정] 원저자 코드는 sin(phi)>b인 경우를 별도 경고로만
        # 처리했는데, 여기서는 sqrt 안의 값을 max(..., 0)으로 감싸서
        # 혹시 음수가 되더라도(=이 isolation 가정으로는 설명 불가능한
        # 정도로 phi가 큰 경우) 에러 없이 0으로 처리되도록 안전장치를
        # 추가함. 이는 물리적으로 "이 정도로 강한 Fano 효과라면 이
        # isolation 가정 자체가 틀렸다"는 신호이므로, 실전에서는 이
        # 경고가 뜨면 isolation 값을 재검토해야 함.
    R_min = R_mid - R_err
    R_max = R_mid + R_err

    Qc_min = Ql / (n_ports * R_max) if R_max > 0 else np.inf
    Qc_max = Ql / (n_ports * R_min) if R_min > 0 else np.inf
    Qi_min = Ql / (1 - n_ports * R_min) if (1 - n_ports*R_min) > 0 else np.inf
    Qi_max = Ql / (1 - n_ports * R_max) if (1 - n_ports*R_max) > 0 else np.inf
        # 이 네 값이 바로 오늘 처음 짚었던 "Fano 간섭으로 인한
        # 체계적 불확실성 범위"입니다. 어제 우리 피셔 행렬의 NaN이
        # "축퇴로 인해 오차를 정의할 수 없다"는 걸 나타냈다면, 오늘
        # 이 Qi_min/Qi_max는 "정확한 값 하나로는 못 정하지만, 최소
        # 이 범위 안에는 있다"는 훨씬 더 구체적이고 실전적인 정보를
        # 제공합니다 - 이게 원저자 논문의 핵심 기여입니다.

    fitresults.update({
        'Qc_min': Qc_min, 'Qc_max': Qc_max,
        'Qi_min': Qi_min, 'Qi_max': Qi_max,
        'fano_b': b,
    })

    return fitresults


if __name__ == "__main__":
    print("fano_models.py 자가진단: 기대되는 함수들이 모두 있는지 확인")
    print("-" * 55)
    expected = ['Sij', 'fit_circle_algebraic', 'phase_centered',
                'periodic_boundary', 'phase_distance', 'fit_phase', 'autofit']
    for name in expected:
        print(f"  {name:28s} : {'OK' if name in dir() else '누락!!'}")


# =========================================================================
# 용어 참고표 (Glossary) - 이 파일 전체에서 쓰인 물리/코드 용어 모음
# =========================================================================
#
# [물리 용어]
# -------------------------------------------------------------------------
# S21 / S11        : 산란 파라미터. 포트1->포트2 투과(S21) 또는 포트1
#                     반사(S11) 신호의 복소수 비율.
# fr               : 공진 주파수. 공진기가 가장 강하게 반응하는 주파수.
# Ql (loaded Q)    : 총 손실을 반영한 품질계수. 클수록 좋은 공진기.
#                     Ql = fr / (총 감쇠율).
# Qc (coupling Q)  : 외부(케이블)로의 결합에 의한 손실만 나타내는 품질계수.
# Qi (internal Q)  : 공진기 자체(재질 결함 등)의 내부 손실만 나타내는
#                     품질계수. 1/Ql = 1/Qi + 1/Qc 관계.
# phi (Fano 위상)  : 배경 신호와의 간섭으로 인한 비대칭 정도(라디안).
#                     0이면 대칭 로렌츠, 0이 아니면 비대칭.
# a, alpha         : 전체 신호의 진폭 스케일/위상 오프셋 (장비 배경 보정용).
# delay            : 케이블 전기 지연(초). 신호가 배선을 지나며 생기는
#                     선형적 위상 회전의 원인.
# isolation        : 배경(간섭) 경로가 원래 신호 대비 얼마나 억제되어
#                     있는지를 나타내는 dB 값.
# offrespoint      : 공진에서 무한히 먼 지점(off-resonant point)에
#                     대응하는, 원 위의 특정 위치.
# critical/over/undercoupled : 결합 세기에 따른 공진기 상태 분류.
#                     Qc와 Qi가 같으면 critical, Qc<Qi면 overcoupled,
#                     Qc>Qi면 undercoupled (어제 lineshape_gallery.py
#                     에서 시각화했던 개념과 동일).
#
# [수학/통계 용어]
# -------------------------------------------------------------------------
# algebraic fit (대수적 피팅) : 반복 없이 방정식을 한 번 풀어 답을
#                     구하는 방식. 이 파일의 fit_circle_algebraic이 이 방식.
# iterative fit (반복적 피팅) : 후보를 여러 번 시도하며 점점 정답에
#                     접근하는 방식 (예: 어제의 MCMC, 이 파일의 fit_phase
#                     내부 leastsq도 작은 규모로는 이 방식).
# 특성방정식 (characteristic polynomial) : 어떤 조건(여기서는 "원이
#                     되기 위한 제약")을 만족시키는 값을 찾기 위해
#                     세우는 다항식. 그 근(root)이 우리가 찾는 해.
# 뉴턴법 (Newton's method) : 함수의 기울기(도함수) 정보를 이용해
#                     방정식의 근을 빠르게 찾는 반복 알고리즘.
# SVD (특이값분해, Singular Value Decomposition) : 행렬을 세 개의
#                     간단한 행렬(U, 특이값들, V)의 곱으로 분해하는
#                     선형대수의 표준 도구.
# 최소제곱법 (least squares) : 데이터와 모델 예측값의 차이(잔차)의
#                     제곱합을 최소로 만드는 파라미터를 찾는 방법.
# Levenberg-Marquardt : 비선형 최소제곱 문제를 풀 때 널리 쓰이는
#                     표준 알고리즘 (scipy의 leastsq가 내부적으로 사용).
# unwrap (위상 펼치기) : -pi/+pi 경계에서 생기는 인위적 불연속을
#                     제거해 위상을 매끄러운 연속 곡선으로 만드는 처리.
# periodic boundary (주기적 경계) : 각도처럼 "한 바퀴 돌면 제자리로
#                     돌아오는" 값을 다루기 위한 처리 방식.
#
# [코드/파이썬 용어]
# -------------------------------------------------------------------------
# np.linalg.svd    : 특이값분해를 계산하는 numpy 함수.
# np.argmin/argmax : 배열에서 최솟값/최댓값의 "위치(인덱스)"를 찾는 함수.
# scipy.optimize.newton : 뉴턴법으로 방정식의 근을 찾는 scipy 함수.
# scipy.optimize.leastsq : Levenberg-Marquardt 알고리즘으로 비선형
#                     최소제곱 문제를 푸는 scipy 함수.
# np.unwrap        : 위상 펼치기를 수행하는 numpy 함수.
# np.gradient      : 배열의 국소 기울기(수치 미분)를 계산하는 함수.
# np.convolve      : 두 배열의 합성곱(convolution)을 계산 - 여기서는
#                     이동평균 스무딩에 사용.
# complex(x, y)    : 실수부 x, 허수부 y를 갖는 복소수를 만드는 파이썬
#                     내장 함수 (= x + 1j*y 와 동일).
# =========================================================================
