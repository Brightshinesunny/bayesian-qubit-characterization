"""
tls_position_quick_attempt.py
=================================================================
[한계부터 확실히] 전기장 시뮬레이션도, 전극 실좌표도 없어서
"진짜 (x,y,z) 위치"는 못 구함. 대신 4개 전극을 사각형 모서리에
임의로 배치했다고 "가정"하고, 결합세기로 가중평균한 대략적인
방향(어느 쪽으로 치우쳤는지)만 뽑는 간이(illustrative) 버전.
진짜 위치추정 아니고, 그냥 "감 잡는 용도"라고 보면 됨.

[방법 - 가중 중심(weighted centroid)]
전극을 정사각형 4모서리에 둠(alpha=(1,1), beta=(-1,1), gamma=(-1,-1),
delta=(1,-1), 단위는 임의). 각 TLS 위치는
    position ≈ sum(|gamma_i| * pos_i) / sum(|gamma_i|)
로 근사. 결합세기 큰 전극 쪽으로 위치가 쏠리는 원리 - 물리적으로는
"결합이 강할수록 그 전극에 더 가깝다"는 가정을 그대로 좌표에 반영한 것.
"""

import h5py
import numpy as np

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
aa_gamma_rows = [0, 2, 9, 11]   # 어제 해독한 행 번호 - 순서대로 alpha,beta,gamma,delta로 임시 배정
aa_electrode_positions = {
    0: np.array([1, 1]),    # alpha - 임의 단위, 실좌표 아님. 그냥 사각형 모서리
    2: np.array([-1, 1]),   # beta
    9: np.array([-1, -1]),  # gamma
    11: np.array([1, -1]),  # delta
}
    # 이 4개 좌표는 순전히 "동서남북 대충 나눠보자"는 편의상 배치임.
    # 실제 칩 위에서 이 4개 전극이 진짜 이런 정사각형 모양으로
    # 놓여있는지조차 확인 안 했음 - 그냥 방향 감 잡는 용도.


def mat_string(f, dataset_or_ref):
    """MATLAB의 uint16 코드 배열 문자열을 파이썬 문자열로 변환.
    문자열 아니면(숫자면) None 반환해서 위에서 걸러내는 용도."""
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
    freq_row = 4   # 어제 확인한 TLS 고유주파수가 들어있는 행

    print(f"{'열':>5} {'주파수(GHz)':>12} {'가중중심(x,y)':>18} {'가장 가까운 전극':>16}")
    print("-"*60)

    results_for_story = {}   # 나중에 스토리텔링 예시에 쓸 원본 gamma값 저장용

    for col in range(n_cols):
        gamma_vals = {}
        skip = False
        for row in aa_gamma_rows:
            ref = tabledata2_refs[row, col]
            val = np.array(f[ref])
            if val.dtype == np.uint16:
                # 문자열 필드가 섞인 열은 그냥 건너뜀(빈 자리거나 다른 용도)
                skip = True
                break
            gamma_vals[row] = float(val.flatten()[0]) if val.size > 0 else 0.0
        if skip:
            continue

        max_abs = max(abs(v) for v in gamma_vals.values())
        if max_abs < 1e6:
            # 4개 값이 다 거의 0이면 빈 TLS 항목(200열 중 실제 채워진 건 20개뿐)
            continue

        # 절댓값을 가중치로 씀 - 부호(+/-)는 "어느 방향으로 편이하는지"를
        # 나타낼 뿐 "얼마나 가까운지"랑은 무관하니까 절댓값만 씀
        weights = np.array([abs(gamma_vals[r]) for r in aa_gamma_rows])
        positions = np.array([aa_electrode_positions[r] for r in aa_gamma_rows])
        centroid = np.sum(weights[:, None] * positions, axis=0) / np.sum(weights)

        nearest_electrode_row = aa_gamma_rows[np.argmax(weights)]
        electrode_names = {0: 'alpha', 2: 'beta', 9: 'gamma', 11: 'delta'}

        freq_val = np.array(f[tabledata2_refs[freq_row, col]]).flatten()[0]
        print(f"{col:>5} {freq_val/1e9:>12.4f} ({centroid[0]:>6.2f},{centroid[1]:>6.2f}) "
              f"{electrode_names[nearest_electrode_row]:>16}")

        results_for_story[col] = {'gammas': gamma_vals, 'centroid': centroid,
                                     'nearest': electrode_names[nearest_electrode_row],
                                     'freq': freq_val}

    print("\n[다시 한번 강조] 위 (x,y)는 실제 물리적 좌표가 아니라,")
    print("임의로 배치한 전극 모서리 기준 상대적 방향일 뿐임.")
    print("정확한 위치 구하려면 전극 실좌표 + 전기장 시뮬레이션 필요.")

    # =====================================================
    # [신규] 결과 하나를 골라 스토리텔링 형식으로 해석 예시 작성
    # =====================================================
    # 열11(delta 쪽으로 확 쏠린 경우)과 열19(alpha 쪽으로 쏠린 경우)를
    # 예시로 골라서, "숫자를 어떻게 읽으면 되는지"를 이야기 형식으로
    # 풀어씀 - 실제 해석할 때 참고용.
    print("\n" + "="*70)
    print("[해석 예시 - 스토리텔링]")
    print("="*70)

    if 11 in results_for_story:
        r = results_for_story[11]
        g = r['gammas']
        print(f"""
열11번 TLS를 예로 들면:
  alpha={g[0]/1e6:.1f}, beta={g[2]/1e6:.1f}, gamma={g[9]/1e6:.1f}, delta={g[11]/1e6:.1f} (MHz/V)

  네 숫자 중 delta({g[11]/1e6:.1f})가 압도적으로 큽니다. alpha나 beta는
  게이트 전압을 아무리 흔들어도 이 TLS 주파수가 거의 안 움직이는데,
  delta 전압만 살짝 건드려도 주파수가 크게 요동칩니다.

  비유하자면: 방 안에 4명이 서 있고, 그중 delta 씨만 목소리를 조금
  높여도 이 TLS(마이크처럼)가 크게 반응하고, 나머지 3명은 소리를
  질러도 거의 반응이 없는 상황. 그러면 "이 마이크는 delta 씨 쪽에
  가까이 있다"고 추측하는 게 자연스럽습니다 - 딱 그 논리입니다.

  그래서 가중중심 좌표도 ({r['centroid'][0]:.2f},{r['centroid'][1]:.2f})로,
  delta 자리(1,-1) 방향으로 확 쏠려 나왔습니다.
""")

    if 19 in results_for_story:
        r = results_for_story[19]
        g = r['gammas']
        print(f"""
반대로 열19번 TLS는:
  alpha={g[0]/1e6:.1f}, beta={g[2]/1e6:.1f}, gamma={g[9]/1e6:.1f}, delta={g[11]/1e6:.1f} (MHz/V)

  이번엔 alpha가 압도적으로 큽니다({g[0]/1e6:.1f}). 좌표도 (0.84,0.82)로
  alpha 자리(1,1) 바로 근처까지 쏠렸습니다 - 열11이 delta 쪽으로
  쏠린 것과 정반대 패턴입니다.
""")

    print("[주의] 위는 '숫자를 읽는 방법 예시'일 뿐, 실제 물리적 위치")
    print("확정이 아님. 진짜 확정하려면 전극 실좌표+시뮬레이션 필요.")

    # =====================================================
    # [신규] 20개 TLS의 가중중심을 산점도로 시각화
    # =====================================================
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    for r in results_for_story.values():
        ax.scatter(r['centroid'][0], r['centroid'][1], s=80, alpha=0.7)
    # 전극 4개 위치도 참고용으로 같이 표시
    for row, pos in aa_electrode_positions.items():
        names = {0:'alpha', 2:'beta', 9:'gamma', 11:'delta'}
        ax.scatter(pos[0], pos[1], marker='*', s=300, color='red')
        ax.annotate(names[row], pos, fontsize=12, ha='center', va='bottom')
    ax.set_xlabel('x (임의 단위, 실좌표 아님)')
    ax.set_ylabel('y (임의 단위, 실좌표 아님)')
    ax.set_title('20개 TLS의 가중중심 산점도 (간이 추정)')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    plt.tight_layout()
    plt.savefig('./outputs/tls_position_scatter.png', dpi=140)
    print("\n산점도 저장 완료: ./outputs/tls_position_scatter.png")

    # =====================================================
    # [용어 설명 - 물리/통계 공통, 앞으로도 스크립트 끝에 계속 붙임]
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
68% 신뢰구간           : 진짜 값이 이 구간 안에 있을 확률이 68%라는 베이지안적 해석
가중중심(weighted centroid): 여러 값(전극 결합세기)에 가중치를 줘서 평균 낸 대표 위치
""")
