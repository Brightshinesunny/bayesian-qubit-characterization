"""
tls_avcross_visual_check.py
=================================================================
[오늘 A단계 - 육안 검증] 지금까지 세 가지 다른 방법으로 얻은
"TLS 주파수 vs 전압" 직선을, 실측 dispamp 히트맵 위에 겹쳐 그려서
어느 것이 실제로 보이는 어두운 딥 띠를 따라가는지 직접 눈으로
확인합니다.

[비교 대상 3가지]
  1. 픽셀 추적(오늘 초반, 9점만): gamma_stark≈109MHz/V
  2. 2D모델 MCMC STEP2(작은-gamma 시작): gamma_stark≈-1.67MHz/V
  3. 2D모델 MCMC STEP3(큰-gamma 시작): gamma_stark≈0.64MHz/V

[이 스크립트가 확인해주는 것]
숫자로는 셋이 100배 넘게 다르다는 걸 알았지만, 그림으로 보면
"어느 직선이 실제 어두운 띠 위에 정확히 올라가 있는지" 한눈에
판단할 수 있습니다. 이게 오늘 마지막으로 필요했던 검증입니다.
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

aa_mat_filepath = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module02/TLS/quadspec_Segments1to200_20TLSfitted.mat'
aa_target_raw_index = 12
aa_observable = 'dispamp'
aa_V0 = -56.5

# --- 오늘 확인된 세 가지 (f_TLS0, gamma_stark) 후보 ---
aa_candidates = {
    '픽셀추적(9점, 교차전)':      {'f_TLS0': 5.135, 'gamma_stark': 0.109, 'color': 'red'},
    '2D모델 STEP2(작은-gamma)':  {'f_TLS0': 5.1403, 'gamma_stark': -0.00167, 'color': 'cyan'},
    '2D모델 STEP3(큰-gamma시작)': {'f_TLS0': 5.1723, 'gamma_stark': 0.00064, 'color': 'yellow'},
}
    # [주의] f_TLS0 값들이 서로 다른 V0/기준점 정의를 쓴 결과라, 아래
    # 코드에서 "V=aa_V0일 때의 값"으로 통일해서 그림. 픽셀추적은
    # 원래 V=-56.8 기준이었던 걸 V0=-56.5로 변환해서 표시.


def mat_string(f, dataset_or_ref):
    try:
        if isinstance(dataset_or_ref, h5py.Reference):
            dataset_or_ref = f[dataset_or_ref]
        arr = np.array(dataset_or_ref)
        return ''.join(chr(int(c)) for c in arr.flatten())
    except Exception:
        return None


with h5py.File(aa_mat_filepath, 'r') as f:
    qdat_group = f[f['qdats']['qdat'][aa_target_raw_index, 0]]
    sweep1vals = np.array(qdat_group['sweep1vals']).flatten()
    sweep2vals = np.array(qdat_group['sweep2vals']).flatten()

    calib_field = np.array(f['qset']['calibdat']['field']).flatten()
    calib_freq_hz = np.array(f['qset']['calibdat']['qubitfreq']).flatten()
    order = np.argsort(calib_field)
    fq_axis = np.interp(sweep2vals, calib_field[order], calib_freq_hz[order]) / 1e9

    obs_val_refs = qdat_group['obs']['vals']
    obs_name_refs = qdat_group['observables']
    target_col = None
    for i in range(obs_name_refs.shape[0]):
        if mat_string(f, obs_name_refs[i, 0]) == aa_observable:
            target_col = i
            break
    data_2d = np.array(f[obs_val_refs[target_col, 0]])

# --- 배경 정규화 (열별 중앙값 제거) - 오늘 계속 써온 방식 ---
col_median = np.median(data_2d, axis=0, keepdims=True)
normalized = data_2d - col_median

# --- 시각화 ---
fig, ax = plt.subplots(figsize=(12, 7))
im = ax.pcolormesh(sweep1vals, fq_axis, normalized, shading='auto', cmap='viridis')
plt.colorbar(im, ax=ax, label=aa_observable)

V_line = np.linspace(sweep1vals.min(), sweep1vals.max(), 100)
for label, params in aa_candidates.items():
    # 픽셀추적 후보는 V=-56.8 기준으로 얻은 값이므로, V0=-56.5 기준으로
    # 변환: f_TLS(V)=f_TLS0_at_V0+gamma*(V-V0) 이고 V=-56.8일 때
    # f=5.135라는 걸 알고 있으므로, 이 식을 f_TLS0_at_V0에 대해 풀면:
    #   f_TLS0_at_V0 = 5.135 - gamma*((-56.8)-V0)
    # [주의] 부호를 빼기로 해야 정확함 - 처음에 더하기로 잘못 써서
    # 역산 검증(V=-56.8 대입 시 5.135가 나와야 함)에서 5.0696으로
    # 틀렸던 걸 직접 확인하고 수정함.
    if '픽셀추적' in label:
        f_TLS0_at_V0 = params['f_TLS0'] - params['gamma_stark'] * (-56.8 - aa_V0)
    else:
        f_TLS0_at_V0 = params['f_TLS0']
    freq_line = f_TLS0_at_V0 + params['gamma_stark'] * (V_line - aa_V0)
    ax.plot(V_line, freq_line, '-', color=params['color'], lw=2, label=label)

ax.set_xlabel('게이트 전압 V')
ax.set_ylabel('큐빗 주파수 (GHz)')
ax.set_title('seg12 실측 dispamp + 오늘 확인된 3가지 후보 직선 겹쳐그리기\n'
              '(어느 선이 실제 어두운 딥 띠 위에 있는지 육안 확인)')
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(fq_axis.min(), fq_axis.max())

plt.tight_layout()
plt.savefig('./outputs/tls_avcross_visual_check.png', dpi=150, bbox_inches='tight')
print("저장 완료: ./outputs/tls_avcross_visual_check.png")


# =========================================================
# [용어 정리 - 물리 + 통계, 매번 첨부]
# =========================================================
"""
--- 물리 용어 ---
avoided-crossing(회피교차) : 두 시스템의 에너지 준위가 직접 만나지
    않고 밀어내며 갈라지는 현상.
Stark 편이                 : 전압(전기장)에 의해 TLS 공명주파수가
    이동하는 현상.
결합세기 gamma_stark(MHz/V): 전압 1V당 TLS 주파수 이동량.
결합강도 g(GHz)             : 큐빗-TLS 에너지 교환 속도.

--- 통계/시각화 용어 ---
배경 정규화(열별 중앙값 제거) : 각 전압(열)마다 중앙값을 빼서, 전체
    밝기 변화(전압에 따른 배경 레벨 차이)를 없애고 국소적인 구조만
    남기는 전처리.
육안 검증(visual overlay check) : 숫자(파라미터 값)만으로는 판단하기
    어려운 "모델이 실제로 맞는 특징을 설명하는지"를, 그림 위에 겹쳐
    그려서 직접 확인하는 절차. 오늘처럼 여러 방법이 서로 다른 숫자를
    낼 때, 이 확인이 최종 판정에 결정적인 역할을 함.
"""
