"""
fano_fit_real_data.py
=================================================================
실제 Fano 데이터(resonator_1_powersweep_overcoupled.npz)에
fano_models.autofit()을 적용해서, 공진기의 물리 파라미터(fr, Ql,
Qc, Qi, phi)와 Fano 간섭에 의한 불확실성 범위(Qi_min/Qi_max)를
뽑아내는 스크립트입니다.

[오늘까지의 흐름 - 이 스크립트의 위치]
fano_loader.py(데이터 로드) -> fano_visualize.py(데이터 형태 확인,
복소평면이 실제로 원을 그리는지 확인) -> fano_models.py(circle fit
알고리즘 이식 및 합성 데이터로 검증) -> 이 스크립트(실제 데이터 적용)

어제(Zenodo)는 "우리 MCMC 결과 vs 원저자 Mathematica 참값"을
대조했다면, 오늘은 원저자의 방법(circle fit) 자체를 우리가 그대로
재구현해서 돌리는 것이므로, 비교 대상이 되는 별도의 "원저자 참값"은
없습니다 - 대신 결과가 물리적으로 타당한 범위인지(예: Ql, Qc가
극단적으로 크거나 작지 않은지, Qi_min<Qi_max인지 등)로 검증합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.getcwd())
import fano_loader
import fano_models


# =========================================================
# STEP 0. 분석자 설정값 (aa_ 접두사)
# =========================================================
aa_drive_folder = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module01/Fano/overcoupled/'
aa_filename = 'resonator_1_powersweep_overcoupled.npz'
aa_output_dir = './outputs/'
os.makedirs(aa_output_dir, exist_ok=True)

aa_slice_index_to_fit = 0
    # [수정] 원저자의 "Example Circle Fit.ipynb"를 확인한 결과, 원저자는
    # -1(최고 전력)이 아니라 0(최저 전력, -80dBm)을 기본 예제로 사용함.
    # 우리가 "신호가 세니까 최고 전력이 유리할 것"이라고 판단했던 것과
    # 반대인데, 이유를 추정해보면: 전력이 높아질수록 비선형 효과(어제
    # 다룬 Duffing 등)나 공진 폭 자체의 변화가 섞여 들어갈 위험이 있고,
    # 원저자는 "가장 깨끗하고 전형적인 선형 공진 형태"를 예제로 삼기
    # 위해 일부러 가장 낮은 전력을 골랐을 가능성이 높음.

aa_n_ports = 1.0
    # [수정] 2.0(notch_port, 2포트 투과) -> 1.0(reflection_port, 1포트 반사).
    # circuit.py의 n_ports 설명: 1=단일 포트 반사(S11), 2=2포트 투과/notch(S21).
    # 원저자 예제 노트북을 직접 확인한 결과, "from circuit import
    # reflection_port as circle_fit"로 reflection_port(n_ports=1)를
    # 쓰고 있었음 - 우리가 처음에 "KIT 논문이 notch-type일 것"이라고
    # 추측했던 게 틀렸던 것. Sij 공식에서 n_ports가 분모에 직접 곱해지는
    # 항이라, 이 값이 틀리면 원의 반지름/형태 전체가 물리적으로 안 맞는
    # 스케일로 계산되어 circle fit이 크게 어긋날 수 있음 - 지난번 겪은
    # roll_off 부족(원을 다 못 그림) 문제의 실제 원인이었을 가능성이 높음.

aa_isolation_db = 15
    # circuit.py 기본값을 그대로 사용. "배경(간섭) 경로가 원래 신호보다
    # 15dB만큼 억제되어 있다"는 가정. 이 값이 클수록(격리가 잘 될수록)
    # Fano 효과가 작다고 가정하는 것이므로, Qi_min/Qi_max 범위가 좁아짐.
    # 실제 장비 스펙을 모르므로 원저자 예제 기본값을 그대로 채택.

print("=" * 60)
print("분석자 설정값(aa_*) 요약:")
for k, v in list(globals().items()):
    if k.startswith("aa_"):
        print(f"  {k:24s} = {v}")
print("=" * 60)


# =========================================================
# STEP 1. 데이터 로드
# =========================================================
full_path = os.path.join(aa_drive_folder, aa_filename)
data = fano_loader.load_fano_npz(full_path)
fano_loader.summarize_loaded_data(data)

f_data_hz = data['freq_hz']
    # fano_models.Sij, autofit 등은 원저자 코드 그대로 "Hz 단위"를
    # 기대함(delay 항이 2*pi*f*delay 형태라, f가 GHz면 delay 단위도
    # 그에 맞춰 바뀌어야 해서 헷갈리기 쉬움 - Hz로 통일해 혼란을 방지).
s21_slice = data['s21'][aa_slice_index_to_fit, :]
power_val = data['power_dbm'][aa_slice_index_to_fit]
cable_delay = data['cable_delay']

print(f"\n피팅 대상 slice: power={power_val:.1f} dBm, "
      f"데이터 포인트 수={len(f_data_hz)}")
print(f"고정으로 사용할 cable_delay: {cable_delay} 초 "
      f"(fano_loader.py가 settings에서 자동 추출한 값)")


# =========================================================
# STEP 2. autofit 실행 - circle fit으로 물리 파라미터 한 번에 추출
# =========================================================
# [주의] autofit은 MCMC와 달리 반복 탐색이 아니라 대수적 계산이므로,
# nsteps/nwalkers 같은 MCMC 관련 설정이 전혀 필요 없음 - 이게 어제
# 실험과 오늘 실험의 코드 구조가 근본적으로 다른 지점.
result = fano_models.autofit(
    f_data_hz, s21_slice,
    n_ports=aa_n_ports,
    fixed_delay=cable_delay,
        # cable_delay를 고정으로 넘김: 장비가 이미 측정해 알려준 값이므로,
        # 어제 MCMC에서 tau를 추정하느라 겪었던 phi-tau 축퇴 문제 자체가
        # 여기서는 원천적으로 발생하지 않음(delay를 추정 대상에서
        # 제외했으므로).
    isolation=aa_isolation_db,
)

print("\n" + "=" * 60)
print("circle fit(autofit) 결과")
print("=" * 60)
print(f"공진 주파수 fr        : {result['fr']/1e9:.6f} GHz")
print(f"Loaded Q (Ql)         : {result['Ql']:.1f}")
print(f"Coupling Q (Qc)       : {result['Qc']:.1f}")
print(f"Internal Q (Qi)       : {result['Qi']:.1f}")
print(f"Fano 위상 phi         : {result['phi']:.4f} rad "
      f"({np.degrees(result['phi']):.2f}도)")
print(f"진폭 스케일 a          : {result['a']:.5f}")
print(f"위상 오프셋 alpha      : {result['alpha']:.4f} rad")
print(f"사용된 delay          : {result['delay']:.4e} 초")

print(f"\n[Fano 간섭에 의한 체계적 불확실성 범위]")
print(f"  Qi_min = {result['Qi_min']:.1f}")
print(f"  Qi_max = {result['Qi_max']:.1f}")
print(f"  (isolation={aa_isolation_db}dB 가정 하에서, 진짜 Qi는 이 범위")
print(f"   안 어딘가에 있다는 뜻 - 딱 한 값으로 확정할 수 없음)")

# 물리적 타당성 자체 점검 (원저자 참값이 없으므로, 결과가 "말이
# 되는지"를 직접 확인하는 것이 이번 검증의 핵심)
print(f"\n[물리적 타당성 자가점검]")
checks = {
    'fr이 스캔 범위 안에 있는가': f_data_hz.min() <= result['fr'] <= f_data_hz.max(),
    'Ql > 0 인가': result['Ql'] > 0,
    'Qc > 0 인가': result['Qc'] > 0,
    'Qi > 0 인가 (물리적으로 음수 Qi는 불가능)': result['Qi'] > 0,
    'Qi_min <= Qi_max 인가': result['Qi_min'] <= result['Qi_max'],
}
for check_name, passed in checks.items():
    print(f"  {'✅' if passed else '⚠️'} {check_name}: {passed}")


# =========================================================
# STEP 3. 시각화 - 실측 데이터와 circle fit 결과를 겹쳐서 확인
# =========================================================
model_fit = fano_models.Sij(
    f_data_hz, result['fr'], result['Ql'], result['Qc'], result['phi'],
    result['a'], result['alpha'], result['delay'], n_ports=aa_n_ports
)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].plot(data['freq_ghz'], np.abs(s21_slice), '.', ms=3, alpha=0.5, label='실측 데이터')
axes[0].plot(data['freq_ghz'], np.abs(model_fit), '-', color='red', lw=2, label='circle fit')
axes[0].set_xlabel('Frequency (GHz)')
axes[0].set_ylabel('|S21|')
axes[0].legend(fontsize=9)
axes[0].set_title('진폭 비교')

axes[1].plot(data['freq_ghz'], np.angle(s21_slice), '.', ms=3, alpha=0.5, label='실측 데이터')
axes[1].plot(data['freq_ghz'], np.angle(model_fit), '-', color='red', lw=2, label='circle fit')
axes[1].set_xlabel('Frequency (GHz)')
axes[1].set_ylabel('phase (rad)')
axes[1].legend(fontsize=9)
axes[1].set_title('위상 비교')

axes[2].plot(np.real(s21_slice), np.imag(s21_slice), '.', ms=3, alpha=0.5, label='실측 데이터')
axes[2].plot(np.real(model_fit), np.imag(model_fit), '-', color='red', lw=2, label='circle fit')
axes[2].set_xlabel('Re(S21)')
axes[2].set_ylabel('Im(S21)')
axes[2].set_aspect('equal')
axes[2].legend(fontsize=9)
axes[2].set_title('복소평면 (원 형태 비교)')

plt.tight_layout()
plt.savefig(os.path.join(aa_output_dir, 'fano_circle_fit_result.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\n비교 그래프 저장 완료: {os.path.join(aa_output_dir, 'fano_circle_fit_result.png')}")
