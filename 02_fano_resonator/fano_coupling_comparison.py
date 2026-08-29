"""
fano_coupling_comparison.py
=================================================================
어제 확인한 원저자 예제 노트북의 힌트("undercoupled에서는 Qi
불확실성이 더 낮고 Qc 불확실성이 더 높다, overcoupled는 반대")를
직접 검증합니다. overcoupled(어제 이미 분석 완료)와 undercoupled를
같은 코드로 나란히 circle fit해서, calc_fano_range()가 계산하는
Qi_min~Qi_max, Qc_min~Qc_max 폭을 직접 비교합니다.

[가설 - 왜 이런 경향이 나올 것으로 예상되는가]
Fano 불확실성 범위는 fano_models.py의 calc_fano_range 안에서
"원의 반지름(R)에 얼마나 오차가 있을 수 있는가"로부터 유도됩니다.
  - overcoupled(외부결합 Qc가 작아 강함): 원이 (1,0)에서 상대적으로
    멀리 떨어진 큰 원을 그림 -> 반지름 오차가 Qi 계산식(Ql/(1-R))의
    분모(1-R)에 큰 영향을 못 미침 -> Qi_min~Qi_max 폭이 상대적으로
    좁고, 대신 Qc=Ql/R의 분모(R)가 그 오차에 더 민감 -> Qc 불확실성이 큼
  - undercoupled(외부결합 Qc가 커서 약함): 원이 (1,0)에 가까운 작은
    원을 그림 -> 정반대 경향
이건 오늘 이미 만든 도구(circle fit, calc_fano_range)를 그대로
재사용해서 "물리적 직관을 데이터로 검증"하는 작업입니다 - 새 모델을
만들 필요가 없습니다.
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
aa_base_folder = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module01/Fano/'
aa_output_dir = './outputs/'
os.makedirs(aa_output_dir, exist_ok=True)

aa_resonator_id = 1
    # 어제 분석한 것과 같은 resonator_1로 두 폴더(overcoupled/undercoupled)를
    # 비교. 같은 번호라고 해서 물리적으로 "같은 칩"이라는 보장은 없을 수
    # 있지만(원저자가 실험 편의상 같은 번호를 붙였을 가능성), 적어도
    # 같은 조건에서 나온 비교이므로 의미 있는 대조가 됨.

aa_slice_index_to_fit = 0   # 어제와 동일하게 최저 전력 (원저자 예제와 일치)
aa_n_ports = 1.0             # reflection_port
aa_isolation_db = 15

print("=" * 60)
print("분석자 설정값(aa_*) 요약:")
for k, v in list(globals().items()):
    if k.startswith("aa_"):
        print(f"  {k:24s} = {v}")
print("=" * 60)


# =========================================================
# STEP 1. 두 조건(overcoupled/undercoupled) 파일 로드 및 circle fit
# =========================================================
def load_and_fit(coupling_type):
    """
    coupling_type: 'overcoupled' 또는 'undercoupled'.
    해당 폴더에서 resonator_{id}_powersweep_{coupling_type}.npz를
    읽어서 circle fit까지 수행하고 결과를 반환.
    """
    filename = f'resonator_{aa_resonator_id}_powersweep_{coupling_type}.npz'
    full_path = os.path.join(aa_base_folder, coupling_type, filename)

    data = fano_loader.load_fano_npz(full_path)
    f_data_hz = data['freq_hz']
    s21_slice = data['s21'][aa_slice_index_to_fit, :]
    power_val = data['power_dbm'][aa_slice_index_to_fit]
    cable_delay = data['cable_delay']

    result = fano_models.autofit(
        f_data_hz, s21_slice, n_ports=aa_n_ports,
        fixed_delay=cable_delay, isolation=aa_isolation_db
    )
    result['power_val'] = power_val
    result['n_power_slices'] = data['power_dbm'].shape[0]
    return result


print(f"\nresonator_{aa_resonator_id}: overcoupled / undercoupled 각각 circle fit 실행 중...")
result_over = load_and_fit('overcoupled')
result_under = load_and_fit('undercoupled')


# =========================================================
# STEP 2. Qi, Qc 불확실성 폭 비교
# =========================================================
def summarize(result, label):
    Qi_width = result['Qi_max'] - result['Qi_min']
    Qc_width = result['Qc_max'] - result['Qc_min']
    print(f"\n--- {label} (power={result['power_val']:.1f}dBm, "
          f"{result['n_power_slices']}개 slice 중 1개) ---")
    print(f"  fr={result['fr']/1e9:.6f} GHz, Ql={result['Ql']:.1f}, "
          f"Qc={result['Qc']:.1f}, Qi={result['Qi']:.1f}")
    print(f"  phi={result['phi']:.4f} rad ({np.degrees(result['phi']):.2f}도)")
    print(f"  Qi 범위: [{result['Qi_min']:.1f}, {result['Qi_max']:.1f}]  "
          f"(폭={Qi_width:.1f})")
    print(f"  Qc 범위: [{result['Qc_min']:.1f}, {result['Qc_max']:.1f}]  "
          f"(폭={Qc_width:.1f})")
    return Qi_width, Qc_width

print("\n" + "=" * 60)
print("결과 비교")
print("=" * 60)
Qi_width_over, Qc_width_over = summarize(result_over, "OVERCOUPLED")
Qi_width_under, Qc_width_under = summarize(result_under, "UNDERCOUPLED")

print("\n" + "=" * 60)
print("가설 검증: '언더커플드는 Qi 불확실성 낮고 Qc 불확실성 높다'")
print("=" * 60)
print(f"Qi 불확실성 폭: overcoupled={Qi_width_over:.1f} vs "
      f"undercoupled={Qi_width_under:.1f}")
print(f"  -> undercoupled가 더 작은가? {Qi_width_under < Qi_width_over}")
print(f"Qc 불확실성 폭: overcoupled={Qc_width_over:.1f} vs "
      f"undercoupled={Qc_width_under:.1f}")
print(f"  -> undercoupled가 더 큰가? {Qc_width_under > Qc_width_over}")

hypothesis_confirmed = (Qi_width_under < Qi_width_over) and (Qc_width_under > Qc_width_over)
print(f"\n가설 전체 확인 여부: {hypothesis_confirmed}")


# =========================================================
# STEP 3. 시각화 - 두 조건의 복소평면 원 형태를 나란히 비교
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

for ax, result, label, color in [
    (axes[0], result_over, 'Overcoupled', 'tab:blue'),
    (axes[1], result_under, 'Undercoupled', 'tab:orange'),
]:
    model_fit = fano_models.Sij(
        np.linspace(result['fr']*0.9995, result['fr']*1.0005, 1001),
        result['fr'], result['Ql'], result['Qc'], result['phi'],
        result['a'], result['alpha'], result['delay'], n_ports=aa_n_ports
    )
    ax.plot(np.real(model_fit), np.imag(model_fit), '-', color=color, lw=2)
    ax.plot(1, 0, 'k+', ms=12, mew=2, label='(1,0) 기준점')
    ax.set_xlabel('Re(S21)')
    ax.set_ylabel('Im(S21)')
    ax.set_aspect('equal')
    ax.set_title(f'{label}\nQi=[{result["Qi_min"]:.0f},{result["Qi_max"]:.0f}], '
                 f'Qc=[{result["Qc_min"]:.0f},{result["Qc_max"]:.0f}]', fontsize=9)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(aa_output_dir, 'fano_coupling_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\n시각화 저장 완료: {os.path.join(aa_output_dir, 'fano_coupling_comparison.png')}")
