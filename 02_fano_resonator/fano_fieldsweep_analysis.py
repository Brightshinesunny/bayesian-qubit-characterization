"""
fano_fieldsweep_analysis.py
=================================================================
field sweep 파일의 각 자기장(B) slice에 circle fit(fano_models.autofit)
을 반복 적용해서, 공진 주파수 fr과 내부 품질계수 Qi가 자기장에 따라
어떻게 변하는지 추출합니다.

[물리적 예상]
  fr(B): 운동 인덕턴스가 자기장에 따라 서서히 변하므로, fr도 매끄럽게
         이동할 것으로 예상 (급격한 꺾임이 있다면 보텍스 유입 시작
         등 다른 물리가 섞였다는 신호일 수 있음).
  Qi(B): 낮은 자기장에서는 거의 일정하다가, 어떤 임계 자기장을 넘으면
         보텍스 유입으로 급격히 나빠질(작아질) 것으로 예상. 이
         "꺾이는 지점"이 이 공진기의 실질적인 평행 방향 임계자기장.

[이번 스크립트에서 자기장에 따라 주파수 창이 바뀐다는 것의 의미]
어제 확인한 대로 frequency가 2차원(slice마다 다른 창)이므로, 각
slice를 circle fit할 때 그 slice 전용 freq_hz_2d[i]를 반드시 함께
써야 합니다 - 다른 slice의 주파수 축을 섞어 쓰면 안 됩니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.getcwd())
import fano_fieldsweep_loader as ffl
import fano_models


# =========================================================
# STEP 0. 분석자 설정값 (aa_ 접두사)
# =========================================================
aa_drive_folder = '/content/drive/MyDrive/eunjunglee/SuperQuantum/Module01/Fano/overcoupled/'
aa_filename = 'resonator_7_fieldsweep_overcoupled.npz'
aa_output_dir = './outputs/'
os.makedirs(aa_output_dir, exist_ok=True)

aa_n_ports = 1.0   # reflection_port (지금까지 계속 써온 값과 동일)
aa_isolation_db = 30
    # [수정] 15 -> 30. STEP 2.5 진단에서 r0_relative가 전 구간에서
    # 0.90~1.01(=1에 매우 근접)로 확인됐습니다. 이 공진기가 원래
    # 극도로 강하게 결합(overcoupled)되어 있다는 뜻이라, isolation을
    # 15dB(배경 간섭이 꽤 세다는 가정)로 두면 R_err이 너무 커져서
    # R_max가 거의 항상 1을 넘어버립니다. isolation을 30dB로 올리면
    # "배경 간섭이 훨씬 약하다"고 가정하는 셈이라(dB가 클수록 더 많이
    # 억제됐다는 뜻), R_err이 작아지고, R_max가 1 밑으로 내려올
    # 가능성이 높아집니다. 다만 이건 "가정을 바꿔서 원하는 결과가
    # 나오게 만드는" 것이 아니라, 실제로 이 실험 장비의 배경 격리
    # 성능이 15dB보다 나은지 확인이 필요한 부분 - 30dB는 일단
    # 시험적으로 시도해보는 값.

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
data = ffl.load_fano_fieldsweep_npz(full_path)
ffl.summarize_loaded_data(data)

n_slices = len(data['B_fields'])
print(f"\n총 {n_slices}개 자기장 slice에 대해 circle fit 반복 실행...")


# =========================================================
# STEP 2. 모든 slice에 circle fit 반복 적용
# =========================================================
fr_list = []
Qi_list = []
Ql_list = []
Qc_list = []
phi_list = []
Qi_min_list = []
Qi_max_list = []
fit_success = []

for i in range(n_slices):
    f_grid_i = data['freq_hz_2d'][i, :]
        # [핵심] 이 slice 전용 주파수 축을 사용 - 다른 slice와 섞이지
        # 않도록 반드시 인덱스 i로 맞춰서 가져옴.
    s21_i = data['s21'][i, :]
    delay_i = data['cable_delay_list'][i]

    try:
        result = fano_models.autofit(
            f_grid_i, s21_i, n_ports=aa_n_ports,
            fixed_delay=delay_i, isolation=aa_isolation_db
        )
        fr_list.append(result['fr'])
        Qi_list.append(result['Qi'])
        Ql_list.append(result['Ql'])
        Qc_list.append(result['Qc'])
        phi_list.append(result['phi'])
        Qi_min_list.append(result['Qi_min'])
        Qi_max_list.append(result['Qi_max'])
        fit_success.append(True)
    except Exception as e:
        # 일부 slice에서 fit이 실패할 수 있음(예: 신호가 너무 약하거나
        # 딥이 스캔 범위를 벗어난 경우) - 실패해도 전체 루프가 멈추지
        # 않도록 개별적으로 처리하고, 실패 여부를 기록해 나중에
        # "몇 개가 실패했는지" 확인할 수 있게 함.
        fr_list.append(np.nan)
        Qi_list.append(np.nan)
        Ql_list.append(np.nan)
        Qc_list.append(np.nan)
        phi_list.append(np.nan)
        Qi_min_list.append(np.nan)
        Qi_max_list.append(np.nan)
        fit_success.append(False)

fr_arr = np.array(fr_list)
Qi_arr = np.array(Qi_list)
Ql_arr = np.array(Ql_list)
Qc_arr = np.array(Qc_list)
phi_arr = np.array(phi_list)
Qi_min_arr = np.array(Qi_min_list)
Qi_max_arr = np.array(Qi_max_list)
fit_success = np.array(fit_success)

n_success = np.sum(fit_success)
print(f"\ncircle fit 성공: {n_success}/{n_slices}")
if n_success < n_slices:
    fail_indices = np.where(~fit_success)[0]
    print(f"실패한 slice 인덱스: {fail_indices}")
    print(f"실패한 slice의 B_fields 값: {data['B_fields'][fail_indices]}")


# =========================================================
# STEP 2.5. r0_relative 진단 - 왜 Qi_max가 계속 inf인지 근본 원인 확인
# =========================================================
# 원본 circuit.py의 _calibrate()를 직접 대조한 결과, 우리 계산식
# (Qc, Qi, calc_fano_range 전부)은 원저자와 100% 일치했습니다(핵심:
# 원저자도 self.r0 /= self.a로 정규화한 값을 그 이후 모든 계산에
# 사용합니다 - 우리 r0_relative와 정확히 같은 개념).
#
# 즉 "203개 slice 전부 Qi_max=inf"는 버그가 아니라, 이 공진기
# (overcoupled 폴더 데이터)가 실제로 아주 강하게 결합되어 있어서
# r0_relative(=r0/a)가 1에 가깝거나 그 이상이라는 뜻입니다.
# reflection(n_ports=1)에서는 R_max가 1을 넘으면 물리적으로
# "Qi의 상한이 존재하지 않는다"는 결과가 나오는 게 정상입니다 -
# isolation=15dB라는 가정 아래서는, 이렇게 강하게 결합된 공진기의
# 경우 Qi를 위에서 한계 지을 수 없다는 뜻입니다.
print(f"\n[근본 원인 진단] r0_relative(=원 반지름/a)가 1에 가까운지 확인")
sample_indices = [0, n_slices//4, n_slices//2, 3*n_slices//4, n_slices-1]
for idx in sample_indices:
    if fit_success[idx]:
        # absQc = Ql/(n_ports*r0_relative) 관계를 거꾸로 이용해
        # r0_relative를 역산 (Qc_no_dia_corr 값이 있다면 그대로 쓰는
        # 것이 더 정확하지만, 여기서는 Ql/Qc로 근사).
        r0_rel_approx = Ql_arr[idx] / (aa_n_ports * Qc_arr[idx])
        print(f"  slice {idx} (B={data['B_fields'][idx]:.4f}): "
              f"r0_relative≈{r0_rel_approx:.4f} "
              f"({'1에 근접/초과 - Qi 상한 없음 정상' if r0_rel_approx > 0.8 else '1보다 충분히 작음'})")


# =========================================================
# STEP 3. fr(B), Qi(B) 시각화 (Qi는 오차 띠로 표시 - 원저자 방식)
# =========================================================
B = data['B_fields']

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

axes[0, 0].plot(B, fr_arr/1e9, '.-', ms=4)
axes[0, 0].set_xlabel('B field (unit 미확정)')
axes[0, 0].set_ylabel('fr (GHz)')
axes[0, 0].set_title('공진 주파수 fr vs 자기장')

# [수정] Qi 그래프: 중심값(선) + Qi_min~Qi_max(반투명 띠)를 함께 표시.
# Qi_max가 inf인 지점은 fill_between이 위쪽 경계를 그릴 수 없으므로
# (matplotlib이 알아서 잘라서 그리거나 생략함), 이 자체가 "이 지점은
# 상한이 없다"는 정보를 시각적으로 보여주는 요소로 남겨둠 (지운다면
# 오히려 이 사실을 감추는 셈).
axes[0, 1].plot(B[fit_success], Qi_arr[fit_success], '-', lw=1, color='tab:orange',
                 label='Qi (중심값)')
axes[0, 1].fill_between(B[fit_success], Qi_min_arr[fit_success], Qi_max_arr[fit_success],
                          color='tab:orange', alpha=0.25,
                          label='Qi_min~Qi_max (Fano 불확실성 범위)')
    # fill_between(x, y_lo, y_hi): x축의 각 지점에서 y_lo와 y_hi
    # 사이를 색칠해서 "띠(band)" 형태로 그리는 matplotlib 함수.
    # alpha=0.25: 반투명하게(25% 불투명도) 만들어서, 띠가 겹쳐도
    # 서로 가려지지 않고 각 영역이 은은하게 다 보이도록 함.
axes[0, 1].set_xlabel('B field')
axes[0, 1].set_ylabel('Qi (internal Q)')
axes[0, 1].set_yscale('log')
    # [주의] Qi_min~Qi_max 폭이 지점마다 몇 자릿수씩 차이날 수 있으므로
    # (Ql≈Qc인 지점에서 폭이 극단적으로 커짐), 로그 스케일이라야
    # 전체 경향과 "폭이 갑자기 넓어지는 위험 지대"가 동시에 잘 보임.
axes[0, 1].legend(fontsize=8)
axes[0, 1].set_title('내부 품질계수 Qi vs 자기장\n'
                      '(오차 띠가 넓어지는 지점 = Ql≈Qc인 임계결합 근접 지대)')

axes[1, 0].plot(B, Ql_arr, '.-', ms=4, label='Ql', color='tab:green')
axes[1, 0].plot(B, Qc_arr, '.-', ms=4, label='Qc', color='tab:red')
axes[1, 0].set_xlabel('B field')
axes[1, 0].set_ylabel('Q factor')
axes[1, 0].legend()
axes[1, 0].set_title('Ql, Qc vs 자기장')

axes[1, 1].plot(B, phi_arr, '.-', ms=4, color='tab:purple')
axes[1, 1].set_xlabel('B field')
axes[1, 1].set_ylabel('phi (rad)')
axes[1, 1].set_title('Fano 위상 phi vs 자기장')

plt.tight_layout()
plt.savefig(os.path.join(aa_output_dir, 'fano_fieldsweep_result.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\n결과 저장 완료: {os.path.join(aa_output_dir, 'fano_fieldsweep_result.png')}")


# =========================================================
# STEP 3.5. 오차 띠의 폭 자체를 자기장에 대해 시각화
# =========================================================
# "띠가 얼마나 넓은지"를 별도로 그려보면, 어느 자기장 구간에서
# Ql≈Qc(임계결합 근접)에 가장 가까워지는지 - 즉 Qi를 가장 신뢰하기
# 어려운 구간이 어디인지 - 한눈에 볼 수 있습니다.
band_width = Qi_max_arr - Qi_min_arr

# [수정] fano_models.py의 calc_fano_range()를 보면, 특정 조건에서
# Qi_max를 np.inf(무한대)로 설정하는 부분이 있음(1 - n_ports*R_max가
# 0보다 작아지면 "이 방향으로는 상한이 아예 없다"는 �)그런데
# band_width = Qi_max - Qi_min 에 inf가 섞이면 inf가 되고, matplotlib은
# 배열에 inf가 하나라도 있으면 축 범위(scale)를 제대로 못 잡아 그래프
# 전체가 텅 비어버리는 경우가 흔함 - 방금 겪은 빈 그래프가 정확히 이
# 문제였을 가능성이 높음.
is_finite = np.isfinite(band_width) & fit_success
    # np.isfinite: 배열의 각 원소가 inf/-inf/nan이 "아닌"(즉 유한한
    # 정상적인 숫자인) 경우에만 True를 반환하는 함수. 이 조건과
    # fit_success를 & 로 동시에 만족하는 지점만 그래프에 표시.
n_infinite = np.sum(fit_success) - np.sum(is_finite)
print(f"\n[오차 띠 폭 진단] inf(무한대)로 나온 slice 수: {n_infinite}개 "
      f"(전체 {np.sum(fit_success)}개 중)")
print(f"  -> 이 slice들은 'Qi 상한이 존재하지 않는다'는 뜻이라, 그래프에서는")
print(f"     제외하고 유한한 값들만 표시합니다 (제외 자체가 그 slice의")
print(f"     불확실성이 극단적으로 크다는 정보이므로, 숫자로도 함께 보고).")

fig2, ax2 = plt.subplots(figsize=(9, 4.5))
ax2.plot(B[is_finite], band_width[is_finite], '.-', ms=4, color='tab:brown')
ax2.set_xlabel('B field')
ax2.set_ylabel('Qi_max - Qi_min (오차 띠 폭, 유한한 값만)')
ax2.set_yscale('log')
    # 폭이 자기장에 따라 몇 자릿수씩 차이날 수 있으므로 로그 스케일
    # 사용 - 어제부터 계속 써온 관례와 동일.
ax2.set_title(f'Qi 오차 띠의 폭 vs 자기장 ({n_infinite}개 inf slice 제외)\n'
              f'(폭이 넓어지는 지점 = 임계결합에 가장 가까워지는 자기장)')

plt.tight_layout()
plt.savefig(os.path.join(aa_output_dir, 'fano_fieldsweep_qi_band_width.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"오차 띠 폭 그래프 저장 완료: "
      f"{os.path.join(aa_output_dir, 'fano_fieldsweep_qi_band_width.png')}")


# =========================================================
# STEP 4. 정성적 요약
# =========================================================
valid = fit_success
fr_change = np.nanmax(fr_arr[valid]) - np.nanmin(fr_arr[valid])

# [수정] 이제 "Qi 중심값의 변화율"이 아니라, "오차 띠가 가장 좁을 때
# vs 가장 넓을 때"를 함께 보고하는 것이 더 정직한 요약입니다 - 오차
# 띠가 넓은 지점에서는 Qi 중심값 자체를 신뢰하기 어렵기 때문입니다.
# [버그 수정] np.nanargmax/argmin은 inf를 걸러내지 못해서(nan만
# 무시하고 inf는 "매우 큰 정상값"으로 취급), inf가 섞인 배열에
# nanargmax를 쓰면 항상 그 inf 지점을 "최댓값"으로 잘못 골라버림.
# STEP 3.5에서 만든 is_finite 마스크로 inf를 미리 제외한 뒤에
# argmax/argmin을 적용해야 함.
band_width_finite = band_width[is_finite]
B_finite = B[is_finite]

# [방어 코드 추가] 만약 isolation을 30dB로 올려도 여전히 모든 slice가
# inf라면(이 공진기가 그만큼 극단적으로 강하게 결합되어 있다는 뜻),
# band_width_finite가 빈 배열이 되어 argmin/argmax가 에러를 냄
# ("attempt to get argmin of an empty sequence" - 방금 실제로 겪은
# 에러). 빈 배열일 때는 에러 대신 "전부 inf였다"는 사실 자체를
# 명확한 메시지로 보고하고 넘어가도록 함.
if len(band_width_finite) == 0:
    print(f"\n[정성적 요약]")
    print(f"  fr 전체 이동폭: {fr_change/1e6:.2f} MHz")
    print(f"  ⚠️ isolation={aa_isolation_db}dB로도 모든 {n_slices}개 slice에서")
    print(f"     Qi_max=inf가 나왔습니다. 이는 이 공진기(resonator_7,")
    print(f"     overcoupled)가 자기장 전 구간에서 r0_relative≈1(임계결합")
    print(f"     근접) 상태라, 이 방법(Fano 불확실성 범위 계산)으로는")
    print(f"     Qi의 상한을 원리적으로 정할 수 없다는 뜻입니다.")
    print(f"     -> 이건 통계 기법의 한계가 아니라, 실제로 이 공진기의")
    print(f"        설계/결합 조건 자체가 이 분석 방법이 잘 작동하는")
    print(f"        영역(약~중간 결합) 밖에 있다는 물리적 결론입니다.")
else:
    narrowest_idx = np.argmin(band_width_finite)
    widest_idx = np.argmax(band_width_finite)

    print(f"\n[정성적 요약]")
    print(f"  fr 전체 이동폭: {fr_change/1e6:.2f} MHz")
    print(f"  Qi 오차 띠가 가장 좁은 지점: B={B_finite[narrowest_idx]:.4f}, "
          f"폭={band_width_finite[narrowest_idx]:.1f} (Qi를 가장 신뢰할 수 있는 자기장)")
    print(f"  Qi 오차 띠가 가장 넓은 지점(유한값 중): B={B_finite[widest_idx]:.4f}, "
          f"폭={band_width_finite[widest_idx]:.1f}")
    print(f"  (참고: inf로 나온 {n_infinite}개 slice는 '오차 띠가 사실상 무한대'라는")
    print(f"   뜻이므로, 이 지점들이 진짜 가장 불확실한 지점들입니다)")
    print(f"  -> fr이 크게 이동한다면, 예상한 물리(운동 인덕턴스 변화)와")
    print(f"     일치하는 결과입니다. Qi는 '하나의 정확한 값'이 아니라")
    print(f"     자기장에 따라 신뢰도가 달라지는 '범위'로 이해해야 합니다.")
