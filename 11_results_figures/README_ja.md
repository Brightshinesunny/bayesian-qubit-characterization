# 11_results_figures

[한국어](README.md) | 日本語（本ページ）

このフォルダの3枚の図は、本日の文書（README、PORTFOLIO_DETAILED.md）で
述べた核心的発見を、実際の生データを用いて再実行して得た結果です。
作り出した例示ではなく、アップロードされた実測データ（Fano共振器9個の
npzファイル、TLSのraw segment matファイル）と、原著者の方法論をそのまま
移植したコード（`fano_models.py`、Rieger & Günzler / Probst circle-fit）
を用いて直接再計算したものです。

---

## fig1_qi_divergence.png — 臨界結合付近での内部Q値の発散

**02_fano_resonatorプロジェクトの核心的結論を、実データ9個全てで再現。**

9個の共振器（overcoupled 5個、undercoupled 4個）それぞれの中間パワー
地点でcircle fitを実行してQl、Qcを得て、Ql/Qc比（1に近いほど臨界結合）
に対して|Qi|を対数スケールでプロットしました。Undercoupledの共振器
ほど臨界結合に近く（Ql/Qcが1に近接）、それだけ|Qi|が数万から数百万
まで指数関数的に発散することが、実データで確認されます。

## fig2_circle_fit_example.png — Circle Fitの例（最も極端な発散事例）

fig1で最も大きく発散したresonator_4（undercoupled、Ql/Qc=1.074）を
選び、左側に複素平面上の円フィット（生データ点＋フィットされた円）、
右側に周波数に対する|S21|の大きさとフィット曲線を並べて描きました。
Ql≈181,500、Qc≈169,000と両者が非常に近く、|Qi|が246万まで跳ね上がる
ことが視覚的に確認できます。

## fig3_systematic_bands.png — 系統誤差候補バンド（TLS生データ）

TLSのraw segment一つ（`quadspec_Segments1to200_20TLSfitted.mat`内部
の参照データ）の2次元振幅マップを描いたものです。明るい縦バンドが
3本、電圧スイープ軸（y）とは無関係に特定の周波数地点（x）に固定
されているのが見えますが、これは03_avoided_crossing_tls /
05_tls_noise_diagnosisプロジェクトで述べた「ゲート電圧と無関係な
系統誤差バンド」と同種のパターンを示す再現例です。（この図の特定の
3本のバンドが、当時確定した4個の系統誤差と正確に同一の周波数か
どうかは、今回の再実行では別途照合していません — 「同じ現象が
実データでこのように見える」ことを示す例としてご覧ください。）

---

## 再現方法

元データとコードは`01_zenodo_baseline/`、`02_fano_resonator/`フォルダ
のスクリプトを参照してください。これらの図は次の方法で生成しました：

```python
from fano_models import autofit  # 02_fano_resonator/ のcircle-fit移植コード
import numpy as np

d = np.load('resonator_N_powersweep_XXX.npz', allow_pickle=True)
z_data = d['amplitude'][idx] * np.exp(1j * d['phase'][idx])
result = autofit(d['frequency'], z_data)
# result['Ql'], result['Qc'], result['Qi'] を使用
```
