# 超伝導量子ビットデータ解析 — マスターインデックス

[English](README_en.md) | 日本語（本ページ）

全作業（130以上のファイル）をプロジェクトごとに整理したものです。
**どのファイルから見ればよいか分からない
場合は、まずこの文書をお読みください。**

---

## 0. 最初に読むもの — 要約文書4点

| ファイル | 内容 |
|---|---|
| `09_summaries/SUMMARY_fano_avoided_tls_physics_and_stats.py` | Fano/avoided-crossing/TLS の3つの物理系における物理・統計手法の対応関係 |
| `09_summaries/SUMMARY_statistical_toolkit_playbook.py` | Gaussian/Robust/MCMC/フィッシャー行列などツール別まとめ、標準作業手順8段階 |
| `09_summaries/SUMMARY_catalog_to_raw_mapping_journey.m` | TLSカタログ↔raw マッピングの全過程（v1〜v4、列4・列17の最終判断） |
| `09_summaries/SUMMARY_robots_performance_and_full_journey.py` | 検証ボット8種の実戦性能評価＋全体行程の統合版 |

---

## 01_zenodo_baseline — 基礎パイプライン

目標：モックデータで avoided-crossing（fr, g, kappa）のベイズ
パイプラインを確立し、その後実際のZenodo公開データ（Sett et al.,
PRX Quantum 5, 010327 (2024) — `DATA_SOURCES.md` 参照）に適用。
**フォルダ内には v0（最初の単一スクリプト）→ core（モジュール化）
→ mock_data_validation（検証）→ real_data_application（実データ
適用）という時系列4段階が収められています。詳細は
`01_zenodo_baseline/README.md` を参照してください。**

**結論**：`1j` 規約のバグを確定修正（`2j` を使うと結合強度が実際の
半分に過小評価される）。Gaussian/Robust比較パイプラインを確立 →
以降すべてのプロジェクトの土台となった。

---

## 02_fano_resonator — Fano共振器、circle fit vs ベイズ/フィッシャー

目標：9個の共振器（overcoupled + undercoupled）のQi/Qc/Qlを推定。

**主要ファイル**（残りの `fano_*.py` はほとんどがこの過程の試行錯誤）：
- `fano_final_pipeline.py`、`fano_final_estimation_with_corner.py`：最終確定パイプライン
- `fano_undercoupled_summary.py`：undercoupled 8個の結合比（Ql/Qc）vs 信頼区間方向の分析
- `fano_noise_diagnosis.py`、`fano_correlated_noise*.py`：残差自己相関診断（AR(1)/compound symmetryを試みたが最終的に断念）

**最終結論**：circle fitは初期値の安全網としてのみ使用し、Qiは
ベイズ/フィッシャー法で推定すべき。5%外れ値除去＋Gaussian/Robust
交差検証が最も安定。undercoupledについては、「undercoupledか
否か」よりも Ql/Qc 比（結合強度）こそが信頼区間パターンの真の
決定要因である。

---

## 03_avoided_crossing_tls — avoided-crossing 復習＋TLS拡張

目標：1日目のavoided-crossingモデルをTLS swap spectroscopyの
2Dデータに適用。

```
tls_avcross_models.py       # V0再中心化された2D avoided-crossingモデル
tls_avcross_likelihood.py   # 2D実数値の汎用尤度関数
tls_avcross_fit_seg12.py    # seg12の実践フィッティング（multi-start+MCMC）
tls_avcross_visual_check.py # 3候補（ピクセル追跡/STEP2/STEP3）の目視検証
tls_avcross_pipeline.py     # 再利用可能な関数群＋水平多重線代替モデル
```
**最終結論**：目視検証の結果、seg12は対角線（avoided-crossing）
ではなく水平多重帯構造であることが判明 — そもそもの仮定
（avoided-crossing）自体が誤りだった。ピクセル追跡（109 MHz/V）と
2DモデルMCMC（0.6〜1.7 MHz/V）の100倍の不一致は、「仮定が誤って
いた」ことで解消された。

---

## 04_tls_data_structure — TLSデータ構造の解読

HDF5（v7.3）構造の最初の探索から `tabledata2`（13行）の完全解読まで。
- `tls_decode_tabledata2.py`：スケール分析による行の意味の逆算
- `tls_catalog_final_estimation.py`、`tls_noise_removal_bayesian_full.py`：TLSカタログ20件のGaussian/Robust＋フィッシャー行列分析
- `tls_position_quick_attempt.py`：シミュレーションなしの簡易加重重心位置推定

**確定事項**：行0, 2, 9, 11＝結合強度（4電極）、行4＝TLS周波数。
行1, 3, 10, 12は「電圧」ではない（後のセクションで確定）。

---

## 05_tls_noise_diagnosis_dip_tracking — 系統誤差診断＋ディップ追跡

ピクセルディップ追跡アルゴリズムを5回にわたり修正（インデックス
→非線形マッピング→交差点→停滞検出）。`tls_seg12_v5_precrossing_only.py`
が最終成功版（Gaussian/Robustの一致率0.1%）。系統誤差4件を確定：
フラックス基準3件（0.045, −0.030, −0.065）＋5.168 GHz。

---

## 06_tls_multi_file_integration — TLS3ファイルの統合

Segments 1-200／200-400／400-640の3ファイル（TLSカタログ計59件）。
`tls_heterogeneity_check.py`：ファイル間の異質性検定 → 棄却
（均質、単なる統計的変動であった）。

---

## 07_catalog_to_raw_mapping — カタログ↔raw 逆マッピング

**重要：v1→v4の順に失敗原因を絞り込んでいく過程そのものが
最大の学びです。**
- v1：行1/3/10/12＝電圧という仮定 → 物理的に不可能な値（−151Vなど）→ 仮定を放棄
- v2：周波数＋主電極でのマッチング → フィルタが実質的に無意味であることが判明（全体品質スコアが汚染され、偽の1位を選出）
- v3：局所品質スコアと全体品質スコアを分離＋A/B/Cケース分類を導入 → 偽の1位を正確に排除
- v4：4電極すべてを体系的に確認 → 列4は4電極・639セグメント全てでAケース0件
- `tls_multi_column_scan.py`：カタログ19件全てをスキャン → raw と一致したのは15.8%のみ、列17が支配的
- `tls_column17_visual_check.py`：連続性追跡の結果、列17もほぼ水平（0.36 MHz/V、カタログの452 MHz/Vとは無関係）

**最終結論**（`SUMMARY_catalog_to_raw_mapping_journey.m` 第6部参照）：
カタログ項目の大部分（84%）は、原著者による複数セグメント連結
アルゴリズムの領域にあり、われわれの単一フレーム検索では再現
不可能。列17の5.096 GHz帯は系統構造である可能性が高い（確定では
ない）。

---

## 08_validation_pipeline — 検証パイプライン

8種の検証ボット（prior妥当性チェック／パラメータ復元／multi-start
一貫性／パラメータ相関／残差自己相関／Gaussian-Robust交差検証／
品質スコア／総合診断）。実戦性能は
`09_summaries/SUMMARY_robots_performance_and_full_journey.py`
第1部を参照 — 6種は設計通り即座に成功、2種（Gaussian-Robust比較、
品質スコア）は実戦投入時に再設計が必要だった（特に品質スコアは
「全体」と「局所」を区別できず、カタログマッピング失敗の根本原因
となったが、v3で修正）。

---

## 10_mathematica_fisher_pipeline — Mathematicaベースの独立したフィッシャーパイプライン

01〜09がすべてPythonであるのに対し、これは同じGW由来のベイズ/
フィッシャー方法論を **Mathematica（Wolfram Language）で独立に
実装**した記録です。減衰振動（T2緩和＋ラビ振動）信号モデルに
フィッシャー行列解析を適用し、単一量子ビット→多重量子ビット→
PSD/2量子ビット交差結合→prior込み4D MAP→系統誤差込みまで、
7段階にわたって拡張しています。詳細は
`10_mathematica_fisher_pipeline/README.md` を参照してください。

---

## 11_results_figures — 実測データ再実行結果図（事後追加）

`02_fano_resonator`、`05_tls_noise_diagnosis_dip_tracking`プロジェクト
の核心的発見を、実際の元データ（Fano共振器9個のnpzファイル、TLSの
raw matファイル）で再実行して得た図3枚。9個の共振器全てにおける
臨界結合付近の|Qi|発散、最も極端な発散事例のcircle fit例、TLS raw
segmentの系統誤差バンド。韓国語・日本語README付き。

## 12_bugfixes_pending — 検討済み、再実行待ちのバグ修正版

`02_fano_resonator`のFano不確かさ計算バグ（前提が破れたときに
不確かさがゼロと報告されていた問題）の修正、`01`/`03`のFano補正
モデルバグ（信号を最大5倍縮小させていた問題）の修正、avoided-crossing
ストレステストパイプラインの堅牢性改善11件、およびその結果をまとめた
注入試験の総合報告書（injection_test_JP.md）を含む。まだ本体コードに
反映されていない状態。

## ファイルが多すぎるときの早見ガイド

1. **今日の物理・統計の全体像を掴む**：SUMMARY文書4点を順に読む
2. **再利用できるコード**：`08_validation_pipeline/tls_avcross_pipeline.py`（avoided-crossing）、
   `08_validation_pipeline/validation_pipeline.py`（検証）、
   `02_fano_resonator/fano_final_pipeline.py`（Fano）
3. **「なぜこうなったのか」という経緯が気になる場合**：各セクションの
   v1→v4のような順番をたどれば、試行錯誤の論理がそのまま見えます
