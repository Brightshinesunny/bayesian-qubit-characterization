# 01_zenodo_baseline

[English](README_en.md) | 日本語（本ページ）

本リポジトリで最初に取り組んだプロジェクトです。超伝導量子ビットの
avoided-crossing（回避交差）S21スペクトロスコピーに対する、ベイズ統計・
フィッシャー情報行列に基づく解析パイプラインです。モックデータでの
パイプライン検証から始まり、実際のZenodo公開データ（"Emergent
Macroscopic Bistability Induced by a Single Superconducting Qubit"、
DOI 10.5281/zenodo.10518320 — 詳細は `../DATA_SOURCES.md` を参照）
への適用まで、時系列に沿った4段階の進化過程を収めています。

## フォルダ構成（時系列4段階）

```
01_zenodo_baseline/
├── v0_first_prototype/         # 第1段階：モジュール化前の単一スクリプト
│   ├── anticrossing_full_flux_sweep.py   # 最初のバッチMCMCフィッティング
│   ├── generate_noisy_mock_data.py       # 1/fノイズ+ドリフト+外れ値入りモックデータ
│   ├── threshold_methods_comparison.py   # 安定推定しきい値の3手法比較
│   └── delta_vs_g_uncertainty.py         # |Δ| vs g不確かさの診断
│
├── core/                        # 第2段階：再利用可能な部品へのモジュール化
│   ├── models.py                # フォワードモデル（avoided-crossing S21）
│   ├── likelihood.py            # prior/likelihood（gaussian/robust）
│   ├── mcmc_pipeline.py         # emcee実行部（単一/バッチ、warm-start）
│   ├── diagnostics.py           # ノイズ推定、しきい値検出
│   ├── bayesian_toolkit.py      # 確信区間、R-hat、AIC/BICなど
│   └── fisher_matrix.py         # フィッシャー行列による誤差推定（MCMCの代替）
│
├── mock_data_validation/        # 第3段階：モックデータによるパイプライン検証
│   ├── mock_data.py             # クリーン信号+ノイズ注入ジェネレータ
│   ├── example_run.py           # core6モジュールを組み合わせた実行例
│   ├── example_run_bay.py       # + bayesian_toolkitによる詳細解析版
│   └── compare_gaussian_vs_robust.py  # 精度（accuracy）比較
│
└── real_data_application/       # 第4段階：実際のZenodoデータへの適用
    ├── zenodo_loader.py, zenodo_load_real_data.py
    ├── zenodo_models.py, zenodo_likelihood.py, zenodo_mcmc_pipeline.py
    ├── zenodo_bayesian_toolkit.py, zenodo_diagnostics.py, zenodo_fisher_matrix.py
    ├── zenodo_fit_and_compare.py, zenodo_joint_fit.py
    ├── zenodo_multi_file_check.py, zenodo_verify_parser_equivalence.py
```

## 設計方針

**変わる部分（物理系ごとに入れ替え）**
- `core/models.py`：新しい物理系を扱う際に関数を1つ追加
- `mock_data_validation/mock_data.py` のクリーン信号生成ロジック
- `core/likelihood.py` の `_theta_to_kwargs`（パラメータの種類が変われば
  マッピングも変わる）

**ほとんど変わらない部分（骨格）**
- `core/mcmc_pipeline.py`：サンプラー実行、warm-start、収束診断ロジック
- `core/diagnostics.py`：ノイズ推定、しきい値検出の方法論
- `core/likelihood.py` のファクトリ関数構造（`make_uniform_log_prior`、
  `make_log_probability`）

## 拡張シナリオ

| 状況 | 変更箇所 |
|---|---|
| 別の物理系（例：3量子ビット） | `core/models.py` に関数を追加、`example_run.py` のSTEP1〜4を入れ替え |
| 外れ値に頑健な解析 | `example_run.py` で `aa_likelihood_type = 'robust'` に変更 |
| 異なるノイズ特性の実験 | `example_run.py` の `aa_flicker_level`、`aa_drift_amplitude`、`aa_outlier_probability` を調整 |
| 階層モデル（パラメータ共有） | `core/mcmc_pipeline.py` を拡張し、複数sliceを同時にフィットする新関数が必要 |
| 別のサンプラー（nested sampling） | `core/mcmc_pipeline.py` に `run_single_nested()` のような並行関数を追加 |

## 使い方

```bash
cd 01_zenodo_baseline
# coreモジュールをインポートできるように、mock_data_validation
# または real_data_application 内で実行するか、PYTHONPATHにcoreを追加
PYTHONPATH=core python mock_data_validation/example_run.py
```

`example_run.py` 冒頭の `aa_` 変数を調整することで、ノイズの種類の
オン/オフや尤度関数（gaussian/robust）の切り替えが可能です。

## 検証履歴

- `core/models.py` の `s21_anticrossing_model`：モックデータで
  fr、g、kappa のすべてが真値と誤差範囲内で一致することを確認
  （`1j` 規約を確定 — `2j` を使うと結合強度が実際の半分に
  過小評価されるバグがあった）
- `core/diagnostics.py` のしきい値検出3手法の比較：crossing方式と
  inflection方式は、互いに異なるアプローチであるにもかかわらず
  同じしきい値（|Δ| ≈ 0.29 GHz）に収束。discrete-label方式（方式A）
  は外れ値に弱いことを確認

## データ出典

`real_data_application/` で使用した実測データの出典については、
リポジトリ最上位の `../DATA_SOURCES.md` を参照してください。
