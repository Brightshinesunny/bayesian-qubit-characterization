# 10_mathematica_fisher_pipeline

[English](README_en.md) | 日本語（本ページ）

01〜09のフォルダがすべてPython（emcee/scipy）ベースであるのに
対し、このフォルダは**「重力波物理学のベイズ統計・フィッシャー
行列の方法論を量子ビット信号に適用する」という同じアイデアを
Mathematica（Wolfram Language）で独立に実装**した記録です。

## なぜ別のツールでやり直したのか

Mathematicaは記号計算（symbolic computation）と統計分布の扱いに
強みがあり、理論中心の研究グループでは今でも広く使われています。
同じ目標（フィッシャー行列に基づくパラメータ不確かさの推定）を
PythonとMathematicaの両方で実装してみたこと自体が、特定のツールに
依存しない方法論的理解を示す根拠になります。

## 対象とする信号モデル

01〜03、07〜09フォルダで扱った avoided-crossing とは異なり、
ここでは**減衰振動（damped oscillation）モデル**を扱います：

```
signalModel(t, A, gamma, omega) = A * Exp(-gamma*t) * Cos(omega*t)
```

これは量子ビットの**T2コヒーレンス減衰＋ラビ振動（Rabi
oscillation）**を表す標準モデルです。コードのコメントには
「重力波のstrainデータ d(t) = h(t) + n(t)」という表現がそのまま
残っており、これはGW波形フィッティングで用いていた信号＋ノイズ
モデル（d = h + n）の枠組みを、そのまま量子ビット信号に移植した
痕跡です。

## ファイル構成（内容に基づき順序を整理、元のファイル名は韓国語）

| ファイル | 元のファイル名 | 内容 |
|---|---|---|
| `01_single_qubit_fisher_prototype.nb` | 초전도.nb | 最初のプロトタイプ：単一量子ビットの減衰振動信号＋フィッシャー行列 |
| `02_qubit_fisher_pipeline_precision.nb` | (초전도 큐빗 정밀 피셔 파이프라인) | 関数化：`QubitFisherFixedSNR`、`RunQubitFisherPipeline` など |
| `03_mock_data_fisher_validation.nb` | (가짜 데이터 Mock Data 생성 및 피셔행렬 검증) | `NonlinearModelFit` による検証、最も規模の大きいバージョン |
| `04_mock_data_multiqubit_classification.nb` | (Mathematica 기반 Mock Data 생성 및 분류 전처리) | `MultiQubitFisherFixedSNR`、`RunMultiQubitPipeline` — 複数量子ビットへの拡張 |
| `05_psd_two_qubit_cross_coupling.nb` | (확장 코드: PSD, 2-큐빗 교차 결합) | パワースペクトル密度（PSD）＋2量子ビット交差結合の拡張 |
| `06_prior_informed_map_4d_fisher_report.nb` | (Prior가 적용된 MAP 4D 피셔 정밀 분석 리포트) | `sigmaGamma2Prior` — prior情報を組み込んだ4次元MAP推定 |
| `07_systematic_bias_fisher_pipeline.nb` | (초전도 큐빗 피셔 + Systematic Bias 연산 파이프라인) | 系統誤差（systematic bias）まで含む最終拡張版、最大サイズのファイル |

## 閲覧時の注意

`.nb` ファイルはプレーンテキスト（ASCII、Wolfram ソース形式）
なので、GitHub上でdiff・コードレビューは可能ですが、**GitHubは
ノートブックをレンダリングしてプレビュー表示することはしません**
（`.ipynb` とは異なります）。内容をそのまま見るには、Mathematica
または無料の Wolfram Player で開く必要があります。

## 本日のPython作業との関係

このMathematicaパイプラインと01〜09のPythonパイプラインは、
**異なる信号モデル**（減衰振動 vs avoided-crossing）を扱っています
が、**核となる方法論**（フィッシャー行列、prior、多パラメータ
不確かさ推定）は共通しています。同じ統計的な考え方を、2つの
異なるツール・2つの異なる信号モデルに一貫して適用した事例として
見ることができます。
