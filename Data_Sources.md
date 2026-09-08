## Data Sources

Raw data files are **not included** in this repository. Download them
directly from the sources below before running the scripts.

原データファイルは本リポジトリには含まれていません。スクリプトを
実行する前に、下記のソースから直接ダウンロードしてください。

---

### 1. Fano Resonator Data (Circle Fit / Qi–Qc–Ql Estimation)

**Rieger, D., Günzler, S., Spiecker, M., Nambisan, A., Wernsdorfer, W., Pop, I. M.**
"Fano Interference in Microwave Resonator Measurements"
Karlsruhe Institute of Technology (KIT), 2023.
Supplement to arXiv:2209.03036.

- Zenodo: https://zenodo.org/records/7767046
- DOI: 10.5281/zenodo.7767046
- License: CC BY 4.0
- Format: NumPy `.npz`

**日本語**: KIT（カールスルーエ工科大学）の Rieger, Günzler らによる
Fano干渉共振器測定データ。論文 arXiv:2209.03036 の付属データセット。
円フィット（circle fit）法の実装例も含まれています。

---

### 2. TLS Position-Mapping Data (Swap Spectroscopy / Gate-Voltage Stark Tuning)

**Lisenfeld, J. et al.**
"Mapping the positions of Two-Level-Systems on the surface of a
superconducting transmon qubit"
npj Quantum Information 12, 80 (2026). arXiv:2511.05365.

- Zenodo: https://zenodo.org/records/18847452
- DOI: 10.5281/zenodo.18847452
- Format: MATLAB v7.3 (`.mat`, HDF5-based)
- Files used: `quadspec_Segments{1-200,200-400,400-640}_20TLSfitted.mat`

**日本語**: Lisenfeld らによる、超伝導トランズモン量子ビット表面上の
TLS（二準位系）位置マッピングデータ。ゲート電極電圧でTLS共鳴周波数を
Stark シフトさせ、swap spectroscopy で結合強度を測定した生データ。

---

*Please cite the original papers/datasets above when reusing this
analysis code. これらのデータを再利用する際は、必ず元論文・データ
セットを引用してください。*
