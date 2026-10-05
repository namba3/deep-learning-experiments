# Optimizer performance design

更新日: 2026-09-14

この文書は`AdamW`、`AdamW-SF`、`CAME`、`APOLLO`のstep高速化について、2026-09-14時点の設計・検証案を記録する。現行のstate dtype契約は[optimizer仕様](../optimizers.md)を参照する。

これは2026-09-14時点の設計・検証計画snapshotです。提案された実装段階や未完了項目はその後変わっている可能性があります。現在のbackend、数値契約、既定値はコードと[`optimizers.md`](../optimizers.md)を参照してください。

## 目標と非目標

目標は、現在の更新式を保ったまま、次を削減することである。

- CUDA kernel launch 数
- full-size temporary tensor の生成・読み書き
- Python の parameter 単位 dispatch
- 同じデータを複数回 global memory から読む処理

次は本作業の非目標とする。

- CAME/APOLLO の更新式を AdamW 相当へ近似すること
- projection、CAME factor、Schedule-Free delta を省略すること
- persistent optimizer state と一時 workspace を同じ memory 指標として扱うこと
- GPU 実測前に backend の既定値を変更すること

ここでいう `Adam-SF` は、コード上の `AdamW-SF` / `AdamWScheduleFree` を指す。

## 記録時点の実装経路とボトルネック仮説

| optimizer | 現行の主要経路 | 主な高速化対象 | 優先度 |
| --- | --- | --- | --- |
| AdamW | CUDA では更新全体を `adamw_triton` の1 kernelへ融合 | 不要な FP32 gradient conversion、small tensor の launch | 高 |
| AdamW-SF | CUDA では second moment、`y`、`z`、parameter 更新を1 kernelへ融合 | fallback、small tensor、train/eval transition | 中 |
| CAME | row/column reduction、factor reconstruction、RMS、residual reductionを多数の PyTorch opで実行 | reductionを含む段階的 fusion | 最優先 |
| APOLLO | projection GEMM、low-rank Adam統計、full-rank scaling、limiter、parameter applyを別処理 | workspace再利用、scaling/apply fusion | 高 |

現行の CUDA 比較では AdamW が full recurrence を Triton kernelで処理する一方、CAME の Triton path は最後の parameter applyだけを融合する。この差を、アルゴリズム固有の差と分けて測定する必要がある。

## 数値仕様の reference

高速化版は、まず以下の reference と同じ state 更新順序を維持する。

### AdamW

パラメータごとに、`g` は gradient、`m` は first moment、`v` は second moment とする。

```text
m <- beta1 * m + (1 - beta1) * g
v <- beta2 * v + (1 - beta2) * g^2
d <- (m / bias_correction1) / (sqrt(v / bias_correction2) + eps)
p <- (1 - lr * weight_decay) * p - lr * d
```

CUDA fast path では、内部の accumulation は kernel 内で明示的に定義する。state の storage dtype は別作業の契約に従い、FP32 accumulation を採用するかどうかは dtype 変更の検証結果と同じ基準で扱う。

### AdamW-SF

`AdamWScheduleFree` は `exp_avg_sq` と hidden parameter `z` を更新し、group scalar `ckp1` と `lr` から現在の parameter `y` を生成する。

```text
v <- beta2 * v + (1 - beta2) * g^2
h <- g / (sqrt(v / bias_correction2) + eps)
h <- h + weight_decay * y
y <- y + ckp1 * (z - y) + gradient_scale * h
z <- z - z_scale * h
```

`train()` / `eval()` による parameter と `z` の切り替えも仕様の一部であり、step kernel の高速化で省略しない。

### CAME

行列 parameter を `G`、row/column second moment を `S_r,S_c`、first moment を `M`、residual factor を `R_r,R_c` とする。

```text
U       <- G^2 + eps_square
S_r     <- beta2 * S_r + (1 - beta2) * mean_last(U)
S_c     <- beta2 * S_c + (1 - beta2) * mean_second_last(U)
A       <- approx_sq_grad(S_r, S_c)
U       <- A * G
U       <- U / max(RMS(U) / clip_threshold, 1)
M       <- beta1 * M + (1 - beta1) * U
E       <- (U - M)^2 + eps_instability
R_r     <- beta3 * R_r + (1 - beta3) * mean_last(E)
R_c     <- beta3 * R_c + (1 - beta3) * mean_second_last(E)
D       <- approx_sq_grad(R_r, R_c) * M
p       <- decoupled_weight_decay(p) - lr * D
```

`approx_sq_grad` の broadcast、row/column 軸、RMS の reduction は変更しない。Triton の reduction は PyTorch reference と accumulation dtype、mask、境界 shapeを明示し、bitwise一致ではなく許容誤差付きの数値契約として検証する。

### APOLLO

行列 `G` を、parameter の元の軸を保った `m x n` に view する。projection `P` と rank `r` に対して、現在の実装は次の順序で動作する。

```text
L       <- project(G, P)
m1      <- beta1 * m1 + (1 - beta1) * L
m2      <- beta2 * m2 + (1 - beta2) * L^2
N       <- bias_correct(m1, m2)
scale   <- norm(N) / norm(L)       # tensor / row / column policy
U       <- G * scale
p       <- limiter_and_decay_apply(p, U)
```

`G @ P` または `P @ G` の向き、`m >= n` / `m < n` の scaling 軸、projection refresh と moment transport は維持する。projection refresh は通常 step と別の低頻度経路として測定する。

## 共通実行設計

当時の共通案は、shape・dtype・device・strideに応じたmetadataを初回に作り、parameterのdevice移動で無効化すること、数学的なpersistent stateとstep間で再利用するworkspaceを分けて計測すること、対応条件を満たすCUDA/Triton入力だけをfused pathへ送り、それ以外はPyTorch/referenceへfallbackすることだった。実測では選択backend、fallback数、shape bucket、state bytes、workspace、peak allocated/reservedを別々に記録する。

## Optimizer別の実装方針（当時の案）

| Optimizer | 記録した方針 | 数値・性能上の制約 |
| --- | --- | --- |
| AdamW | CUDA kernel前の不要なgradient conversionを避け、fallbackでは同じshape/dtype/deviceをbucket化してforeachを検討する。 | bias correction、AMSGrad、decoupled decay、state dtypeを保つ。 |
| AdamW-SF | group scalarをparameter loopの外で計算し、fallback metadata、small tensor、train/eval passを測定対象とする。 | hidden stateとtrain/eval遷移を省略・変更しない。 |
| CAME | second moment、adaptive update/RMS clip、residualを依存段階ごとに処理する。workspace保存と再計算を比較し、vector経路は別bucketにする。 | 巨大kernelへの一括融合は避け、row/column reduction、clip、broadcastをreferenceと照合する。 |
| APOLLO | projection GEMMは維持し、low-rank統計とfull-rank applyのelementwise処理をまとめる。 | limiter reduction、AutoScheduleが必要とするupdate tensor、1D/CAME fallbackを維持する。 |

AdamWのtemporary除去とCAMEのfactor-reconstruction workspace再利用を最初の候補とした。これは2026-09-14時点の優先案であり、実装・採用状況は現行コードと後続記録で確認する。

## 検証・ベンチマーク基準（当時の案）

同一初期parameterとgradient列でPyTorch/reference、auto、Tritonを比較し、vector・square/rectangular・conv-like shapeと実modelのparameter shapeを含める。warmup/compile、通常step、refresh stepを分け、CUDA event device timeとhost timeを別に測る。

FP32/BF16で1/2/20 step後のparameterと全stateを比較し、weight decay、AMSGrad、非連続fallback、limiter、projection refreshも対象にする。CUDAがない環境のCPU比較は数値・shape確認に限り、GPU性能やTriton一致の証拠にしない。

受け入れ条件は、許容誤差内のreference一致、代表shapeでの再現可能な速度計測、persistent state/workspace/allocator peakの分離、backend/fallback契約のテストである。複数shapeで数値と速度が確認できるまで既定backendを変えず、巨大kernelやreduction近似を測定前に採用しない。この項目は当時の検証計画であり、実施状況は後続記録と現行コードで確認する。
