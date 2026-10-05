# `core/`

`core/`は、複数の実験から共有するモデル関連のクラス・関数を置くパッケージです。学習スクリプトや実行時管理機能をここへ持ち込まないことを基本にします。

## 配置するもの

- `layers/`: 共有する層を責務別に分割
  - `positional.py`: Sin/Cos埋め込み、dtype-aware RMSNorm、1D/2D RoPE
  - `convolution.py`: gated convolution、pre-norm Conv FFN residual block、residual block、down/up-sampling
  - `feedforward.py`: gated FFN、補間Linear、BF16 Linear
  - `attention.py`: SDPA/GQA、MHLA、KV attention
  - `joint_attention.py`: joint primary/context KV attention
  - `adaptive_norm.py`: scale-only AdaRMS components
  - `transformer_blocks.py`, `transformers.py`: blocks and encoders/decoders
  - `depth.py`: depth-conditioned shared-weight transformer
  - `pooling.py`: attention pooling
  - `dit.py`: generic DiT block and reference SwiGLU MLP
- `mhla.py`: 共有するMHLA実装
- `kernels/`: モデル層から呼び出すPyTorch/Triton kernel
- `low_rank.py`: Linearへ注入するLoRAなどの低rank adapter
- `utils.py`: モデル構築、parameter管理、checkpointなどモデルに密接な補助処理

## 配置しないもの

- 学習時の進捗表示、GC、CUDA cache管理 → [`runtime/`](../runtime/)
- optimizerとlearning-rate制御 → [`optimizers/`](../optimizers/)
- dataset固有の処理、train loop、実験用benchmark → 各実験ディレクトリまたは[`benchmarks/`](../benchmarks/)

新しい共有モデル機能を追加するときは、入力shape・出力shape・dtype/device・contiguous条件を小さなAPI契約として明記し、reference実装とfallbackを保てるか確認してください。数値実装の基準は[`AGENTS.md`](../AGENTS.md)、image-latentモデルのshape契約は[`docs/architecture/image-latent-dit.md`](../docs/architecture/image-latent-dit.md)を参照します。


## 汎用DiT block

core.layers.DiTTransformerBlockは(B,T,D)を受け取り、pre-norm、head単位のQ/K RMSNorm、GQA、SDPA、SwiGLUを順に適用します。condition_dimを指定すると、z: (B, condition_dim)からheadごとのattention gateを作ります。gate射影をゼロ初期化し、2 * sigmoidで初期倍率を1にします。

RoPEは座標の意味をblockへ固定しないよう、qk_transform(q, k)として呼び出し元から渡します。画像の2D座標、テキストの1D位置、modality-awareなjoint座標をblock内で混同しません。boolean attention maskはSDPA規約どおりTrueが有効keyです。

このblockのQ/K/VとSwiGLU gate/value射影は、入力が同じでも別々のLinear/GEMMを使うreference実装です。tests/unit/test_core_dit_block.pyで手計算attentionとの出力・勾配一致と、将来の連結GEMM候補の前向き同値性を固定しています。連結化や他の融合は、このreferenceに対するforward/backward回帰を追加してから行います。一般化blockの学習品質・CUDA性能は未評価です。

同一入力の射影を連結Linearへまとめる候補は、[`benchmarks/benchmark_dit_projection_fusion.py`](../benchmarks/benchmark_dit_projection_fusion.py)でQ/K/VとSwiGLUのforward+backwardを比較できます。出力・入力勾配・全射影parameter勾配の誤差を照合し、中央値とCUDA peak allocated/reservedを表示します。性能計測は対象GPUが空いている状態で実行してください。例: `python3 -m benchmarks.benchmark_dit_projection_fusion --device cuda --dtype bfloat16`。

`Grid2DMHLA`は正規化済みlinear-attention内部にdropoutを適用しません。dropoutが必要な場合は、attention出力を残差へ加える呼び出し側で適用します。CIFAR-10の`WindowMHLACompositeBlock`はこの契約に従います。

## Pre-norm convolutional FFN residual blocks

`PreNormConvFFNResidual2d` and `PreNormGatedConvFFNResidual2d` share the NCHW residual computation used by `image_ae` and `image_gen`. Their wrappers retain each experiment's historical `state_dict` key layout; `zero_last=True` keeps the image_gen output-refinement block initialized as an identity. CPU shape, backward, and strict state-dict regressions live in `tests/unit/test_core_conv_ffn.py`.
