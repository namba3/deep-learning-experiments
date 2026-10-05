# リポジトリ実装・数学監査メモ

最終確認日: 2026-09-12（再調査）

この文書は監査日の実装・検証結果を残すsnapshotです。記載した対応状況、TODO、検証結果は現在の状態を保証しません。現行仕様はコードと各パッケージREADMEを、後続の監査結果はこの文書の更新日を確認してください。

本文中のソース行番号・項目番号・ファイル構成への言及も2026-09-12時点の記録として保持しています。現在の該当箇所を示す参照ではないため、現行実装の確認にはコード検索と各パッケージREADMEを使ってください。

この文書は、リポジトリ全体について、通常の実装不具合だけでなく、テンソル形状、dtype、broadcast、mask、勾配経路、attention/RoPE、Flow Matching の数式的整合性も確認した結果をまとめたものです。

現行のimage-latent DiT仕様は[`architecture/image-latent-dit.md`](../architecture/image-latent-dit.md)、性能課題は[`performance-review-2026-09-12.md`](performance-review-2026-09-12.md)を参照してください。ここでは監査時点の検証結果と未検証範囲を保持します。

## このsnapshotの要点

| 項目 | 読み方 |
|---|---|
| 実装上の指摘 | 監査時点でP1相当と対象にしたP2の数式・設定不整合は対応済みとして記録。詳細は下の優先度一覧と個別項目を参照。 |
| CPU・静的検査 | ここにある検査結果は2026-09-12時点の記録。後続のpytest、compileall、Pyright、Ruff等は[2026-10-05 validation follow-up](repository-audit-2026-10-05-validation.md)を参照。 |
| Pyrightの件数 | 上の検証表は93 errors、後段の型検査メモは122 errorsと記録している。呼び出し条件や対象範囲の差を復元できないため、両値は同一条件の比較として扱わない。後続の結果は上記follow-upを参照。 |
| CUDA/Triton・品質 | 上記follow-upの対象外。GPU上の数値一致・性能・モデル品質を確認した根拠としては使わない。 |
| 測定値 | `output/`のパスは当時の実験記録。現在のファイルの所在や再現可能性を示さない。本文またはリンク先にある集計値は、記載された条件の範囲で読む。 |

## 再調査時の検証結果

| 検証 | 結果 |
|---|---|
| `PYTHONPATH=. python3 -m pytest -q` | 233 passed, 17 warnings |
| `python3 -m compileall -q core image_gen image_ae optimizers tests verify` | 成功 |
| `PYTHONPATH=. pytest --cov=core --cov=optimizers tests/unit -q` | 218 passed, coverage 64% |
| `ruff check .` | 成功（指摘0件） |
| Optimizer runtime、収束、norm-growth測定（別CUDA環境） | CUDA/BF16のfallback・refresh benchmarkと小型MLP sweepを実行。条件、数値、limitationは[optimizer performance review](optimizer-review-2026-09-12.md)に集約 |
| ImageAE synthetic optimizer probes（別CUDA環境） | BF16の固定入力probe。batch 4/8、limiter有効/無効の詳細条件・数値・制約は[optimizer performance review](optimizer-review-2026-09-12.md)を参照 |
| CIFAR-10 ImageAE training（別CUDA環境） | 全train split、FP32、batch 8の1/5 epoch比較。validation、checkpoint再読込、VRAM記録の範囲は[optimizer performance review](optimizer-review-2026-09-12.md)と[APOLLO experiment results](../apollo-experiment-results.md#cifar10-five-epoch)を参照 |
| `python3 -m verify.qwen_vae --vae-model <local Qwen snapshot> --vae-dtype fp32` | CPUで実Qwen VAEのencode、全default bucket、encode→decodeを成功 |
| `python3 -m verify.core_kernels --dtype fp32` | 現在の環境ではCUDA unavailableのため`status=skipped`、core kernelの実GPU比較は未実行 |
| `python3 -m pyright` | 93 errors, 0 warnings |
| `torch.cuda.is_available()` | `False`（torch 2.13.0+cu130、device_count=0） |

テストは通過しているが、core kernelのCUDA/Triton経路、optimizer以外のBF16実機精度、高解像度時の実VRAM挙動は未検証である。pytestではpynvml/NVML、Python 3.14のtorch.jit、backward hookに関する警告も発生している。

## 結論

再調査時点でP1相当の不具合と対象P2の数式・設定不整合は対応済みでした。RoPE・shape/layout・mask・checkpoint復元・optimizer・ImageAE/VAE境界・dataset IOの個別経緯は以下に記録しています。共有attention、未使用計算、CUDA/Triton実機、実学習の収束・step時間・VRAMには要確認範囲が残っていました。

optimizer-only fallback・projection-refresh計測と小型MLPの収束・norm-growth sweepは、いずれも実モデルの品質や性能を直接示すものではありません。測定条件、数値、制約は[optimizer performance review](optimizer-review-2026-09-12.md)にまとめています。後続のMini-ImageNet比較とAPOLLO norm-growth limiterの採否は[実験結果](../mini-imagenet-gqa-results.md#apollo-norm-growth-limiter-ablation)を参照してください。

## 優先度付き一覧

| 優先度 | 箇所 | 内容 |
|---|---|---|
| P2（CPU契約対応済み） | `core/mhla.py`, `core/layers/attention.py` | Triton availability/bridgeでnon-contiguousなkernel入力を拒否し、`Grid2DMHLA(backend="auto")`はvectorizedへfallbackする。CUDA実機の数値検証は未実施 |
| P2 | `core`, `image_gen`, `optimizers` | Triton経路の数値・勾配・境界shape・fallback・VRAM検証が不足 |
| P3（対応済み） | `image_gen/train.py:885-928` | `RecordsDataset`のbucket走査と`__getitem__`をcontext manager化し、画像ファイルのクローズをCPUテストで固定 |
| P3（対応済み） | `core/layers/attention.py:Grid2DMHLA` | 未使用のdropout引数を削除。CIFARの外側残差blockがattention出力dropoutを担当 |
| P3 | `image_ae/train.py:1317-1322,2172-2177` | wavelet loss無効時のmetric計算コスト（詳細は項目16） |
| P3（CPU契約対応済み） | `core/kernels/rms_norm.py` | Triton RMSNormのweightについてshape・device・dtype・contiguous契約をkernel起動前に検証。CUDA実機のkernel挙動は未検証 |
| P3（対応済み） | `core/kernels` | 未使用importはなく、RMSNormをruff formatで整形。FFN/RoPE/RMSNormのTriton `tl.constexpr`注釈にはpyright抑制を局所指定し、kernel単位のpyrightが通過 |

## 1. core側の2D RoPE（対応済み、CUDA/Tritonは未検証）

### 対応内容

修正前は`core/layers.py`で位相を次の順序で作り、half-split回転を適用していました。

```text
[y0, y0, y1, y1, x0, x0, x1, x1]
```

一方、回転は次のhalf-split方式です。

```python
x1, x2 = x.chunk(2, dim=-1)
return torch.cat((-x2, x1), dim=-1)
```

この2つは対応せず、Y側とX側の成分が混ざっていました。現在は隣接する偶奇成分を個別に回転するpair-wise方式へ変更し、`RotaryEmbedding2D`と`Grid2DMHLA`で共通の実装を使用しています。

修正前の`Grid2DMHLA` の `_rope()` を偶奇ペア回転と比較したところ、ランダム入力で最大絶対差は約 `3.34` でした。丸め誤差ではありません。

なお、可変gridでは初期gridのcos/sinを補間すると位相が整数座標のRoPE式からずれるため、`RotaryEmbedding2D.forward()`は要求された`(H, W)`の整数座標から位相を再生成する。初期gridの`cos`/`sin` bufferはcheckpoint互換性のため保持し、通常の固定gridでは従来どおりbufferを使用する。

### 影響範囲

- `core.layers.RotaryEmbedding2D`
- `Grid2DMHLA`
- `image_ae` のWindow Attention
- core側の画像用MHLA

### 残る確認事項

CPUの2×3固定gridと3×5可変gridでpair-wise reference式とのforward/backwardを比較し、`head_dim % 4`、正のgrid、token数不一致を入口で拒否します。CUDA/Triton実機の数値比較は未実施です。

## 2. samplerのcheckpoint構成復元（対応済み、実checkpoint未検証）

`train.py` のcheckpoint metadataには、次の構成情報が保存されています。

- `model_dim`, `depth`, `heads`, `kv_heads`
- `context_depth`, `context_heads`, `context_kv_heads`
- `attention_gate`, `attention_pattern`
- `mhla_*`
- text adapterのtransformer dimensions、heads、KV heads

修正前の`generate_samples.py`のモデル構築では一部しか使用していませんでした。欠落していた項目は次のとおりです。

- `kv_heads`
- `context_kv_heads`
- `attention_gate`
- `attention_pattern`
- `mhla_latent_blocks`
- `mhla_image_blocks`
- `mhla_text_blocks`
- `mhla_backend`
- `mhla_recompute_output`
- text adapterのKV head数

さらに、現行の`TextConditioningAdapter.__init__()`は`transformer_heads`と`transformer_kv_heads`を別引数で受け取る。修正前のsampler側はKV headsを省略したまま`transformer_ff_mult`を位置引数で渡していました。

現在は`build_checkpoint_models()`を追加し、学習時metadataのDiT/Adapter設定を名前付き引数で復元します。KV heads、attention pattern/gate、MHLA設定を含む小規模構成でconstructorとstrict `load_state_dict()`のCPU smoke testを追加済みです。実checkpoint、実VAE、CUDAでのロード・生成は未検証です。

### 今後の確認

実checkpointを使ったロードテストを追加し、VAEのlatent channel・reference解像度・text encoder hidden sizeがmetadataと一致することを確認します。

## 3. ImageContextEmbedderの最小解像度（対応済み、CUDA未検証）

`ImageContextEmbedder` は `kernel_size=4, stride=2, padding=1` の畳み込みを3回適用します。通常の8倍VAEを仮定すると、224pxのbucketはlatentが約28x28になります。

```text
224px → 28x28 latent
      → LatentDownsample後 14x14
      → ImageContextEmbedder後 1x1
```

入力空間が8x8以上なら、3回のdownsample後に1x1以上が残る。修正前はlatent空間が4x4以下で畳み込み自体が失敗し、`context_dim=16`または`32`ではGroupNormのgroup数がchannel数と同じになって1x1・batch=1で次のエラーになった。

```text
Expected more than 1 value per channel when training
```

### 対応内容

`ImageContextEmbedder.forward()`でNCHWと各空間軸8以上を検証し、context用GroupNormは1グループあたり最低2 channelとなるようにした。`context_dim=1`では1×1を扱える`SpatialRMSNorm`へ切り替える。`context_dim=1/16/32`、batch=1、8×8入力のCPUテストを追加済みである。

## 4. Muonの未接続parameter更新（対応済み）

対象: `optimizers/muon.py:33-57`

修正前は `parameter.grad is None` の場合にzero gradientを作り、更新を継続していた。

```python
grad = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.float()
```

このため、momentumまたはweight decayが残っているparameterは、現在のstepで勾配が無くても変化する。実測では、1 step勾配を与えた後に `grad=None` でstepしたところ、parameterの最大変化量は `0.0054266` だった。

現在は通常のPyTorch optimizerと同じく、勾配がないparameterをskipする。

```python
if parameter.grad is None:
    continue
```

1 step後に`grad=None`とした場合のparameter、momentum不変をCPUテストで固定した。CUDA/Triton backendは未検証である。

## 5. KVSelfAttentionのスケール係数（対応済み）

対象: `core/layers/attention.py`の`KVSelfAttention`

修正前の`KVSelfAttention` はKをqueryとkeyの両方に使い、次のように計算していました。

```python
k = k / math.sqrt(head_dim)
scores = torch.einsum("nlhd,nshd->nhls", k, k)
```

この場合、内積全体は `head_dim` で割られます。一方、scaled dot-product attentionの標準的な温度は `sqrt(head_dim)` であり、期待される式は次のいずれかです。

```python
scores = torch.einsum("nlhd,nshd->nhls", k, k) / math.sqrt(head_dim)
# または query側だけを sqrt(head_dim) でscaleする
```

現在は内積全体を`sqrt(head_dim)`で一度だけ割る標準式へ変更した。`mnist/train.py` の`AttentionPoolingWithKVSelfAttention`がこの経路を使用する。Identity projectionを使ったattention weight比較テストを追加済みである。

## 6. prediction type（対応済み）

現在の学習式は次のRectified Flowです。

```text
x_t = (1 - t) x_0 + t x_1
v   = x_1 - x_0
x̂_0 = x_t - t v̂
```

この数式自体は整合しています。samplerの `t=1` から `t=0` へ向かうEuler更新も、現在の定義と一致しています。

`--prediction-type`はtarget生成関数へ接続した。現在の直線補間では、Rectified Flowとconditional Flow Matchingの速度targetがともに`noise-clean`になるため、数値結果は同じである。両方を明示的に受け付け、未知の値は`ValueError`とするCPUテストを追加した。別のpath/objectiveを導入する場合は、この関数へ独立した数式を追加する。

## 7. Reconstruction loss cap

reconstruction lossの寄与率capは、正のlossを仮定すれば次の制約を満たす構造です。

```text
reconstruction / (diffusion + reconstruction) <= cap
```

cap係数の計算でlossをdetachしているため、cap係数自体に勾配を流さない設計です。Richの複数行表示、TensorBoard、epochログでは「cap前」と「cap後」のweighted loss・寄与率を分けて出力し、CPU固定値テストでpost-capの上限を確認しています。

## 8. AutoScheduleの設定接続（対応済み、実運用未検証）

修正前は次の設定値がCLI・optimizerから`AutoScheduleMixin`へ渡され、検証・保存もされる一方、LR更新式では実質的に参照されていませんでした。

- `target_update_ratio`
- `trust_alpha`
- `max_increase`
- `max_decrease`

固定gradientで値を変更しても、parameter updateとLR multiplierが同一になることを確認しています。これは「optimizerへ渡されていない」のではなく、「設定値が死んだまま保持されている」問題です。一方、`min_factor`、`max_factor`、`confidence_floor`、`stability_gain`、`limiter_gain`などは更新式・cap計算で使用されています。

### 対応内容

現在は、1 stepごとのEMA比を基準に次の制御を行います。

```text
error      = gain * (log(target_update_ratio) - log(ema_ratio))
raw_factor = exp(trust_alpha * error)
factor     = 1 + controller_rate * (raw_factor - 1)
factor     = clamp(factor, max_decrease, max_increase)
multiplier = clamp(previous_multiplier * factor, min_factor, max_factor)
```

`target_update_ratio`は目標比、`trust_alpha`はlog誤差への反応強度、`max_increase`/`max_decrease`は1 stepの増減幅として更新式に接続した。既存の`gain`と`controller_rate`も同じ制御経路で使用している。

固定統計を使い、目標比を`0.01`から`0.2`へ変更したときにmultiplierがそれぞれ下限`0.5`・上限`2.0`へ向かうこと、増減幅のclampが効くことをCPUテストで固定した。実際の学習でのEMA収束、step時間、lossへの影響は未検証である。

## 9. head-wise gateの初期倍率（image_gen対応済み、互換wrapperは例外）

実装によって初期倍率が異なります。

```python
sigmoid(0)       = 0.5
2.0 * sigmoid(0) = 1.0
```

`core.layers.GroupedQueryAttention` や `image_ae` のWindow Attentionは `2.0 * sigmoid` を使う一方、`image_gen/train.py` のText Transformer、MMDiT、Context Transformer、MHLA系は `sigmoid` のみでした。なお、旧API互換の`GatedMultiheadAttention`は従来の初期倍率を維持します。

`image_gen`側は`HEAD_GATE_SCALE = 2.0`へ統一し、zero initialization時のhead倍率を1.0にした。DiT本体のAda residual gateはzero initializationなので、学習開始時の残差経路は維持される。互換wrapperの0.5倍率は既存checkpoint/API挙動を保つため、別仕様として残している。

zero gateに対する倍率が1.0になることをCPUテストで固定した。実学習での初期loss・収束速度への影響と、既存checkpointとの比較は未検証である。

## 10. latent channel数1のdecorrelation loss（対応済み）

対象: `image_ae/train.py:1219-1227`

`compute_channel_decorrelation_loss()` は相関行列の非対角要素の平均を返します。latent channel数が1の場合、非対角要素が空tensorになるため、修正前は`mean()`がNaNを返しました。

```text
C=1: nan
C=2: finite
C=4: finite
```

現在は`C < 2`で勾配グラフを維持したゼロlossを返します。channel=1のfinite性とゼロ勾配をCPUテストで固定しました。

## 11. GQAの全key maskとattention weights（対応済み）

対象: `core/layers/attention.py`の`GroupedQueryAttention`

通常の出力経路はSDPAが全key maskを安全に扱いますが、修正前の`need_weights=True`分岐では手動softmaxにより`softmax([-inf, ...])`となり、出力とweightsがNaNになっていました。

現在は全mask行を安全なscoreへ置換し、softmax後にweightsをゼロ化します。全key maskで出力・weights・backwardがfiniteになるCPUテストを追加済みです。Triton実装はCUDA環境で別途確認します。

## 13. Grid2DMHLAのdropout引数（対応済み）

`Grid2DMHLA`の未使用`dropout`引数と保存値を削除しました。CIFARの`WindowMHLACompositeBlock`は、MHLA出力を残差へ加える位置で`nn.Dropout(drop_out)`を適用しており、正規化済みの線形attention内部へdropoutを導入せずに既存の有効な正則化を保ちます。

## 14. Tritonのlayout契約 — P2/P3

`core.mhla.triton_available()` はCUDA・dtypeに加えてQ/K/Vのcontiguous条件を確認する。`Grid2DMHLA` のQ/K/Vはprojection後のtranspose/permute由来で、CPU上でkernel入力を追跡すると、RoPE後のQ/Kはcontiguousになる一方、Vはnon-contiguousのままです。公開kernel APIに任意のview/transpose入力を渡した場合はQ/Kもnon-contiguousになり得ます。`core.mhla`から呼ばれるTriton kernelはflat pointer indexingを前提としているため、strideを無視して誤読する可能性があります。

一方、`core/kernels/rope.py` は明示的にcontiguous化しているため、同じ契約になっていません。現在のMHLA側はcontiguousでない入力をavailability判定で拒否し、`backend="auto"`では安全にvectorizedへfallbackする。明示的な`core.mhla.attention()` bridgeも、未対応入力をkernelへ渡さず理由付きで`RuntimeError`にする。CUDA実機でnon-contiguous入力を使い、PyTorch/vectorizedとの差をforward/backwardで確認するまで、Grid2DMHLAのTritonを既定経路にしない方が安全です。

また、`core/kernels/rms_norm.py` のTriton pathはweight shapeだけを検証し、weightのdevice、dtype、layoutを検証していません。通常のmodule呼び出しでは一致していることが多いものの、共有kernel APIとしては入力契約を明示して、誤った入力をfallbackまたはValueErrorにすべきです。

## 15. GroupNormのchannel幅契約（対応済み）

`core.layers._default_group_norm()` と`image_ae.train.group_norm()` は、修正前は`min(32, channels)`をそのままgroup数にしていました。channel数48では32で割り切れないため、GroupNormの構築自体が失敗します。`ImageAE(bottleneck_channels=48, encoder_type="residual_conv_ffn")`で再現しました。window transformer経路はbottleneckを32の倍数に制約していますが、Conv系経路には同じ制約がありません。

現在は32以下でchannel数を割り切れる最大のgroup数を選ぶ。48 channelなら24 groupになる。channel数0以下は明示的に`ValueError`とし、core・image_aeの境界をCPUテストで固定した。encoder/decoder全バリエーションの構築網羅と実運用channel幅は未検証である。

## 16. Wavelet lossのlevel上限（対応済み、metricコストは残存）

修正前の`compute_wavelet_loss()`は`wavelet_levels > 0`しか検証せず、入力の高さ・幅が2未満になるまでstride=2の畳み込みを繰り返していました。現在は入力shapeを確認し、`min(H, W)`から計算した最大levelを超える場合を関数入口で`ValueError`にする。4x4入力ではlevel 2まで、level 3は拒否するCPUテストを追加した。

なお、wavelet lossを無効にしてもログ用`wavelet_metric`を計算する経路は残っている。これはlevel不整合とは別の性能課題であり、metricを無効化できる設定と、loss・metricの計算コスト比較が必要である。

## 17. image bucketとVAE strideの契約（実Qwen VAEのCPU probe済み、GPU未検証）

修正前の`image_gen`は`--bucket-step`の正数性や、生成したpixel bucketが実VAEで一定のlatent strideになるかを検証していませんでした。

現在は`--image-size`、`--bucket-step`、`--patch-size`の正数性をmain入口で検証し、dataset生成後に全bucketを実VAEへprobeする`validate_bucket_shapes_with_vae()`を追加した。各bucketについてVAE出力が`(B,C,H,W)`であること、latent gridが空でないこと、pixel/latent各軸が整数倍であること、全bucketのstrideが一致することを確認する。矩形bucketとstride不一致をダミーVAEのCPUテストで固定した。

小型の実diffusers `AutoencoderKL`をCPUで構築し、32x32・64x40の矩形bucketをprobeしてstride `(2, 2)`を確認するunit testを追加した。

ローカルcacheに存在した実Qwen `AutoencoderKLQwenImage`をCPUでロードし、`encode_images()`経路をprobeした。`8x8 -> (1,16,1,1)`が最小実行可能サイズで、`1x1`と`4x4`は畳み込みkernel境界で失敗した。`16x16 -> (1,16,2,2)`、`32x32 -> (1,16,4,4)`、`32x40 -> (1,16,4,5)`で空間stride `(8, 8)`、latent channel数16を確認した。`12x16`、`17x17`、`31x33`のような非整列入力では軸ごとの実効strideが整数8にならず、bucket検証で拒否すべきことも確認した。`image_size=256, bucket_step=32`のdefault bucket `(320,224),(288,224),(256,256),(224,288),(224,320)`は全てstride `(8,8)`で、CPU probe時間は約6.1秒だった。

実Qwen VAEのconfigでは`scaling_factor=None`だったため、scale未指定時に`1.0`へ解決する`resolve_vae_latent_scale()`を追加し、train、generate、latent-statisticsの各経路で共有する。CPU上のencode→decode往復では、`16x16 -> latent (1,16,2,2) -> decoded (1,3,16,16)`、`32x40 -> latent (1,16,4,5) -> decoded (1,3,32,40)`を確認し、出力はfiniteだった。往復probe時間はそれぞれ約0.09秒、0.17秒だった。GPU上のprobe時間とBF16精度は未検証である。

## 18. ImageAEのlatent channel契約（対応済み）

修正前は`ImageAE.__init__()`が`latent_channels`の正数性を検証せず、CLI側の`args.latent_channels = args.latent_channels or 16`も0をデフォルト値へ置換していました。そのため、直接constructorへ0を渡した場合とCLIの場合で挙動が一致しませんでした。

現在はconstructorとCLIの両方で`latent_channels >= 1`を明示的に検証し、`latent_channels=1`は構築可能なまま、decorrelation loss側の退化ケースと分離した。0および負数のconstructor拒否をCPUテストで固定した。

## 19. テンソルshape・layoutの監査軸

CPU確認では小型DiTの`(B,H,W,text_tokens)=(1,32,24,1),(2,32,24,3),(1,16,16,2)`（最後は全text mask）で入力と同じ出力shape・finite性を確認しました。MMDiT/JointMHLAの矩形stream、batch=2、部分/全maskのforward/backward、token数・`H*W`・mask不一致の拒否、ImageAE/DiTの境界shape、window padding/crop、可変3×5 RoPEも確認しました。共通のshape/layoutレビュー基準は[`AGENTS.md`](../../AGENTS.md)を参照してください。当時の追加候補であり、現行の未完了項目ではありません。

## 20. Text LMのAPOLLO optimizer分岐（対応済み）

修正前の`text_lm/train.py`では、CLIの`--optimizer` choicesに`APOLLO`と`APOLLO-CAME`が追加されていました。しかし分岐内では、同ファイルで定義・代入されていない名前を使用していました。

```python
optimizer = APOLLO(
    optimizer_param_groups, lr=lr, rank=8, weight_decay=WEIGHT_DECAY,
)
```

そのため、これらのoptimizerを選択すると、学習開始前に`NameError`になっていました。現在は`build_optimizer()`へ共通化し、`model.parameters()`、`args.lr`、`args.weight_decay`を明示的に渡します。APOLLOとAPOLLO-CAMEの構築・1 step smoke testを追加済みです。

CLI全体のdataset/tokenizerを含む長時間実行と、複数parameter groupの実運用は未検証です。

## 数値・テンソル面で確認できたこと

MHLA naive/vectorizedとdispatcher/referenceのCPU forward/backwardはGQA、mask、irregular block、矩形shape、FP32/BF16で比較しました（BF16 RMSNormはFP32蓄積referenceに対し`atol=rtol=1e-2`）。Optimizerの更新順序は小tensorの手計算と照合し、APOLLO射影shape/rank clamp/3D flatten、RotAPOLLO直交性、VAE latent scaleと逆scale、対角Gaussian KLも確認しました。全optimizer variantの更新式監査は未完了です。

## 追加監査：optimizerのテンソル計算と数式

### APOLLO-CAME-AutoScheduleのconfidence集計（要素数加重に対応済み）

`APOLLO-CAME-AutoSchedule` は、低ランクの `exp_avg` とCAME residualのrow/column factorからgroup-level confidenceを推定しています。

```python
moment_energy = exp_avg.square().mean()
residual_energy = 0.5 * (residual_row.mean() + residual_col.mean())
```

この値は、元のresidual matrixの厳密なエネルギーではなく、factored stateからの近似値です。修正前はテンソルごとのmeanをgroup内で単純加算していたため、group内ではparameter要素数ではなくtensor数を基準にした集計になっていました。現在は各`exp_avg`の要素数を重みとして、moment energyと推定residual energyを加算します。

現在の集計は、low-rank state全体を要素単位の統計として扱う次の形です。

```text
moment_energy += mean(exp_avg²) × numel(exp_avg)
noise_energy  += mean(residual) × numel(exp_avg)
confidence = 1 / (1 + sqrt(noise_energy / moment_energy))
```

`exp_avg`とrow/column residual factorの形状はfull parameterと異なるため、これはfull-rank parameter energyではなく、low-rank stateに対する要素数加重である。異なるstateサイズを混在させたCPUテストで、重み付け後のmoment/noise energyを固定した。Triton経路でのstate一致と実学習時のconfidence収束は未検証である。

### AutoScheduleのsignalの意味

`update_norm_sq` はLR適用前に集計されるため、指定した `pre_lr_update` の定義には概ね一致します。ただし、集計時点でAPOLLO scaling、norm-growth limiter、`scale_front` はすでに適用済みです。

したがって、実際のsignalは次の意味です。

```text
pre-LR scaled and limited update / parameter norm
```

生の低ランク更新を測定したい場合は、scaling・limiter適用前に別の統計を取る必要があります。現在の実装は「実際にparameterへ適用する更新量」を制御する信号としては妥当です。

### APOLLO-CAMEのPyTorch/Triton一致

CPU/PyTorch経路では、ローカルCAMEと参照実装をFP32/BF16・5stepで比較し、最大差0を確認しています。一方、APOLLO-CAMEのTriton経路は実GPUで未検証です。

最低限、次のstateを複数stepで比較する必要があります。

- adaptive update
- `exp_avg`
- row/column second moment
- residual row/column state
- 最終channel scaling

### Triton availabilityの戻り値

`optimizers/apollo_triton.py` の `is_available()` は、条件式の短絡評価によって、理論上 `bool` ではなく空でない `tensors` tupleを返す可能性があります。条件判定では動作しますが、戻り値型の契約とは一致しません。

```python
return bool(
    triton is not None
    and tensors
    and all(...)
)
```

のように明示的にbool化するのが安全です。

### 静的型検査

別のPyright実行では122件のエラーが記録されています。Triton DSLのkernel添字・`tl.constexpr`に起因するものが多数ですが、`apollo.py`の`stats is None`、`is_available()`の戻り値、テストの動的kwargsなど、実装側で整理可能なエラーも含まれます。Ruffは指摘0件です。

`compileall` とpytestは通過しているため、現時点では型検査エラーを実行時不具合と断定していません。ただし、Tritonコードを除外・専用stub化した上で、Python側の型エラーをゼロにするのが望ましいです。

## 検証上の制約

このsnapshotの環境ではCUDAが利用できず、Triton forward/backward、CUDA SDPAとの比較、実機BF16精度、高解像度時のVRAMは未検証でした。pytest/coverage、Pyright、Ruffの当時の結果は上の検証表を参照してください。後続の静的・CPU検証は[2026-10-05 validation follow-up](repository-audit-2026-10-05-validation.md)に記録しています。

## Triton経路の重点監査項目

共通の数値・layout・mask・dtype・fallback・memory確認基準は[`AGENTS.md`](../../AGENTS.md)を参照してください。この監査固有の未検証点は、`core/kernels`のRoPE/RMSNorm/GatedFFN backwardがPyTorchへ戻ることと、`core/mhla.py`・optimizer経路をproductionの`image_gen/train.py`と同じmask/layoutで統合確認していないことです。

## 記録時点の追加確認候補（2026-09-12）

実VAE境界、shape/token/mask、CUDA/Triton勾配、wavelet metricコストは当時の候補です。現在の作業状況を示すものではありません。
