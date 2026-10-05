# Text-LM decoder architecture benchmarks

> Historical benchmark record from the 2026-09 performance review. Conditions and short-run measurements do not establish current architecture recommendations or language-model quality.

Per-entrypoint source inventory and code-reuse proposals were omitted because they duplicated the [UX review snapshot](ux-review-2026-09-12.md) and current package READMEs. The historical benchmark conditions and measurements follow.


当時の比較は7構成で行った。現在の[`benchmarks/benchmark_text_lm_architectures.py`](../../benchmarks/benchmark_text_lm_architectures.py)と学習CLIは5つのactive architectureを対象にし、`shared-fixed`と`shared-variable`は旧checkpoint用のarchive実装として除外している。以下の両variantの数値は記録時点の歴史的測定値であり、現行CLI構成ではない。現行一覧は[`text_lm/README.md`](../../text_lm/README.md)を参照。

benchmark scriptは現在、同一条件でforward/backward/updateと短時間学習を測定する。`--seeds 42,43,44`のように複数seedを指定すると、seedごとの結果とarchitectureごとの平均・標準偏差をJSONへ保存する。CUDAでは`peak allocated/reserved`を記録し、CPUではVRAMを`null`としてprocess RSSを補助値にする。

既定モデル設定（vocab=`50257`、dim=`2048`、heads=`32`、`--num-layers=16`、GQA/MHLAのKV heads=`8`）のmeta-device parameter countは次のとおり。`mhla3-gqa`も`num-layers`を総block数として扱い、4 blockを1 cycleとして4 cycle実行する。

| architecture | 実block数 | parameters | FP32 weight概算 |
| --- | ---: | ---: | ---: |
| `naive` | 16 | 975,442,432 | 3.634 GiB |
| `shared-fixed` | 16 | 227,750,692 | 0.848 GiB |
| `shared-variable` | 16 | 227,750,692 | 0.848 GiB |
| `mhla3-gqa` | 16 | 875,829,248 | 3.263 GiB |
| `looped` | 16（物理1） | 157,460,512 | 0.587 GiB |
| `looped-hybrid` | 16（物理10、中央4回） | 648,249,664 | 2.415 GiB |
| `mhla3-gqa-looped-hybrid` | 16（物理12、中央2回） | 682,604,032 | 2.543 GiB |

GPUが見えないCPU環境での実測（fp32、batch=`1`、tokens=`128`、vocab=`50257`、dim=`256`、heads=`8`、KV heads=`2`、`num-layers=4`、threads=`1`、warmup=`1`、timed repeats=`3`）は以下のとおり。短時間学習は同じsynthetic token batchを5 optimizer steps処理した値であり、lossや品質の比較ではない。

| architecture | parameters | forward median | backward+update median | step median | short train step |
| --- | ---: | ---: | ---: | ---: | ---: |
| `naive` | 16,277,024 | 62.572 ms | 304.223 ms | 358.060 ms | 444.527 ms |
| `shared-fixed` | 14,674,700 | 75.112 ms | 327.173 ms | 402.285 ms | 429.303 ms |
| `shared-variable` | 14,674,700 | 87.187 ms | 367.153 ms | 458.508 ms | 476.988 ms |
| `mhla3-gqa` | 15,892,192 | 77.546 ms | 363.649 ms | 443.861 ms | 430.417 ms |
| `looped` | 13,718,792 | 54.945 ms | 302.214 ms | 354.497 ms | 399.839 ms |
| `looped-hybrid` | 15,424,280 | 83.512 ms | 405.353 ms | 488.865 ms | 505.951 ms |

このCPU条件では、shared系はparameter countを約10%削減したが、super-weight生成の計算が残るためnaiveよりstepが速くなったとは言えない。`looped`は物理block 1個を4回反復したためparameter countが最小で、今回の条件ではstepも最短だった。`looped-hybrid`は独立prefix/suffixを残すため、純粋な`looped`よりparameter countとstepが増える。`mhla3-gqa`は総block数を揃えたためparameter countはnaiveと同程度だった。実GPUのVRAM・速度・長系列でのスケーリングは、CUDAが利用できる環境で再測定する必要がある。

## CUDA/BF16 3 seed benchmark

CUDA/BF16、dim=`512`、総depth=`16`、batch=`2`、tokens=`128`、heads=`8`、KV heads=`2`、warmup=`10`、timed repeats=`30`、short train=`50` steps、seed=`0,1,2`で比較した。parametersはモデル全体、peakはCUDA allocatorの値である。lossはsynthetic token batchによる短時間学習の値で、実Alpacaの品質評価ではない。生JSONはローカルの`output/`に保存した記録で、公開ツリーには含めていない。

| architecture | parameters | forward median | backward+update median | step median | short train step | peak allocated | peak reserved | loss first → last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mhla3-gqa` | 74,058,368 | 56.084 ms | 97.998 ms | 155.189 ms | 162.009 ms | 1,480.156 MiB | 1,725.333 MiB | 138.358 → 76.300 |
| `looped-hybrid` | 59,826,256 | 42.370 ms | 67.033 ms | **110.550 ms** | **119.001 ms** | **1,209.116 MiB** | **1,400.000 MiB** | 381.679 → **62.309** |
| `mhla3-gqa-looped-hybrid` | 61,976,800 | 56.229 ms | 94.188 ms | 151.605 ms | 160.794 ms | 1,308.901 MiB | 1,480.000 MiB | 146.005 → 70.628 |

`mhla3-gqa-looped-hybrid`は`mhla3-gqa`に対してparameters約16.3%、peak allocated約11.6%、peak reserved約14.2%、step時間約2.3%を削減した。`looped-hybrid`ほどの速度・VRAM削減はないが、MHLA3+GQAの計算パターンを維持したまま中央cycleを共有できている。3 seedの揺れはstep時間では小さく、短時間lossはseed依存が残るため、続いて実Alpaca subsetの3 seed学習を行った。

## CUDA/BF16 実Alpaca subset 3 seed比較

実Alpacaのsplit後にtrain=`2048` examples、eval=`256` examplesへ制限し、dim=`512`、総depth=`16`、max_seq_len=`128`、batch=`4`、lr=`3e-4`固定、3 epoch×100 steps、eval=`32` batches、seed=`0,1,2`で学習した。各runのcheckpointと`metrics.jsonl`はローカルの`output/`に保存した記録で、公開ツリーには含めていない。epoch timeはtokenizeを含まず、各epochの学習と評価を含む。train/eval lossとsteps/sは3 seedの平均±標準偏差である。

| architecture | parameters | final train loss | final eval loss | final steps/s | final epoch time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mhla3-gqa` | 74,058,368 | 56.2642 ± 1.7428 | **53.5426 ± 1.6199** | 4.6667 ± 0.2403 | 21.47 ± 1.10 s |
| `looped-hybrid` | 59,826,256 | 64.3571 ± 0.0979 | 61.3053 ± 0.2278 | **7.4267 ± 0.8107** | **13.57 ± 1.55 s** |
| `mhla3-gqa-looped-hybrid` | 61,976,800 | 58.9850 ± 0.6099 | 56.1732 ± 1.1639 | 5.3133 ± 0.4359 | 18.90 ± 1.65 s |

実Alpacaでも`mhla3-gqa-looped-hybrid`は`mhla3-gqa`よりepoch timeが約12.0%短く、parameter countも約16.3%少なかった。一方、eval lossは約4.9%高く、現時点では品質最優先なら`mhla3-gqa`、計算量と品質の折衷なら`mhla3-gqa-looped-hybrid`が候補になる。`looped-hybrid`は最速だがeval lossが最も高い。3 epoch・1 subset条件のため、学習率sweep、subsetを拡大した長期学習、実測peak VRAMは別途必要である。

## mhla3-gqa-looped-hybridの測定

`mhla3-gqa-looped-hybrid`は、総depth=`16`、`prefix-cycles=1`、中央cycle `repeat=2`、`suffix-cycles=1`（物理12 block）で個別に測定した。CPU、fp32、batch=`1`、tokens=`128`、vocab=`50257`、dim=`256`、heads=`8`、KV heads=`2`、warmup=`1`、timed repeats=`3`、short train=`5` steps、threads=`1`の結果は以下のとおり。VRAMはCUDA未使用のため未計測である。

| parameters | forward median | backward+update median | step median | short train step |
| ---: | ---: | ---: | ---: | ---: |
| 21,944,480 | 201.014 ms | 663.288 ms | 873.118 ms | 913.271 ms |

この構成は中央のMHLA3+GQA cycleだけを共有するため、`mhla3-gqa`の16独立blockよりparameter countを削減しつつ、前後の独立cycleで入力・出力側の表現を分けられる。今回のCPU値は総depthが異なる測定との単純比較ではなく、GPUでのpeak VRAMと長めの学習loss推移を別途確認する必要がある。

## loopedの物理block数比較

`looped`だけについて、総展開depth=`4`を固定して物理block数を変えた。以下はCPU、fp32、batch=`1`、tokens=`128`、vocab=`50257`、dim=`256`、heads=`8`、KV heads=`2`、warmup=`1`、timed repeats=`3`、short train=`5` steps、threads=`1`のsynthetic benchmarkである。VRAMはCUDA未使用のため未計測である。

| 物理block数 | loop数 | parameters | forward median | backward+update median | step median | short train step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 13,718,792 | 86.213 ms | 469.880 ms | 556.385 ms | 560.734 ms |
| 2 | 2 | 14,571,536 | 80.686 ms | 409.490 ms | 490.176 ms | 523.978 ms |
| 4 | 1 | 16,277,024 | 78.445 ms | 389.364 ms | 451.848 ms | 499.050 ms |

同じseed・モデル設定（dim=`128`、総depth=`4`、max_seq_len=`64`、batch=`4`）で実 Alpaca データを20 optimizer steps学習した短時間比較は以下のとおり。`epoch time`はtokenize時間を含まず、学習20 stepsと評価20 batchesの時間である。

| 物理block数 | parameters | train loss | eval loss | epoch time | CUDA peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6,646,404 | 98.9903 | 75.9684 | 11.2 s | 未計測 |
| 2 | 6,859,784 | 90.6669 | 58.2810 | 28.6 s | 未計測 |
| 4 | 7,286,544 | 76.8797 | 41.8932 | 9.4 s | 未計測 |

この20 step結果では物理block数が多いほどeval lossは低かったが、短すぎるため品質の結論には使えない。`blocks=1`は最小パラメータ、`blocks=4`は総depth=`4`ではloopを使わない通常stack、`blocks=2`はその中間である。同じ初期checkpointからの長時間比較とGPU peak memoryは、このsnapshotでは未実施。

## 7構成の実Alpaca短時間比較

キャッシュ済みのAlpaca train splitを使い、同じseed・同じデータ分割・同じ小型設定（dim=`128`、総depth=`16`、max_seq_len=`64`、heads=`8`、KV heads=`2`、batch=`4`、CPU、FP32）で各構成を20 optimizer steps学習した。評価は20 batches、tokenize時間を除くepoch timeには学習と評価を含む。各構成は構造が異なるため、重みcheckpointを共有せず同じseedから個別に初期化している。

| architecture | parameters | train loss | eval loss | grad norm | steps/s | epoch time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `naive` | 9,847,168 | 40.5140 | 32.4954 | 25.469 | 0.82 | 24.3 s |
| `shared-fixed` | 7,059,724 | 54.0157 | 41.8980 | 95.332 | 0.82 | 24.5 s |
| `shared-variable` | 7,059,724 | 85.2066 | 46.4004 | 94.864 | 0.63 | 31.9 s |
| `mhla3-gqa` | 9,470,720 | 33.7059 | 30.6724 | 33.323 | 0.79 | 25.3 s |
| `looped` | 6,646,408 | 62.6256 | 49.0406 | 65.371 | 2.01 | 9.9 s |
| `looped-hybrid` | 8,566,864 | 43.5192 | 34.5057 | 30.819 | 1.30 | 15.4 s |
| `mhla3-gqa-looped-hybrid` | 8,711,296 | 36.0214 | 31.9879 | 41.706 | 1.40 | 14.3 s |

20 stepでは`mhla3-gqa`のeval lossが最小で、追加した`mhla3-gqa-looped-hybrid`はそれに近いlossを保ちながら、`mhla3-gqa`より短いepoch timeになった。`looped-hybrid`よりもstepは速かったが、これはMHLA/GQAと通常GQA blockの実装コスト差も含む。いずれも短時間・1 seedの結果であり、構成の優劣や収束性を確定するものではない。VRAMはCUDA未使用のため未計測である。

追加構成について、同じdim=`128`・総depth=`16`・batch=`4`・CPU/FP32・seed=`42`・lr=`3e-4`で100 optimizer steps（20 steps×5 epoch）まで測定した。eval lossは各epochで`29.1083`、`24.0894`、`21.6700`、`20.2760`、`19.2553`、gradient normは最終`6.7244`だった。lossは低下を続けたが、これは単一構成の短期確認であり、他構成との100 step比較とGPU peak memoryはこのsnapshotでは未実施。

総depth=`16`、dim=`128`、tokens=`64`、batch=`1`、vocab=`257`、CPU/FP32、seed=`42`でcycle配置も比較した。`prefix + repeats + suffix = 4` cycleを固定し、短時間学習は5 stepsである。

| prefix | repeats | suffix | parameters | step median | short train step | loss last |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4 | 0 | 792,448 | 105.607 ms | 112.277 ms | 50.8298 |
| 1 | 2 | 1 | 2,311,296 | 88.597 ms | 89.851 ms | 32.5535 |
| 2 | 1 | 1 | 3,070,720 | 100.552 ms | 113.042 ms | 31.0534 |

完全共有の`0+4+0`はparameter countが最小だが、今回の条件ではlossが高くstepも遅かった。`1+2+1`は前後の独立cycleを残し、`2+1+1`よりparameter countとstepを抑えながらlossも近かった。synthetic token batch・5 steps・1 seedのscreenに限られ、いずれも既定値を決める根拠ではない。GPU性能と長期収束の結果はこのsnapshotにない。
