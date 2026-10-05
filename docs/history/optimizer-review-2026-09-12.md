# Optimizer / Scheduler / AutoSchedule review

> Historical snapshot from 2026-09-12. Measurements and implementation status describe that review date; current contracts are in [optimizer documentation](../optimizers.md) and code.

## 調査対象

以下を静的に確認した。

- `optimizers/adamw.py`
- `optimizers/apollo.py`
- `optimizers/came.py`
- `optimizers/schedulefree.py`
- `optimizers/auto_schedule.py`
- `optimizers/lr_scheduler.py`
- `optimizers/muon.py`
- `core/utils.py` のoptimizer補助処理

## AutoScheduleの状態粒度

統計はparameter tensor単位で計算し、controller状態はparameter groupのscalar metadataとして保持する。controllerはparameter-sized tensorを追加しないが、基礎optimizerのmomentやprojection stateのメモリ量は別に評価する。現行の更新契約は[Optimizer documentation](../optimizers.md)、fusion設計のsnapshotは[Optimizer performance design](optimizer-performance-design-2026-09-14.md)を参照。

## 優先度の高い問題・改善候補

Checkpoint/resumeのentrypoint別実装、sampler位置、当時の未復元状態は[UX review snapshot](ux-review-2026-09-12.md)に集約している。この性能記録では重複するresumeの優先度・実施案を省き、optimizerとモデル計測を扱う。

### APOLLOの一時FP32変換とprojection refresh

当時はBF16勾配のFP32変換がstep中の一時領域を増やす可能性と、`update_proj_gap`によるrefresh stepの時間変動を計測仮説として記録した。これは実運用でのボトルネックを確定したものではない。後続のoptimizer-only、ImageAE、CIFAR-10測定は異なるworkloadなので、結果を混ぜずに読む。

この記録で区別する指標はpersistent state bytes、step中のpeak allocated/reserved、通常stepとrefresh stepの時間である。現行のoptimizer契約は[`optimizer documentation`](../optimizers.md)、数値・backendを保った高速化設計snapshotは[`optimizer performance design`](optimizer-performance-design-2026-09-14.md)を参照。

#### optimizer-only CUDA/BF16 benchmark（2026-09-12）

`output/a.json`の設定は、CUDA/BF16、warmup=`10`、steps=`50`で、`4x3`、`16x16`、`32x32`、`64x32x3x3`、`4096`を測定した。全caseが`passed`だった。以下は`cuda_seconds_per_step`の代表値（秒）である。

| shape | CAME | APOLLO rank=1 | APOLLO rank=8 | APOLLO-CAME rank=1 | APOLLO-CAME rank=8 | APOLLOMini |
|---|---:|---:|---:|---:|---:|---:|
| `4x3` | 0.000832 | 0.000529 | 0.000989 | 0.001100 | 0.000764 | 0.000661 |
| `16x16` | 0.000891 | 0.000437 | 0.001025 | 0.001998 | 0.000967 | 0.000550 |
| `32x32` | 0.000667 | 0.000504 | 0.000813 | 0.000738 | 0.001284 | 0.000352 |
| `64x32x3x3` | 0.000565 | 0.000410 | 0.000403 | 0.001039 | 0.001119 | 0.000331 |
| `4096` | 0.000470 | 0.000462 | 0.000313 | 0.000411 | 0.000390 | 0.000603 |

小さい行列ではstate-size policyによりAPOLLO rank=8とAPOLLO-CAME rank=8がCAMEへfallbackしている。`64x32x3x3`ではrank=8のAPOLLOがCAMEより速いが、`32x32`では小行列の処理 overheadによりCAMEの方が速い。APOLLO-CAMEは追加のCAME統計により速度面の優位性がなく、主に収束・品質比較用である。これはoptimizer-onlyの小規模測定であり、実モデルのstep時間や収束を直接示すものではない。

`output/apollo-policy.json` と `output/came-policy.json` は要素数修正後のCUDA/BF16測定で、どちらも全caseが`passed`だった。`small_matrix=apollo`では、`4x3`のAPOLLO rank=8が136 bytes、`16x16`が1540 bytesとなり、直接のCAME（それぞれ106 bytes、1282 bytes）よりstateが大きくなる。小型行列のstate削減を優先する場合は、`small_matrix=auto`でstate見積もりに基づきCAMEへfallbackする方針が妥当である。

`output/came-policy.json`も修正後に再実行し、matrix fallbackは全shapeでCAME本体と同じstate bytes/elementsおよびparameter normになった。全caseが`passed`で、state見積もりと実state bytesも一致している。1D fallbackは従来のCAME-like経路のため、CAME本体とはstate scalarと更新結果が異なる。

`output/auto-policy-refresh.json`でも`--matrix-fallback auto --update-proj-gap 10 --steps 50`を測定した。CAMEへfallbackした小行列と1D parameterではrefreshは発生せず、APOLLO行列経路では各case 5回、合計80回発生した。state bytesは変化せず、APOLLO系のstate見積もりと実state bytesは全caseで一致した。refresh stepのGPU時間はshape・rankごとの揺れが大きく、`64x32x3x3`のAPOLLO rank=1では通常stepより遅くなった一方、rank=8では明確な増加を確認できなかった。50 stepのsynthetic測定だけでは`update_proj_gap`の本番値を固定せず、続く収束probeと実モデル測定で品質も比較した。

#### optimizer convergence probe（2026-09-12）

`output/optimizer-convergence.json`では、同じ初期重み・入力・教師出力を使った小型MLPをCUDA/BF16、warmup=`5`、steps=`200`、batch=`64`、rank=`8`、learning rate=`1e-3`で比較した。

| optimizer | initial loss | final loss | CUDA optimizer step | persistent state |
|---|---:|---:|---:|---:|
| CAME | 34.829 | 0.0172 | 0.00293 s | 15,372 B |
| APOLLO | 34.829 | 33.1040 | 0.00232 s | 11,416 B |
| APOLLO-CAME | 34.829 | 0.0645 | 0.00397 s | 14,166 B |
| APOLLOMini | 34.829 | 32.5859 | 0.00274 s | 2,904 B |

APOLLOとAPOLLOMiniはstate量とoptimizer step時間に利点がある一方、今回の学習率ではlossをほとんど下げなかった。APOLLO-CAMEはCAMEに近い収束を示すが、step時間とstate量は増える。したがって、APOLLO系の採用判断はstate/速度benchmarkだけで完了せず、少なくともこの形式の収束probeと実モデルのloss比較を通す。

追加sweepではnorm-growth limiterの影響が大きかった。APOLLO rank=8・scale=1.0の最終lossは、limiter有効時にlr=`1e-3`で`33.104`、lr=`3e-3`で`24.645`だったのに対し、limiter無効時はそれぞれ`19.219`、`1.663`まで低下した。`output/optimizer-convergence-lr01.json`のlr=`0.01`でもAPOLLOは`4.033`まで改善したが、CAMEの`0.0019`には届かなかった。一方、APOLLO-CAME rank=1ではlimiter無効時の一部条件でlossが数千まで発散したため、limiterを一律に無効化する変更は採用しない。後続のImageAE probeで実モデル候補を比較した。

#### optimizer convergence sweep（2026-09-12）

`output/optimizer-convergence-sweep.json`と`output/optimizer-convergence-sweep-no-limiter.json`で、同じ小型MLPに対してrank=`1,4,8`、learning rate=`3e-4,1e-3,3e-3`、scale=`0.5,1.0`を比較した。steps=`200`、batch=`64`、CUDA/BF16で、初期重み・入力・教師出力は全caseで共有している。

比較基準として`output/optimizer-convergence-came-lr-sweep.json`も同じ条件で実行した。CAMEのfinal lossはlr=`3e-4`で`0.0844`、lr=`1e-3`で`0.0172`、lr=`3e-3`で`0.00448`だった。

| optimizer | limiter | 最良final loss | 条件 |
|---|---|---:|---|
| CAME | — | 0.00448 | lr=3e-3 |
| APOLLO | on | 21.514 | rank=1, lr=3e-3, scale=1.0 |
| APOLLO | off | 1.130 | rank=1, lr=3e-3, scale=1.0 |
| APOLLO-CAME | on | 0.0215 | rank=8, lr=3e-3, scale=0.5 |
| APOLLO-CAME | off | 0.00272 | rank=8, lr=3e-3, scale=1.0 |
| APOLLOMini | on | 23.321 | lr=3e-3, scale=1.0 |
| APOLLOMini | off | 1.944 | lr=3e-3, scale=1.0 |

このprobeではnorm-growth limiterがAPOLLO/APOLLOMiniの更新を強く抑制していた。limiterを無効にすると、同じ`rank=1, lr=3e-3, scale=1.0`でAPOLLOのfinal lossは`21.514`から`1.130`へ、APOLLOMiniは`23.321`から`1.944`へ低下した。一方、APOLLO-CAMEはlimiter無効時にrank=1・高LRで発散するcaseがあり、rank=8では安定して`0.00272`まで収束した。したがって、probeだけからlimiter無効を全モデルの既定値にはせず、APOLLOはlimiter無効を候補、APOLLO-CAMEはrankとLRの制約付き候補として実モデルで検証する。

limiter scalarを除くoptimizer stateは同じであり、APOLLO rank=1は`2,904` bytesから`2,880` bytes、APOLLOMiniは`2,904` bytesから`2,880` bytesになった。limiter無効・最良条件のAPOLLO-CAME rank=8はstate=`14,146` bytesで、CAMEの`15,372` bytesより約8.0%少ない。CAMEのlr=`3e-3`はCUDA optimizer時間`3.264` ms、APOLLO-CAME rank=8の同条件は`2.468` msだった。ただしpeak allocatedは小型MLPでは両者とも`17,117,184` bytesで差がなく、optimizer state削減がそのままpeak VRAM削減を意味しない。これらは小型MLP固有の値であり、実モデルで再測定する。

#### norm-growth rate sweep（2026-09-12）

`output/optimizer-convergence-growth.json`では、同じ小型MLPをCUDA/BF16、warmup=`5`、
steps=`200`、batch=`64`で、APOLLO系の`norm_growth_rate`を`1.01, 1.05, 1.1`で比較した。
rank=`1,8`、learning rate=`1e-3,3e-3`、scale=`1.0`の全30 caseが`passed`だった。

| optimizer | 最良final loss | 条件 | persistent state |
|---|---:|---|---:|
| APOLLO | 6.330 | rank=1, lr=3e-3, growth=1.1 | 2,904 B |
| APOLLO-CAME | 0.0205 | rank=8, lr=3e-3, growth=1.1 | 14,166 B |
| APOLLOMini | 7.210 | lr=3e-3, growth=1.1 | 2,904 B |

`growth=1.01`から`1.1`へ緩めると、APOLLOはfinal loss=`21.514`から`6.330`、
APOLLOMiniは`23.321`から`7.210`へ改善した。ただしlimiter無効時の最良値
（APOLLO=`1.130`、APOLLOMini=`1.944`）には届かず、limiterを完全に無効化する代替とは
言えない。APOLLO-CAME rank=8はgrowthによる差が小さく、`0.0205`〜`0.0219`で安定した。
state bytesはgrowth値によらず各optimizer内で一定だった。これはsynthetic MLPの結果なので、
続くImageAE probeではlimiter有効/無効とrankを比較した。growth値そのものの実モデル比較はこのsnapshotには含まれない。

#### ImageAE optimizer convergence probe（2026-09-12）

`output/image-ae-optimizer-convergence.json`と`output/image-ae-optimizer-convergence-no-limiter.json`では、実際の`image_ae.train.ImageAE`を使い、`residual_conv_ffn` encoder/decoder、latent=`8`、bottleneck=`64`、downsample stages=`2`、image size=`32`の構成をCUDA/BF16、batch=`8`、steps=`200`、lr=`2e-4`で比較した。入力画像は固定したsynthetic画像で、dataset/VAEのロードは行っていない。

| optimizer | limiter | final loss | persistent state | CUDA optimizer step | peak allocated |
|---|---|---:|---:|---:|---:|
| CAME | — | 0.000248 | 2,847,984 B | 14.250 ms | 10,199,040 B |
| APOLLO | on | 0.093630 | 391,948 B | 8.677 ms | 16,260,608 B |
| APOLLO | off | 0.033075 | 391,772 B | 5.153 ms | 16,238,080 B |
| APOLLO-CAME | on | 0.076028 | 616,806 B | 16.046 ms | 16,507,392 B |
| APOLLO-CAME | off | 0.004089 | 616,634 B | 11.247 ms | 16,485,376 B |
| APOLLOMini | on | 0.098207 | 58,240 B | 9.549 ms | 15,938,048 B |
| APOLLOMini | off | 0.075311 | 58,064 B | 5.775 ms | 15,915,520 B |

limiter無効によりAPOLLO-CAMEのfinal lossは`0.0760`から`0.00409`へ改善したが、CAMEの`0.000248`には届かなかった。stateはCAME比でAPOLLOが約86%、APOLLO-CAMEが約78%、APOLLOMiniが約98%少ない。一方、peak allocatedはCAMEよりAPOLLO系が約6 MB大きく、BF16 gradientのFP32化・scaled updateなどstep中の一時領域が支配している。したがって、ImageAEでもstate削減は確認できたが、peak VRAM削減を主張するには一時tensorの再利用・削除を別途最適化する必要がある。続くprobeでは現行に近いlatent=`16`・bottleneck=`256`・downsample stages=`3`構成を測定した。

同じImageAE probeでrank=`1,4,8`を追加比較した。APOLLOのfinal lossはそれぞれ`0.0549`、`0.0394`、`0.0331`で、rank増加による改善は緩やかだった。APOLLO-CAMEはrank=1では`0.2279`まで悪化したが、rank=4で`0.00507`、rank=8で`0.00409`まで収束した。rank=4からrank=8への改善は小さい一方、stateは`337,450` bytesから`616,634` bytesへ増加し、peak allocatedも`16,207,872` bytesから`16,485,376` bytesへ増えた。現時点ではAPOLLO-CAME rank=4をstateと収束のバランスがよい候補とする。

APOLLO-CAME rank=4についてlr=`1e-4,2e-4,5e-4`、scale=`0.5,1.0`を追加比較した。final lossはlrとscaleの増加に伴い改善し、`lr=5e-4, scale=1.0`で`0.00265`が最良だった。CAMEの同条件は`0.000181`である。最良条件のAPOLLO-CAMEはstate=`337,450` bytes（CAME比約88%削減）、CUDA optimizer step=`13.25` ms（CAMEの`16.23` msより約18%短縮）だった。ただしpeak allocatedは`16,207,872` bytesでCAMEの`10,199,040` bytesより大きい。現時点のImageAE候補を`APOLLO-CAME rank=4, lr=5e-4, scale=1.0, limiter無効`とし、続くprobeで現行に近い構成を再検証した。

#### ImageAE current-shape probe（2026-09-12）

`output/image-ae-real-config.json`では、latent=`16`、bottleneck=`256`、downsample stages=`3`、image size=`32`、batch=`4`の現行に近いImageAE構成をCUDA/BF16、steps=`200`、lr=`5e-4`、limiter無効で比較した。入力は固定synthetic画像である。

| optimizer | final loss | persistent state | CUDA optimizer step | peak allocated |
|---|---:|---:|---:|---:|
| CAME | 0.06561 | 46,834,258 B | 20.675 ms | 89,097,216 B |
| APOLLO-CAME rank=4 | 0.0000159 | 1,488,170 B | 14.657 ms | 46,384,640 B |

この構成ではAPOLLO-CAMEがCAMEよりstateを約96.8%、peak allocatedを約47.9%、optimizer step時間を約29.1%削減し、final lossも低かった。小型probeで見られた「stateは減るがpeak VRAMは増える」という傾向が、大きい実用形状では逆転している。APOLLO-CAMEは21個の行列を低rank path、40個をCAME fallbackへ送り、auto fallbackがparameter形状に応じて機能している。ただし固定synthetic画像への200 step測定であり、実データの再構成品質・validation lossを示さないため、続くCIFAR-10学習では実データで比較した。

追加の`output/image-ae-current-config-limiter-off.json`では、同じ構成をbatch=`8`、limiter無効、全4 optimizerで比較した。全caseが`passed`で、APOLLOはfinal loss=`0.00110`、APOLLO-CAME rank=4は`0.0000437`、APOLLOMiniは`0.0121`、CAMEは`0.0728`だった。APOLLO-CAMEはCAME比でstateを約96.8%、peak allocatedを約47.9%、optimizer step時間を約8.2%削減した。APOLLOもstate=`889,068` B、peak=`45,857,792` B、step=`7.55` msで、この固定probeではCAMEより軽量かつ低lossだった。ただしlimiter有効条件が未取得の段階では、この差をlimiter無効の効果と断定しなかった。続く測定でlimiter有効条件と比較した。

`output/image-ae-current-config-limiter-on.json`で同じ条件のlimiter有効版も全case `passed`だった。CAMEはfinal loss=`0.0728`で変わらない一方、APOLLOは`0.0827`、APOLLO-CAMEは`0.0734`、APOLLOMiniは`0.0862`までしか低下しなかった。limiter無効版と比べてstateは約240 B増え、peak allocatedの差は約0.03 MB以下だったが、optimizer stepはAPOLLO-CAMEで`18.78` msから`23.92` msへ増加した。この現行形状ではlimiter無効候補の収束差が明確だが、既定値を全モデルへ変更せず、後続のCIFAR-10実データで有効/無効を比較した。

#### CIFAR-10 short training（2026-09-12）

現行に近いImageAE構成（latent=`16`、bottleneck=`256`、downsample stages=`3`、image size=`32`）で、CIFAR-10全train splitを1 epoch、batch=`8`、lr=`5e-4`で学習した。両runとも実行時dtypeはFP32で、学習終了・validation・checkpoint保存・再構成artifact生成に成功した。

| optimizer | train reconstruction | validation reconstruction | step time | steps/s |
|---|---:|---:|---:|---:|
| CAME | 0.011237 | 0.007496 | 42.877 ms | 23.32 |
| APOLLO-CAME rank=4, limiter off | 0.008442 | 0.004608 | 35.402 ms | 28.25 |

APOLLO-CAMEはCAME比でvalidation lossを約38.5%低下させ、step timeを約17.4%短縮した。checkpointは両方とも61 tensors・5,201,251 elements・約20.8 MBで、safetensorsの再読込も成功した。再構成画像は両runで生成できたが、1 epochの固定評価だけでは最終品質の優劣を確定しない。また、今回のtrain metricsにはpeak VRAMがないため、VRAM削減はImageAE probeの結果と混同しない。

5 epochのCIFAR-10測定値は[APOLLO experiment results](../apollo-experiment-results.md#cifar10-five-epoch)に集約した。CAMEとAPOLLO-CAME rank=4は各epochでvalidation lossが改善し、checkpointとresume stateを保存した。実行時dtypeはFP32で、peak VRAMはtrain metricsに含まれない。この固定seedの短期比較だけでoptimizerの既定値は変更しない。

この比較結果と、Projection Refreshを構造化探索ノイズとして解釈する仮説は、設計目標・実測事実・未検証の実験候補を分離した[APOLLO研究記録](../apollo-experiment-records.md)にまとめた。refresh後のloss spikeがFlat Minima探索を引き起こすとは断定せず、Frozen projection、refresh方式、update norm、recovery time、sharpnessを含む対照実験が必要である。

#### APOLLO-SPR smooth refresh予備測定（2026-09-12）

APOLLO/APOLLO-CAMEの`R_update`へ、旧projectionと新projectionのlow-rank momentを
double-bufferする`smooth` refreshを追加した。CPU・FP32のCIFAR-10固定subsetで、
seed=`0`、train/validation=`16/16`、batch=`4`、epochs=`2`、rank=`2`、
`update_proj_gap=2`、window=`2`、`smoothstep`、moment transportの条件を実行した。

| optimizer | hard/reset state | smooth/transport state | smooth final validation |
|---|---:|---:|---:|
| APOLLO | 466,172 B | 890,476 B | 0.0657502 |
| APOLLO-CAME | 866,804 B | 1,687,708 B | 0.0805906 |

smoothはwindow中のstateが概ね2系統になるため、persistent stateが約1.9〜2.0倍になった。
この測定は実装・有限性・state量の確認であり、品質や探索性の優位性を示さない。smooth window
途中のcheckpoint/resume後にparameterとlow-rank stateが一致するunit testも追加済みである。
CUDAのpeak VRAM、step時間、長時間validation、hard/resetおよびfixed projectionとの品質比較は
未実行である。

APOLLO-CAME rank=8のvalidation reconstruction、step time、およびcheckpoint再読込結果も[集計表](../apollo-experiment-results.md#cifar10-five-epoch)に統合した。rank=4/8はvalidation値・速度とも近く、この短期結果ではrank=4を優先候補とした。

学習ループ修正後、同じ現行に近い構成をbatch=`32`、CIFAR-10全train split、1 epochで再実行した。修正前は`global_step=1`となっていたが、修正後は`global_step=1563`で全バッチが個別にoptimizer更新された。実行時dtypeはFP32で、run IDはCAMEが`image_ae.train_20260912T103437Z_a78241e2`、APOLLO-CAMEが`image_ae.train_20260912T103628Z_78353597`である。

| optimizer | train reconstruction | validation reconstruction | state | step time | steps/s |
|---|---:|---:|---:|---:|---:|
| CAME | 0.012940 | 0.010611 | 46,834,380 B | 46.385 ms | 21.56 |
| APOLLO-CAME rank=4, limiter off | 0.017493 | 0.008025 | 1,488,172 B | 39.914 ms | 25.05 |

APOLLO-CAMEはCAME比でvalidation reconstructionを約24.4%、step timeを約14.0%、optimizer stateを約96.8%削減した。train reconstructionはCAMEより高いため、1 epochのvalidation値だけで品質優位を断定しない。resume fileはCAMEが約46.99 MB、APOLLO-CAMEが約1.64 MBだった。peak VRAMはこのtrain metricsでは未記録なので、ImageAE probeのpeak値とは分離して扱う。

今回、行列ParameterのBF16 gradientをFP32へ変換した行列viewを、低ランク統計計算とscaled update生成で共有するようにした。従来は`_apply_scaling()`でもう一度`grad.float()`を呼んでいたため、大きな一時FP32テンソルと追加のメモリ帯域が発生していた。更新則とoptimizer stateの構成は変更していない。実GPUではAPOLLO step時間とpeak allocatedを同一条件で再計測する。

APOLLOの低ランクAdam統計では、`low_rank_grad.square()`を`exp_avg_sq.addcmul_(low_rank_grad, low_rank_grad, ...)`へ置き換え、bias correctionのscalar倍率もin-placeで適用する。低ランクsquare用の一時Tensorと不要なcopyを抑える変更であり、optimizer stateの形状・更新則は維持する。`apollo_low_rank_stats`とAPOLLO全体を同一条件で再計測する。

通常のAPOLLO/APOLLO-CAMEではLR controllerを使用しないため、AutoSchedule用のparameter norm、scale norm、update normの集計は不要である。stepごとに発生していたこれらの追加reductionと一時scalar Tensorを、AutoSchedule版でのみ実行するようにした。AutoSchedule版のcontroller stateと統計は従来どおり維持する。続いて通常APOLLOの`optimizer[APOLLO]`を同一条件で再計測した。

実測では、最適化前のログ（`20260911_185435`）に対して、最適化後のログ（`20260911_190602`、各ログの初回区間を除外）で、`seconds_per_optimizer_step.optimizer[APOLLO]`が平均0.462秒から0.264秒へ、`host_seconds_per_optimizer_step.optimizer[APOLLO]`が平均0.432秒から0.249秒へ低下した。GPU時間で約43%、host時間で約42%の削減である。通常時のAPOLLO処理はまだbackwardに次ぐ大きさなので、次はbackwardとDiT attentionの改善を優先する。

その後、MMDiTおよびContext Transformerのゲート付き残差を`x + gate * update`から`torch.addcmul(x, update, gate)`へ置き換えた。数学的な更新則は維持しつつ、elementwise乗算と加算のkernel launch削減を狙う変更である。GPUでは12層のMMDiT全体で効果を再計測し、改善しない場合は元の表現へ戻す。

## 小規模なOptimizer提案の対応状況

- **Parameter loopの共通化:** helperや`foreach`案は性能効果の測定・実装記録がなく、優先課題にはしなかった。現行契約は[optimizer documentation](../optimizers.md)を参照。
- **AutoSchedule preview:** group stateを変更せず旧checkpointの欠損値を一時viewで補う。実装は[`optimizers/auto_schedule.py`](../../optimizers/auto_schedule.py)、回帰テストは[`tests/unit/test_optimizer_adamw.py`](../../tests/unit/test_optimizer_adamw.py)。
- **Optimizer memory report:** parameter、persistent state、一時領域、allocator reservedを分ける案は後続で実装済み。容量表示とstate推定は[`core/utils.py`](../../core/utils.py)、計測項目は[optimizer verification report](../../verify/optimizers.py)と[optimizer probe guide](../../verify/optimizer-probes.md)を参照。この案は未対応タスクではない。
