# APOLLO experiment results

> Historical research record. Protocol proposals and completion state refer to their recorded context; they are not current work instructions. This document preserves selected aggregate observations and conditions; local run outputs and per-run reports are not included in the public tree. The current optimizer contract is in [optimizers](optimizers.md).

この資料は主にImageAE/CIFAR-10でのAPOLLO系optimizerとprojection refreshの集計記録です。TinyStoriesで行ったLRTDO、Schedule-Free low-rank、APOLLO-Confの比較は[LRTDO研究要約](history/lrtdo-research-summary.md)を参照してください。両資料はtaskと実験条件が異なるため、結果を直接比較しません。

## 1. このリポジトリで確認した事実（記録時点）

<a id="cifar10-five-epoch"></a>

現在までのImageAE/CIFAR-10測定では、latent=`16`、bottleneck=`256`、downsample stages=`3`、batch=`8`、5 epoch、FP32の条件で次を確認した。

| optimizer | 最終train reconstruction | 最終validation reconstruction | 平均step time |
|---|---:|---:|---:|
| CAME | 0.005794 | 0.005606 | 38.62 ms |
| APOLLO-CAME rank=4、limiter無効 | 0.002858 | 0.002843 | 34.89 ms |
| APOLLO-CAME rank=8、limiter無効 | — | 0.002867 | 35.41 ms |
| APOLLO rank=4、limiter無効 | — | 0.003034 | 24.72 ms |
| APOLLO-Mini、limiter無効 | — | 0.003214 | 23.32 ms |

この結果は、今回の固定seed・5 epoch・CIFAR-10条件での観測である。APOLLO系の既定optimizer化、一般的なAdamW超え、Flat Minima到達を直接示すものではない。AdamW、CAME、APOLLO系を同一条件で比較する専用probeは追加済みである。optimizer-onlyのCUDA runtime出力は公開アーカイブに含めておらず、その比較値は本資料にも集計していない。

記録ではCAMEとAPOLLO-CAME rank=4が全epochを完了し、各epochでvalidation lossが改善した。9件のcheckpointのsafetensors再読込にも成功している。実行時dtypeはFP32。peak VRAMはtrain metricsに含まれていないため、この表からは比較できない。

### APOLLO refreshのCUDA/BF16予備比較（2026-09-13）

専用probeをCUDA/BF16、seed=`0`、5 epoch、CIFAR-10 train/validation各512枚、batch=`8`、
rank=`4`、learning rate=`5e-4`、`update_proj_gap=200`で実行した。refresh方式以外の条件は共有している。
このrunでは各方式でrefresh開始は1回（step 200）、smoothのactive stepは32、orthogonalは320 stepだった。

| optimizer / mode | validation loss | state bytes | step time | 観測 |
|---|---:|---:|---:|---|
| AdamW / none | 0.0183339 | 41,610,000 | 3.52 ms | 基準optimizer |
| CAME / none | 0.0138432 | 46,830,000 | 22.98 ms | 品質基準 |
| APOLLO / none | 0.0629255 | 889,300 | 13.73 ms | 固定projection |
| APOLLO / hard | 0.0629255 | 889,300 | 15.02 ms | event=1、最終lossはnoneと同じ |
| APOLLO / smooth-ema | 0.0629255 | 889,300 | 15.57 ms | active=32、最終lossはnoneと同じ |
| APOLLO / smooth-stochastic | 0.0629255 | 889,300 | 14.54 ms | active=32、最終lossはnoneと同じ |
| APOLLO / orthogonal | 0.0469532 | 889,300 | 23.94 ms | 320回、noneより約25.4%低いloss |
| APOLLO-CAME / none | 0.0341763 | 1,488,000 | 23.62 ms | 固定projection |
| APOLLO-CAME / orthogonal | 0.0624228 | 1,488,000 | 36.58 ms | noneより悪化、約54.9%遅い |
| APOLLO-Mini / none | 0.0661612 | 254,000 | 13.96 ms | 固定projection |
| APOLLO-Mini / orthogonal | 0.0474755 | 254,000 | 26.71 ms | 320回、noneより約28.2%低いloss |

3 seedの平均validation lossは次の通りである。hard、smooth-ema、smooth-stochasticは、
全optimizerでnoneと同じ値になった。

| optimizer / mode | none・hard・smooth系 | orthogonal | orthogonalの変化 |
|---|---:|---:|---:|
| APOLLO | 0.0614073 | 0.0471276 | 3 seedすべて改善 |
| APOLLO-CAME | 0.0377350 | 0.0557100 | 3 seedすべて悪化 |
| APOLLO-Mini | 0.0650781 | 0.0500587 | 3 seedすべて改善 |

hard/smoothのrefresh eventでは、`projection_change_max_abs`がAPOLLO/APOLLO-CAMEで
`2.63891`、APOLLO-Miniで`4.42190`となり、projection tensorの実変化は確認できた。
したがって、最終lossの一致はrefresh未発生ではなく、この条件で品質差に現れなかった結果である。

### Update normと探索分散（2026-09-13）

`--record-update-norms`を有効にして同じ3 seedを再実行した。`none`とorthogonalの3 seed平均は次の通りである。
この計測はparameterのCPU copyを伴うため、通常の速度・peak memory比較とは分離して扱う。

| optimizer | mode | update norm mean | update norm variance |
|---|---|---:|---:|
| APOLLO | none | 0.010336080 | 0.000229408 |
| APOLLO | orthogonal | 0.017267730 | 0.009857378 |
| APOLLO-CAME | none | 0.041413809 | 0.035390234 |
| APOLLO-CAME | orthogonal | 0.127872791 | 1.747948958 |
| APOLLO-Mini | none | 0.010296139 | 0.000056469 |
| APOLLO-Mini | orthogonal | 0.017229394 | 0.009052850 |

orthogonalではupdate norm varianceが、APOLLOで約43倍、APOLLO-CAMEで約49倍、APOLLO-Miniで
約160倍になった。APOLLOとAPOLLO-Miniではこの分散増加とvalidation改善が同時に観測されたが、
APOLLO-CAMEでは分散増加が過剰でvalidationが悪化した可能性がある。これは「探索ノイズがある」
ことの証拠にはなるが、flat minima到達や因果効果の証明ではない。

hard/smooth-ema/smooth-stochasticのrefresh event後は、APOLLOとAPOLLO-Miniでloss deltaが3 seed
とも負、recoveryは1 stepだった。APOLLO-CAMEはseed=0だけ一時的な正のloss deltaと3 step recoveryを
示し、seed=1,2では1 step recoveryだった。loss-directed方式では単にlossを最大化するのではなく、
この過大なupdate norm varianceを制約しながら、controlled spikeとrecoveryを作ることを目標にする。

この3 seedの予備runでは、APOLLOのpersistent stateはAdamW比で約97.9%、APOLLO-CAMEはCAME比で
約96.8%、APOLLO-MiniはAdamW比で約99.4%小さい。これはoptimizer stateの比較であり、
peak reservedはCUDA allocatorのwarmup・実行順序の影響を受けるため、VRAM削減率と同一視しない。

orthogonal refreshはAPOLLOとAPOLLO-Miniでは有望な候補だが、step timeはそれぞれ約74.2%、
約91.2%増加し、APOLLO-CAMEでは品質が悪化した。hard、smooth-ema、smooth-stochasticは
3 seedすべてでnoneと同じ最終lossであり、品質優位や探索性は主張しない。step時間の最終比較は
修正版の結果を使う。`--record-update-norms`を付けた追加runは実施済みであり、次のloss-directed比較へ引き継ぐ。

### Loss-directed proxyのCUDA/BF16比較（2026-09-13）

`direction="loss_directed"`を指定したorthogonal-only条件を、既存の`none`およびrandom
orthogonalと同じCIFAR-10 subset・batch順序・rank・seedで比較した。3 seed、各320回の毎step回転、
全24ケースが`passed`だった。ここでのloss-directedは実損失を評価せず、射影勾配エネルギーを
増やす一次近似である。

| optimizer | none平均 | random orthogonal平均 | loss-directed平均 | loss-directed - none | loss-directed step倍率 | update variance倍率 |
|---|---:|---:|---:|---:|---:|---:|
| APOLLO | 0.0614073 | 0.0471276 | 0.0472014 | -0.0142059 | 1.65x | 40.7x |
| APOLLO-CAME | 0.0377350 | 0.0557100 | 0.0558513 | +0.0181164 | 1.53x | 45.8x |
| APOLLO-Mini | 0.0650781 | 0.0500587 | 0.0499325 | -0.0151456 | 1.70x | 152.7x |

loss-directedはAPOLLOとAPOLLO-Miniでnoneより改善し、APOLLO-CAMEでは悪化した。このoptimizer別の
傾向はrandom orthogonalと同じで、3 seed平均ではrandomを上回る品質改善は確認できなかった。
randomとの差はAPOLLOで`+0.0000739`、APOLLO-CAMEで`+0.0001414`、APOLLO-Miniで`-0.0001262`
であり、この条件ではproxyの優位性は小さい。step時間はnoneの約1.5--1.7倍、peak allocatedは
約1.07倍で、update norm varianceは従来のrandom orthogonalと同程度に大きくなった。

したがって、loss-directed proxyは実装・再現性の確認まで完了したが、random orthogonalの代替や
既定化を支持する結果ではない。次は回転率のsweep、optimizer別のvariance cap、実lossを使わない
候補方向の改良を優先し、flatness効果は引き続き未検証仮説として扱う。

### Orthogonal rate sweepのCUDA/BF16比較（2026-09-13）

`rate={0.001, 0.005, 0.01, 0.02, 0.05}`、`direction={random, loss_directed}`、seed=`0,1,2`
を同じprobeで比較した。90行（2方向×5 rate×3 seed×3 optimizer）はすべて`passed`だった。
以下はnoneに対するvalidation loss deltaとupdate norm variance倍率のseed平均である。

| direction | rate | APOLLO Δ / variance | APOLLO-CAME Δ / variance | APOLLO-Mini Δ / variance |
|---|---:|---:|---:|---:|
| random | 0.001 | -0.014279 / 43.0x | +0.017751 / 49.7x | -0.015011 / 160.2x |
| random | 0.005 | -0.014281 / 43.0x | +0.017717 / 49.7x | -0.015017 / 160.2x |
| random | 0.01 | -0.014280 / 43.0x | +0.017975 / 49.7x | -0.015019 / 160.3x |
| random | 0.02 | -0.014277 / 43.0x | +0.017942 / 49.7x | -0.015019 / 160.3x |
| random | 0.05 | -0.014268 / 43.2x | +0.017970 / 49.8x | -0.015016 / 160.6x |
| loss_directed | 0.001 | -0.014271 / 42.7x | +0.017604 / 49.3x | -0.015015 / 159.8x |
| loss_directed | 0.005 | -0.014245 / 41.7x | +0.017482 / 47.6x | -0.015032 / 156.6x |
| loss_directed | 0.01 | -0.014206 / 40.7x | +0.018116 / 45.8x | -0.015146 / 152.7x |
| loss_directed | 0.02 | -0.014137 / 38.9x | +0.023348 / 42.8x | -0.015139 / 146.3x |
| loss_directed | 0.05 | -0.013820 / 35.7x | +0.021356 / 37.7x | -0.015087 / 134.6x |

randomはこのrate範囲ではlossとvarianceがほぼ一定だった。loss-directedはrateを上げると
update norm varianceが下がる傾向を示したが、APOLLO-CAMEのlossは`0.02`以降に悪化した。
APOLLO-Miniでもvarianceは下がる一方、品質改善量はほぼ一定だった。step時間はnone比で概ね
1.35--1.95倍の範囲にあり、rateを上げれば単純に高速化する傾向はない。

このため、variance capの目標値を全optimizer共通の倍率だけで決めるのは不適切である。まずは
optimizerごとのbaseline varianceと品質劣化を基準に、parameter updateを直接clipする方式と
orthogonal rateを適応的に下げる方式を分けて比較する。集計値は上表に示す。

### Update norm variance capの実験API（2026-09-13）

上記の比較に使える最小の実験APIとして、APOLLO/APOLLO-CAMEへ
`--update-norm-variance-cap V`を追加した。capはoptimizer update（decoupled weight decayの前）
のnormを累積Welford統計で追跡し、新しい上側サンプルが指定分散を超える場合だけupdate全体を
縮小する。過去の統計やAdam/CAME momentを遡って書き換えず、既に履歴分散がcapを超えている
場合はそのstepを無理に補正しない。このため、通常のgradient clippingやmoment clippingとは
異なる実験条件である。

capは既定で無効で、checkpoint stateにはnormのcount/mean/M2とcap回数だけを追加する。
weight decayとの適用順序、APOLLO-CAMEの実際のparameter updateが二重適用されないこと、
state保存に必要な情報はunit testで固定した。CPUの小規模CIFAR-10 smokeではAPOLLO、
APOLLO-CAME、APOLLO-Miniの実行とcap回数の記録を確認済みである。

ただし、cap値はoptimizerごとにupdate normのスケールが異なるため共通値を既定化していない。
CUDA/BF16で、capなし・複数cap値・orthogonal rateの適応制御を分離して比較することが次の課題で
あり、現時点で品質改善やFlat Minima効果を主張するものではない。

`cap=0.001`のseed=0 sanity runを、スカラー倍率経路への変更後に再実行した。全5ケースが`passed`し、
APOLLO/APOLLO-CAME/APOLLO-Miniのcapped stepはそれぞれ`1/11/5`だった。capなしseed=0との差は
validation lossで`+0.00000005/+0.00002891/+0.00000014`であり、この1 seedではAPOLLOと
APOLLO-Miniの差は実質的に観測されず、APOLLO-CAMEは僅かに悪化した。persistent stateはcap state
分として各APOLLO系ケースで`+176 bytes`、peak allocatedの差は`+22,528 bytes`まで減少した。
一方、step時間はAPOLLO/APOLLO-CAME/APOLLO-Miniでそれぞれ約`2.25x/1.90x/1.99x`だったため、
一時tensorのpeak overheadは抑制できたが、norm計算の速度コストは残っている。
初版では毎stepにeffective updateをmaterializeしていたが、スカラー倍率を累積統計へ渡し、cap発動時
だけ`addcmul_`する経路へ変更したことで、peak allocatedの増加は大きく縮小した。

### Update norm variance cap sweep（2026-09-13）

`cap={none, 0.0001, 0.001, 0.01}`、seed=`0,1,2`で、各capのnoneとの差を比較した。全36行は
`passed`だった。以下はoptimizer・capごとのseed平均である。

| optimizer | cap | validation loss delta | step倍率 | capped steps平均 |
|---|---:|---:|---:|---:|
| APOLLO | 0.0001 | +0.00000312 | 1.71x | 1.3 |
| APOLLO | 0.001 | +0.00000849 | 1.65x | 1.3 |
| APOLLO | 0.01 | +0.00000285 | 1.80x | 1.3 |
| APOLLO-CAME | 0.0001 | -0.00005896 | 1.40x | 15.7 |
| APOLLO-CAME | 0.001 | -0.00005348 | 1.42x | 15.7 |
| APOLLO-CAME | 0.01 | -0.00006826 | 1.36x | 15.7 |
| APOLLO-Mini | 0.0001 | +0.00000983 | 1.77x | 5.3 |
| APOLLO-Mini | 0.001 | +0.00000865 | 1.76x | 5.3 |
| APOLLO-Mini | 0.01 | +0.00000599 | 1.61x | 3.7 |

APOLLO-CAMEでは全cap値で僅かな改善が出たが、APOLLOとAPOLLO-Miniでは改善せず、step時間は
概ね`1.36--1.80x`に増えた。したがって、capはoptimizer共通の既定値にはせず、APOLLO-CAMEの
長期・複数seed検証と、capなしに対するparameter update clippingおよびorthogonal rate適応制御を
別条件として比較する。集計値は上表に示す。
このsweepはcap回数をPython integer stateで保持していた版の測定である。その後、毎stepのscalar
同期を避けるためcap判定とcapped countをdevice tensor化したため、現行版ではpersistent stateの
見積もりとstep時間が変わり得る。上記結果は履歴として残し、現行版での同条件再測定を優先する。

### 1.1 APOLLO projection state transportの予備比較

`ProjectionRefreshPolicy`の共通契約をAPOLLOの`R_update`側にも適用し、hard refresh時の低rank momentを
`reset`またはprojection overlapによる`transport`で扱えるようにした。実装の数値契約は
[`optimizers.md`](optimizers.md)とunit testを正とする。smooth refreshでは旧新low-rank momentを
double-bufferし、window中にscalingをmixする。

実装後、CPU・FP32の小規模CIFAR-10 probeで、同じ初期状態・seed・batch順序を使って`reset`と
`transport`を比較した。条件は次の通りである。

```text
seed=0, epochs=2, train/validation=16/16, batch=4
latent=1, bottleneck=16, downsample stages=2, rank=2
learning rate=5e-4, update_proj_gap=2, dtype=FP32
```

| optimizer | refresh state | 最終validation | 平均update norm | update norm variance | 平均optimizer step time |
|---|---|---:|---:|---:|---:|
| APOLLO | reset | 0.078671 | 0.017028 | 0.000778 | 4.470 ms |
| APOLLO | transport | 0.078936 | 0.016750 | 0.000759 | 4.653 ms |
| APOLLO-CAME | reset | 0.063616 | 0.276752 | 0.087689 | 5.509 ms |
| APOLLO-CAME | transport | 0.067768 | 0.305437 | 0.113508 | 6.312 ms |

このrunではtransportによる最終validationの改善は見られなかった。一方、APOLLO-CAMEでは、
step 4のrefresh後に最初に観測されたloss差分がresetの`+0.002025`からtransportの`+0.000847`へ
小さくなった。ただし、これは異なるbatch間のloss差分であり、refreshの因果効果やFlat Minimaを
示すものではない。サンプル数・epoch数が小さく、CPUのみで実施した予備観測なので、transportを
既定化する判断には使わない。代表的なbatch/epoch数での再現性、AdamW・Frozenとの対照、CUDAの
peak memoryとstep時間は引き続き未検証である。

### 1.2 `none` / `hard` modeの実行契約

本番CLIと同じ共有factory経路を使い、CPU・FP32のImageAE/CIFAR-10 probeでmodeの実行契約を確認した。
seed=`0`、train/validation=`16/16`、batch=`4`、epochs=`2`、rank=`2`、
`update_proj_gap=2`の条件では、`none`はAPOLLO/APOLLO-CAMEともrefresh eventが`0`件、
`hard`は両方とも`4`件（step 2, 4, 6, 8）になった。最終validationは次の通りである。

| optimizer | none | hard |
|---|---:|---:|
| APOLLO | 0.0657492 | 0.0657494 |
| APOLLO-CAME | 0.0805918 | 0.0805918 |

この結果はmode・interval・event記録が接続されていることの確認であり、2 epoch・16画像のため、
refreshの品質効果や探索性の比較結果とは扱わない。

### 1.3 APOLLO-SPRのstateコスト予備測定

同じCPU・FP32・seed=`0`・train/validation=`16/16`・batch=`4`・epochs=`2`・rank=`2`・
`update_proj_gap=2`の条件で、`smooth`（window=`2`、`smoothstep`、transport）を実行した。
現行に近いlatent=`16`、bottleneck=`256`、downsample stages=`3`では、smooth中の二系統momentに
よってpersistent stateが増加した。

| optimizer | hard/reset state | smooth/transport state | smooth最終validation |
|---|---:|---:|---:|
| APOLLO | 466,172 B | 890,476 B | 0.0657502 |
| APOLLO-CAME | 866,804 B | 1,687,708 B | 0.0805906 |

この差はsmooth refreshの実装コストを示すもので、品質優位を示さない。state・step時間・loss
spikeの三者を分けて、CUDAおよび長時間学習で再測定する必要がある。

## 2. 記録時点の判断

現状のCIFAR-10 5 epoch測定では、APOLLO-CAME rank=4・limiter無効がlossとstateのバランス候補、APOLLO-Miniがstep時間候補である。一方、次の点は未確定である。

1. AdamWに対する公平な実データ比較結果
2. refreshの有無がloss spike・recovery・validationへ与える因果効果
3. projection近似と探索ノイズの寄与の分離
4. Flat Minima・sharpness・Hessian指標との相関
5. seed・epoch数・dataset規模を変えた場合の再現性

したがって、APOLLOを「AdamWの低メモリ代替」として評価する段階から、「低メモリ更新に加えて構造化探索を持つ可能性があるoptimizer」として検証する段階へ進める。ただし後半の解釈は、上記の対照実験が完了するまで仮説として扱う。
