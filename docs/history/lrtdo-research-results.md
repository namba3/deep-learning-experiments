# LRTDO recorded research results

最終更新: 2026-09-14

この文書は[LRTDO研究要約](lrtdo-research-summary.md)に対応する詳細記録です。未実施のphase案や将来のbenchmark計画は圧縮し、実測値と解釈条件を残しています。「next」「candidate」などは記録時点の表現です。

LRTDOはSchedule-Free trajectory driftやoptimizer stateを低rank表現で扱う研究テーマ名です。APOLLOのgradient projection `R_update`とSchedule-Free差分projection `R_delta`は異なる状態・役割です。現行optimizerの契約・既定動作は[`../optimizers.md`](../optimizers.md)とコードを参照してください。

この詳細記録は主にTinyStoriesのSchedule-Free low-rank、confidence variant、APOLLO-Conf比較を扱います。ImageAE/CIFAR-10でのAPOLLO系optimizerとprojection refreshの結果は[APOLLO実験結果](../apollo-experiment-results.md)にあります。taskと条件が異なるため、両アーカイブの数値は直接比較しません。

初期rank診断にはparameter dtype不一致があり、state-bytes測定を無効値として扱う注記があります。各結果の条件・制約と併せて読んでください。

元のrun JSON、ログ、checkpointは公開treeに含めず、ここには再利用する集計値と解釈条件を残しています。

## 記録された診断・比較結果

### 初期診断とrank比較

<a id="diagnostic-initial"></a>
#### 初回診断結果

修正前に実行した低負荷run（metadata上はBF16、seed=`0`、実効8 step、SVD snapshotは
step=`5`のみ）では、
先頭attention blockの`512 x 512` weightについて次の傾向が得られた。

ただし、このrunは検証器のparameter storage cast修正前であり、parameter自体はFP32のまま
だった。そのため、同JSONのoptimizer state bytesは現行dtype契約の測定値として無効である。
以下のrank傾向は探索上の参考値に留め、修正後のBF16 runで再確認する。

| source | Q/K projectionの傾向 | V/O projectionの傾向 | 初期判断 |
| --- | --- | --- | --- |
| `parameter` / `z` | effective rank約`310`、rank=16 energy約`0.11` | 同様にeffective rank約`310` | 単体の低rank化候補ではない |
| `gradient` | effective rank約`11--13`、rank=8 energy約`0.83--0.86` | effective rank約`1.4--1.5`、rank=8 energy約`0.99` | layer role依存の有望候補 |
| `exp_avg_sq` | effective rank約`7`だがrank=8 energy約`0.82` | effective rank約`1.6--1.7`、rank=8 energy約`0.97` | V/Oは候補、Q/Kの直接低rank化は要検証 |
| `sf_delta` | effective rank約`38--42`、rank=8 energy約`0.58--0.60` | effective rank約`4.0--4.6`、rank=8 energy約`0.89--0.90` | V/Oはrank=8候補、Q/Kはrank不足の可能性 |

この結果は、Schedule-Freeで圧縮すべき対象が`z`全体ではなく`sf_delta`であることを
支持する。一方、全layerへ一律rank=8を適用する根拠にはならず、Q/KとV/Oでrankまたは
fallbackを分ける設計が候補になる。`decoded_lrsf_delta`がrank=8 energy=`1.0`になるのは
LRSFの構造上当然であり、圧縮成功の証拠ではない。full Schedule-Freeの`sf_delta`を同じ
projectionへ射影した誤差、またはそのSVD最適rank-r誤差と比較する必要がある。

また、effective rankと`rank_90/95/99`が一致しない場合がある。例えばV/Oの
`exp_avg_sq`は少数の大きな特異値を持つためeffective rankは小さいが、rank=99 energy
には多くの尾部成分が必要になる。したがって、optimizer stateの置換判断ではeffective
rank単独ではなく、目標errorに対応するretained energyを主指標にする。

<a id="diagnostic-ema-rerun"></a>
#### EMA付き診断の再実行結果

検証器のBF16 storage cast修正後に、同じ低負荷条件で`sf_delta_ema`（decay=`0.9`）を
追加測定した。state容量はAdamW-SFが`726,266,368 bytes`（`4 bytes/parameter`）となり、
現行のparameter-dtype state契約と一致した。rank=`8`のAdamW-LRSFは
`376,119,200 bytes`であり、LRSF側の低rank stateはFP32のため、BF16 full stateだけを
単純に2倍した値にはならない。

なお、以下のrank値は`sf_delta`のFP32減算修正前に取得した出力である。state容量の確認には
利用できるが、rank値は修正後に同じ条件で再取得して確定する。

step=`5`の先頭attention blockでは、`sf_delta_ema`のeffective rankとrank=8 retained
energyは次のようになった。

| parameter role | `sf_delta` effective rank / E8 | `sf_delta_ema` effective rank / E8 | 観測 |
| --- | ---: | ---: | --- |
| Q | `80.6 / 0.462` | `88.8 / 0.419` | EMAで空間rankは下がらない |
| K | `75.6 / 0.474` | `84.8 / 0.438` | 同上 |
| V | `12.6 / 0.746` | `19.8 / 0.688` | EMAでrankが広がる |
| O | `14.7 / 0.731` | `22.1 / 0.676` | 同上 |

この短期runでは、EMAは単発deltaのノイズを単純に除去せず、stepごとに異なる方向を
累積したためeffective rankを増加させた。したがって`sf_delta_ema`は「圧縮後stateの
rankを下げる手法」ではなく、「持続的なtrajectory subspaceの広がりを測る診断」として
扱う。低rank化の判断は、EMAのeffective rank単独ではなく、full deltaに対するrank-r
再構成誤差、時間窓PCAの説明分散、optimizer updateとvalidation lossで行う。

現時点の推奨実装は、既存`AdamW-LRSF`を壊さずに診断機能を追加することである。`exp_avg_sq`削減とunified stateは、Phase 1--3の測定結果を根拠に段階的に進める。

<a id="rank-refresh-sweep"></a>
#### rank・refresh interval sweepの結果

TinyStories・CUDA/BF16・3 seed・実効96 stepで、rank=`4, 8, 16`とhard refresh
interval=`25, 50`の6セルを比較した。全セルで完走し、Schedule-Freeのposition sampleも
19点記録できた。

| rank | interval | optimizer | validation loss mean | persistent state | step時間 | eval trajectory normalized roughness |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 4 | 25 | AdamW-SF | 34.6099 | 726.3 MB | 47.01 ms | 1.386 |
| 4 | 25 | AdamW-LRSF | 42.8135 | 369.7 MB | 23.18 ms | 2.322 |
| 4 | 50 | AdamW-LRSF | 42.4689 | 369.7 MB | 24.23 ms | 2.236 |
| 8 | 25 | AdamW-LRSF | 42.5185 | 376.1 MB | 25.38 ms | 2.361 |
| 8 | 50 | AdamW-LRSF | 42.5454 | 376.1 MB | 22.75 ms | 2.391 |
| 16 | 25 | AdamW-LRSF | 42.2601 | 389.1 MB | 27.17 ms | 2.209 |
| 16 | 50 | AdamW-LRSF | 41.8325 | 389.1 MB | 30.04 ms | 2.491 |

`AdamW-SF`はrank・intervalに依存しないため、全セルで同じ値になった。LRSFでは
rank=`4→16`にしてもvalidation lossは改善せず、stateは約`369.7→389.1 MB`へ増加した。
この条件ではrank拡大よりも、射影されたSchedule-Free deltaの近似誤差が支配的であり、
rank=`16`を直ちに標準値とする根拠は得られない。interval=`25/50`の差も小さく、
interval単独で品質を説明できない。

LRSFの`eval_parameter` roughnessはrank・intervalにかかわらず約`2.2--2.5`で、
先行する短期runの「LRSFではhiddenとevalの平滑化差が小さい」という観測と整合する。
`AdamW-SF`のeval roughness=`1.386`より高いため、次の改善対象はrank拡大ではなく、
LRSF deltaの時間平滑化または評価変換の再設計とする。

ただし、これは3 seed・96 stepの短期スクリーニングであり、refresh後の長期挙動や
validation lossの絶対値を最終結論には使わない。

### Refresh transportとrecovery比較

<a id="refresh-transport"></a>
#### 長期refresh transport診断

同じrank=`8, 16`・interval=`25, 50`・3 seed・実効96 stepで、
`--record-refresh-diagnostics`を有効にしてrefresh eventを測定した。診断を有効にしても
validation lossは通常runと一致し、品質測定への介入は確認されなかった。一方、hard refresh
直後のdelta transportは次のように大きく崩れていた。

| rank | interval | transport error mean | norm ratio mean | cosine mean | diagnostic events/case |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 25 | 0.9917 | 0.1272 | 0.1272 | 291 |
| 8 | 50 | 0.9920 | 0.1249 | 0.1249 | 97 |
| 16 | 25 | 0.9834 | 0.1808 | 0.1808 | 291 |
| 16 | 50 | 0.9828 | 0.1836 | 0.1836 | 97 |

rank=`16`はrank=`8`よりわずかに改善するが、transport errorは依然として約`0.98`で、
新basis上のdeltaがほぼ再現できていない。したがって、hard refreshの品質問題は単純な
rank不足だけでなく、basis交換時にtrajectory driftの座標系を失うことが主因と考えられる。
interval=`25/50`で誤差の大きさはほぼ変わらず、refresh頻度を下げるだけでは解決しない。

overlap=`0.9, 0.95, 0.99`について、rank=`8,16`・interval=`25,50`・3 seed・TinyStories・
CUDA/BF16・実効96 stepの診断付き比較を行った。全36ケースが`passed`となった。

| overlap | rank | interval | validation loss mean±std | persistent state | transport error mean | diagnostic events/case |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.9` | 8 | 25 | `42.218 ± 0.427` | `376.1 MB` | `0.1093` | `291` |
| `0.9` | 8 | 50 | `42.381 ± 0.677` | `376.1 MB` | `0.1092` | `97` |
| `0.9` | 16 | 25 | `41.295 ± 0.847` | `389.1 MB` | `0.1087` | `291` |
| `0.9` | 16 | 50 | `41.282 ± 0.838` | `389.1 MB` | `0.1087` | `97` |
| `0.95` | 8 | 25 | `42.223 ± 0.604` | `376.1 MB` | `0.0521` | `291` |
| `0.95` | 8 | 50 | `42.136 ± 0.539` | `376.1 MB` | `0.0520` | `97` |
| `0.95` | 16 | 25 | `41.396 ± 0.513` | `389.1 MB` | `0.0517` | `291` |
| `0.95` | 16 | 50 | `41.375 ± 0.725` | `389.1 MB` | `0.0516` | `97` |
| `0.99` | 8 | 25 | `42.248 ± 0.616` | `376.1 MB` | `0.0100` | `291` |
| `0.99` | 8 | 50 | `42.326 ± 0.423` | `376.1 MB` | `0.0100` | `97` |
| `0.99` | 16 | 25 | `41.428 ± 0.697` | `389.1 MB` | `0.0099` | `291` |
| `0.99` | 16 | 50 | `41.394 ± 0.687` | `389.1 MB` | `0.0099` | `97` |

Transport errorはoverlapを`0.9→0.95→0.99`と増やすと約`0.109→0.052→0.010`へ低下したが、
validation lossに明確な改善傾向はなく、rank/intervalの優劣も一貫しなかった。overlapは状態の
連続性を優先する`0.99`と、新basisの探索性を残す`0.9`を比較候補とする。診断付き測定のため、
この比較のstep時間は性能評価に使わない。後続の診断なし速度・実効300 step比較でも品質差と
step時間差は小さく、overlap自体を品質改善の主手段とはしない。

##### Transport overlapの診断なし速度比較

`overlap=0.9`と`0.99`について、rank=`16`・interval=`50`・3 seed・実効100 stepの診断なし
fair comparisonを実行した。両条件でAdamW-SF、AdamW-LRSF、APOLLO、APOLLO-Confの全12ケースが
`passed`となった。LRSFの比較は次のとおりである。

| overlap | validation loss mean±std | persistent state | peak allocated | peak reserved | step time mean±std |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.9` | `41.282 ± 0.838` | `371.03 MiB` | `3.282 GiB` | `4.896 GiB` | `32.06 ± 3.71 ms` |
| `0.99` | `41.394 ± 0.687` | `371.03 MiB` | `3.282 GiB` | `4.896 GiB` | `31.15 ± 2.16 ms` |

overlapによるstate量・peak VRAMの差はなく、step時間もseed間分散の範囲である。短期品質は
`0.9`がわずかに良いが、差は標準偏差より小さい。したがって、通常設定ではtransport errorを
約`0.010`まで抑えられる`overlap=0.99`を連続性重視の候補とし、探索性を残す比較対象として
`0.9`を維持する。`0.95`の診断なし速度は未測定だが、診断付き結果は両者の中間に位置する。

##### Transport overlapの実効300 step比較

rank=`16`・interval=`50`・3 seed・TinyStories・CUDA/BF16・実効300 step・診断なしで、
`overlap=0.9`と`0.99`を比較した。LRSFの結果は次のとおりである。

| overlap | validation loss mean±std | persistent state | peak allocated | peak reserved | step time mean±std |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.9` | `36.648 ± 0.874` | `371.03 MiB` | `3.282 GiB` | `4.896 GiB` | `31.26 ± 0.92 ms` |
| `0.99` | `36.717 ± 0.799` | `371.03 MiB` | `3.282 GiB` | `4.896 GiB` | `31.50 ± 0.90 ms` |

長期でもstate量・peak VRAMは同一で、loss差は`0.069`に留まりseed間分散より小さい。step時間も
実質同等である。したがって、`overlap=0.99`はtransport continuityを優先する安全側の設定、
`0.9`は同等品質で少し探索性を残す設定として使い分ける。overlap自体を品質改善の主手段とは
せず、次はLRSF deltaの時間平滑化、refresh後のstate再育成、またはrank/learning rateの再設計へ
進む。

##### `AdamW-LRSF-LR` の latent moment reset / transport 長期比較

上記の plain `AdamW-LRSF` ではなく、latent second momentを持つ統合prototype
`AdamW-LRSF-LR`について、rank=`16`・`refresh_transport_overlap=0.99`・hard refresh
interval=`50`・TinyStories・CUDA/BF16・3 seed・実効300 step・診断なしで、refresh時の
latent moment処理を比較した。`reset`は新basis上のlatent second momentをゼロ化し、局所的な
bias correctionを再開する。`transport`は旧latent second momentをbasis overlapの二乗で
近似写像する。両条件とも全6ケースが`passed`となった。

| latent moment policy | validation loss mean±std | paired loss difference vs transport | improved seeds | persistent state | peak allocated | peak reserved | optimizer step mean±std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `reset` | `36.822 ± 1.284` | `-0.539 ± 0.110` | `3/3` | `46.41 MiB` | `2.964 GiB` | `4.896 GiB` | `61.765 ± 0.953 ms` |
| `transport` | `37.361 ± 1.189` | `-` | `-` | `46.41 MiB` | `2.964 GiB` | `4.896 GiB` | `51.584 ± 5.326 ms` |

seed別の`reset - transport` loss差は`-0.680/-0.528/-0.410`で、今回の条件ではresetが
一貫して改善した。state容量とpeak VRAMは同一であり、resetの品質差はメモリ削減の副作用
ではなく、refresh後に古い座標系のpreconditionerを持ち越さない効果として解釈できる。
一方、step時間はseed・GPU状態の揺らぎが大きく、今回の3 seedだけではresetが高速とは言えない。

これは、短期stress testでresetがtransportより悪化した過去結果を上書きしない。token budgetと
refresh回数が異なるため、refresh後の局所再適応が長期条件で有利になった可能性がある。この後、
診断付き短期runと診断なし速度runを同一の長期条件へ拡張し、既定policy変更は複数の長期seedで
再現するまで保留する。

この診断付きpaired run（rank=`16`・overlap=`0.99`・hard interval=`25`・TinyStories・
CUDA/BF16・3 seed・実効96 step）を実行した。reset/transportとも全3 seedが完走し、refresh
eventは各seedで3回だった。resetではevent後の`moment_step`が毎回`1`に戻り、latent
moment normは直前stepの約`0.024--0.097`へ低下した。transportでは`moment_step`が
`26/51/76`と継続し、moment normもrefresh前後で約`1.014--1.048`倍に留まった。

| latent moment policy | validation loss mean±std | paired loss difference vs transport | improved seeds | update norm mean | update norm variance | persistent state |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `reset` | `40.608 ± 0.309` | `-0.164 ± 0.092` | `3/3` | `0.485` | `0.0817` | `46.41 MiB` |
| `transport` | `40.772 ± 0.401` | `-` | `-` | `0.456` | `0.0905` | `46.41 MiB` |

refresh前後の周期validation lossは両policyとも継続的に低下し、resetだけが明確に速く
回復する証拠は得られなかった。一方、resetのlatent moment ageを局所化する挙動と、長期
診断なしrunで観測したloss改善は整合している。resetは「古いbasisの近似momentを持ち越さない」
ため、品質候補としてはtransportより有望だが、速度差と長期再現性は別途確認する。

診断なし速度比較（rank=`16`・overlap=`0.99`・hard interval=`50`・TinyStories・CUDA/BF16・
3 seed・実効300 step）も完了した。reset/transportとも全3 seedが完走し、次の結果になった。

| latent moment policy | validation loss mean±std | paired loss difference vs transport | optimizer step mean±std | paired step difference vs transport | persistent state | peak allocated | peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `reset` | `36.822 ± 1.572` | `-0.539 ± 0.135` | `59.905 ± 3.851 ms` | `+1.405 ± 5.738 ms` | `46.41 MiB` | `2.964 GiB` | `3.428 GiB` |
| `transport` | `37.361 ± 1.456` | `-` | `58.500 ± 2.125 ms` | `-` | `46.41 MiB` | `2.964 GiB` | `3.428 GiB` |

resetは3 seedすべてでlossを改善したが、step時間のpaired差は標準偏差より小さく、速度差は
実質同等と判定する。今回の測定ではrefresh policyによるstate容量・peak VRAMの差もない。
したがって、`AdamW-LRSF-LR`では`reset`を品質・座標系整合性を優先する実験候補、`transport`
を従来互換候補として扱う。どちらも既定値にはまだ変更しない。

##### `AdamW-LRSF-LR` の rank × refresh interval sweep

上記のreset/transport差がrankとrefresh頻度に依存するかを、rank=`8,16`・hard refresh
interval=`25,50`・overlap=`0.99`・TinyStories・CUDA/BF16・3 seed・実効300 stepで診断なし
比較した。

| rank | interval | latent moment policy | validation loss mean±std | paired loss difference reset-transport | improved reset | optimizer step mean±std | persistent state |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 25 | `reset` | `36.694 ± 2.452` | `-0.425 ± 0.380` | `3/3` | `61.63 ± 1.62 ms` | `23.25 MiB` |
| 8 | 25 | `transport` | `37.119 ± 2.818` | `-` | `-` | `58.93 ± 0.57 ms` | `23.25 MiB` |
| 8 | 50 | `reset` | `36.814 ± 2.432` | `-0.309 ± 0.461` | `2/3` | `59.95 ± 3.18 ms` | `23.25 MiB` |
| 8 | 50 | `transport` | `37.123 ± 2.816` | `-` | `-` | `57.39 ± 1.46 ms` | `23.25 MiB` |
| 16 | 25 | `reset` | `36.775 ± 1.515` | `-0.572 ± 0.393` | `3/3` | `58.32 ± 0.13 ms` | `46.41 MiB` |
| 16 | 25 | `transport` | `37.347 ± 1.459` | `-` | `-` | `58.58 ± 1.89 ms` | `46.41 MiB` |
| 16 | 50 | `reset` | `36.822 ± 1.572` | `-0.539 ± 0.135` | `3/3` | `56.87 ± 0.86 ms` | `46.41 MiB` |
| 16 | 50 | `transport` | `37.361 ± 1.456` | `-` | `-` | `58.89 ± 0.26 ms` | `46.41 MiB` |

4条件すべてでresetの平均lossがtransportより低く、rank=`16`では両intervalで3/3 seedが
改善した。rank=`8`・interval=`50`だけは改善seedが2/3で、resetの効果はrefresh頻度と
rankに依存する可能性がある。stateはrankに比例して`23.25/46.41 MiB`となり、policyでは
増えない。step時間はrank=`8`ではtransportが速く、rank=`16`ではinterval=`25`で同等、
interval=`50`でresetがやや速かったため、測定揺らぎを含む候補値として扱う。現時点では
既定policyを変更せず、resetとtransportのlatent moment age、周期validation loss、effective
update curvature、Schedule-Freeのtrain/hidden/eval trajectoryを診断付きrunで比較した。
診断付きstep時間は速度比較には用いない。

##### reset / transport trajectory診断の初回結果

上記の低負荷wrapperを、rank=`16`・overlap=`0.99`・hard refresh interval=`25`・診断snapshot
およびvalidation interval=`20`・TinyStories・CUDA/BF16・3 seed・実効96 stepで実行した。
実効step数が96となるため、snapshotはstep=`20,40,60,80`の4点である。両policyとも全3 seedが
完走し、`schedulefree_trajectory_curvature_status=passed`となった。

| latent moment policy | validation loss mean±std | train parameter roughness | hidden state roughness | eval parameter roughness | validation loss Δ² abs mean | persistent state |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `reset` | `40.608 ± 0.309` | `0.960` | `2.195` | `0.957` | `4.422` | `46.41 MiB` |
| `transport` | `40.772 ± 0.401` | `0.961` | `2.391` | `0.823` | `4.595` | `46.41 MiB` |

resetはvalidation lossでtransportをpairedに`-0.164 ± 0.092`改善し、3/3 seedで低かった。
また、resetではrefresh event後のlatent moment ageが毎回`1`へ戻り、transportでは`26/51/76`
まで継続した。hidden stateのroughnessはresetの方が小さかったが、eval parameterのroughnessは
transportの方が小さく、Schedule-Freeの内部hidden軌跡が滑らかであることが、そのまま評価軌跡の
平滑化を意味しない。validation lossの2階差分もresetが僅かに小さいが、4 snapshot・1 epochの
短期診断であり、resetがtrajectory smoothingを因果的に実現すると結論しない。長期windowでも
同じ指標を測定し、さらにfrozenとの比較およびrefresh event前後のrecovery proxyを確認した。

長期条件でも同じ診断を追加確認した。rank=`16`・overlap=`0.99`・hard refresh interval=`25`・
診断snapshotおよびvalidation interval=`20`・TinyStories・CUDA/BF16・3 seed・実効288 stepで、
snapshotは14点、refresh eventは各seedで11回だった。

| latent moment policy | validation loss mean±std | train parameter roughness | hidden state roughness | eval parameter roughness | validation loss Δ² abs mean | persistent state |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `reset` | `33.764 ± 0.468` | `0.547` | `1.871` | `0.612` | `0.942` | `46.41 MiB` |
| `transport` | `34.534 ± 0.454` | `0.845` | `2.501` | `0.831` | `0.964` | `46.41 MiB` |

resetのpaired loss差は`-0.769 ± 0.224`で、改善seedは`3/3`だった。短期runで見られた
eval roughnessのtransport優位は長期条件では再現せず、resetが3種類すべてのroughnessを
低下させた。ただし、roughnessは14点のparameter snapshotから得た診断指標であり、これだけで
loss改善の因果要因とは言えない。state容量とpeak VRAMは両policyで同じだったため、resetの
利点は追加メモリ削減ではなく、basis交換後のlatent momentを局所的に再初期化するtrajectory
制御にある可能性がある。

##### frozen / hard-reset / hard-transport refresh recovery比較

resetのloss改善がrefreshそのものの効果か、単なるtransport policy間の差かを分離するため、
`frozen`（refreshなし）、`hard-reset`、`hard-transport`をrank=`16`・overlap=`0.99`・hard
refresh interval=`25`・validation interval=`20`・TinyStories・CUDA/BF16・3 seed・実効288 stepで
比較した。各条件は14 validation snapshotと11 refresh eventを持つ。recovery proxyは各eventの
直前validationから最初のevent後validationまでのloss差であり、validation間隔のため通常の
学習進行を含む。

| policy | validation loss mean±std | paired loss difference vs frozen | improved seeds vs frozen | recovery proxy mean±std | moment age at refresh | persistent state | peak allocated / reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `frozen` | `34.519 ± 0.485` | `-` | `-` | `-` | `-` | `46.41 MiB` | `2.964 / 3.426 GiB` |
| `hard-reset` | `33.764 ± 0.468` | `-0.755 ± 0.204` | `3/3` | `-2.215 ± 3.244` | `1` | `46.41 MiB` | `2.964 / 3.428 GiB` |
| `hard-transport` | `34.534 ± 0.454` | `+0.015 ± 0.042` | `2/3` | `-2.166 ± 3.304` | `151 ± 80` | `46.41 MiB` | `2.964 / 3.428 GiB` |

hard-resetだけがfrozenに対して3/3 seedで改善し、hard-transportはfrozenと実質同等だった。
一方、recovery proxyはresetとtransportで近く、標準偏差も大きいため、refresh直後のvalidation
回復速度がloss改善の直接原因とは判断できない。roughnessはtrain/hidden/evalの順に
frozen=`0.742/2.256/0.825`、hard-reset=`0.547/1.871/0.612`、hard-transport=`0.845/2.501/0.831`
で、resetではfrozenに対しても全sourceで低下した。これは「latent momentをresetすること」が
長期trajectoryを制御する仮説を支持するが、interval・LR・rankを固定した1条件のため、既定policyの
変更根拠にはしない。event-localな短期runは長期品質や速度の結論には使わない。

event-aligned runはrank=`16`・overlap=`0.99`・hard refresh interval=`25`・validationおよび
snapshot interval=`5`・TinyStories・CUDA/BF16・3 seed・実効96 stepで完了した。全9ケースが
成功し、refresh eventはstep=`26,51,76`で観測された。

| policy | validation loss mean±std | paired loss difference vs frozen | improved seeds vs frozen | event 1 proxy | event 2 proxy | event 3 proxy | train/hidden/eval roughness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `frozen` | `40.806 ± 0.353` | `-` | `-` | `-` | `-` | `-` | `1.492/3.376/0.973` |
| `hard-reset` | `40.608 ± 0.309` | `-0.199 ± 0.050` | `3/3` | `-3.166 ± 0.369` | `-0.972 ± 0.026` | `-0.605 ± 0.038` | `0.997/2.285/0.896` |
| `hard-transport` | `40.772 ± 0.401` | `-0.035 ± 0.056` | `2/3` | `-3.287 ± 0.398` | `-0.954 ± 0.064` | `-0.556 ± 0.033` | `1.468/3.322/0.970` |

event proxyは各refresh eventについて「直前のvalidation lossから最初のevent後validation
lossまでの差」で、負値は通常の学習進行を含むloss低下を表す。最初のeventではtransportの
低下が大きく、2・3回目ではresetの低下が大きかった。したがって、resetの最終loss改善は
refresh直後の単発回復速度では説明できず、latent moment ageを局所化する効果が長期的に
累積する仮説が残る。これは1 epochの短期runである。

interval=`50`のevent-aligned screeningも完了した。rank=`16`・overlap=`0.99`・hard refresh
interval=`50`・validationおよびsnapshot interval=`5`・TinyStories・CUDA/BF16・3 seed・実効96
stepで、refresh eventはstep=`51`に各seed1回だった。

| policy | validation loss mean±std | paired loss difference vs frozen | improved seeds vs frozen | recovery proxy | train/hidden/eval roughness |
| --- | ---: | ---: | ---: | ---: | ---: |
| `frozen` | `40.806 ± 0.353` | `-` | `-` | `-` | `1.492/3.376/0.973` |
| `hard-reset` | `40.610 ± 0.415` | `-0.196 ± 0.063` | `3/3` | `-0.988 ± 0.070` | `1.163/2.778/0.986` |
| `hard-transport` | `40.780 ± 0.365` | `-0.026 ± 0.017` | `3/3` | `-0.944 ± 0.057` | `1.458/3.227/0.968` |

interval=`25`と同様に、hard-resetはfrozenより低いlossとなったが、差はinterval=`25`の
event-aligned結果（`-0.199`）と同程度で、refresh頻度による明確な差は見られなかった。
recovery proxyもresetとtransportで近く、resetはtrain/hidden roughnessを下げた一方、eval
roughnessはfrozen/transportと同程度だった。1 epochでeventが1回のみのため、resetの累積効果
とinterval依存性を判断するため、interval=`50`の3 epoch長期runも追加した。

##### interval=50 長期refresh recovery比較

同じ設定で3 epoch・実効288 stepまで延長した。interval=`50`では各seedで5回のrefresh eventが
発生した。

| policy | validation loss mean±std | paired loss difference vs frozen | improved seeds vs frozen | recovery proxy mean±std | moment age at refresh | train/hidden/eval roughness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `frozen` | `34.519 ± 0.485` | `-` | `-` | `-` | `-` | `0.032/1.721/0.027` |
| `hard-reset` | `33.706 ± 0.313` | `-0.813 ± 0.179` | `3/3` | `-1.478 ± 1.485` | `1` | `0.118/8.708/0.114` |
| `hard-transport` | `34.510 ± 0.448` | `-0.009 ± 0.038` | `1/3` | `-1.405 ± 1.463` | `151 ± 73` | `0.032/1.727/0.027` |

hard-resetのloss優位はinterval=`25`の長期runと同じく3/3 seedで再現し、hard-transportはfrozenと
ほぼ同等だった。一方、recovery proxyはresetとtransportで近く、refresh直後の回復だけでは
resetの累積loss改善を説明できない。さらに、interval=`25`長期runではresetのroughnessが低下
したのに対し、interval=`50`ではresetのtrain/hidden/eval roughnessがすべて増加した。この
反転はrefresh間隔、またはseed・短いsnapshot系列への依存を示すため、LRTDOの「resetが軌跡を
平滑化する」という仮説は未確定とする。現時点では、resetはtrajectory smoothingではなく、
latent momentを再初期化することで別の軌跡を選ぶ品質制御として扱う。既定policyは変更しない。

### Confidenceとprojected-gradientの比較

<a id="projected-gradient-ema"></a>
#### 低rank projected-gradient EMAの比較基準

Schedule-Freeの効果と「低rank射影＋EMA」だけの効果を分離するため、
`AdamW-LR-EMA`を追加した。このoptimizerは`z`、Schedule-Free delta、Adam second momentを
持たず、固定orthonormal basis上のgradient EMAをbias correctionしてdecodeし、decoupled
weight decay付きで直接更新する。TinyStories・CUDA/BF16・rank=`8`・seed=`0`・実効100 stepの
未調整runでは、次の結果になった。

| optimizer | validation loss | persistent state | peak allocated | step時間 |
| --- | ---: | ---: | ---: | ---: |
| AdamW | 106.508 | 726.3 MB | 3.861 GB | 19.41 ms |
| AdamW-SF | 33.755 | 726.3 MB | 4.380 GB | 68.41 ms |
| AdamW-LRSF | 41.884 | 376.1 MB | 3.511 GB | 35.29 ms |
| AdamW-LR-EMA | 359.230 | 13.0 MB | 3.147 GB | 36.91 ms |

この結果は`AdamW-LR-EMA`が約`0.018x`のAdamW stateで動作する一方、同じ
learning rate=`3e-4`ではvalidation lossが大きく悪化することを示す。したがって、
低rank trajectory stateだけでAdamW-SF相当の品質が得られるという根拠にはならない。
ただしseed=`0`のみで、adaptive second momentを除去した影響とlearning rate mismatchを
分離していないため、失敗の最終判定には使わない。次は`ema_beta`とlearning rateを調整し、
「EMAによる時間平滑化」自体の寄与を測定する。

<a id="innovation-variance-confidence"></a>
#### innovation variance confidenceの初回結果

`AdamW-LR-EMA`にlatent innovation variance `c_t`を加えた
`AdamW-LR-EMA-Conf`を、TinyStories・CUDA/BF16・rank=`8`・seed=`0`・実効100 stepで
比較した。設定は`beta_m=0.9`、`beta_c=0.99`、`alpha=1e-3`である。

| optimizer | validation loss | persistent state | peak allocated | step時間 |
| --- | ---: | ---: | ---: | ---: |
| AdamW-LR-EMA | 359.230 | 13.0 MB | 3.147 GB | 36.16 ms |
| AdamW-LR-EMA-Conf | 35.554 | 24.4 MB | 3.159 GB | 53.26 ms |

同じ初期条件の別比較で得られた`AdamW-SF`のvalidation loss=`33.755`と比べると、
confidence variantは約`1.8%`高いだけであり、EMA単独の`359.230`から大幅に改善した。
一方、`c_t`を追加することでstateは約`13.0→24.4 MB`、step時間は約`36.2→53.3 ms`へ
増加した。それでもfull AdamW-SF state=`726.3 MB`に対して約`3.4%`であり、低rank空間内
の適応正規化が品質回復に寄与する可能性を示す。

これはseed=`0`のみの初回結果であり、`AdamW-SF`との完全なpaired比較ではない。続く3 seed比較では
`AdamW-SF`、`AdamW-LRSF`、`AdamW-LR-EMA`、`AdamW-LR-EMA-Conf`を同一runに入れ、`beta_c`と
`alpha`の感度、confidence値、更新normを測定した。

この比較は次のwrapperで縮小条件から再現できる。各セルは独立したJSONへ保存され、既存
セルはskipされる。`--confidence-betas`と`--confidence-alphas`で感度グリッドを絞れば、
GPU時間を抑えて候補を選別できる。

```bash
verify/launchers/run_text_lm_lr_ema_confidence_sweep.sh \
  --device cuda --dtype bf16 \
  --rank 8 --seeds 0,1,2 \
  --confidence-betas 0.95,0.99 \
  --confidence-alphas 0.0001,0.001,0.01 \
  --output-dir output/text-lm-lr-ema-confidence-sweep
```

出力の`confidence_diagnostics`はconfidence variantだけに現れ、各snapshotには
`confidence_mean`、`confidence_std`、`innovation_rms_mean`、
`normalized_update_rms_mean`を含む。state-rankを併用する場合は低rank latent `m/c`と
projectionのgeometryも取得できるが、SVDとdevice-to-host scalar取得を伴うため、step時間
の比較は診断なしrunで別途行う。

##### 3 seed sensitivity result

上記wrapperをTinyStories・CUDA/BF16・rank=`8`・実効32 step・3 seedで実行した。4つの
optimizerは同じseedごとに同じ初期parameterとbatch orderを共有している。表のstateは
diagnostic stateではなくpersistent optimizer stateである。

| setting | AdamW-SF loss | AdamW-LRSF loss | AdamW-LR-EMA loss | AdamW-LR-EMA-Conf loss | confidence state | 読み取り |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `beta_c=0.95, alpha=1e-4` | `46.703 ± 1.089` | `55.573 ± 0.823` | `355.262 ± 3.674` | `51.123 ± 0.240` | `23.25 MiB` | confidenceでEMA単独の退行を回復 |
| `beta_c=0.95, alpha=1e-3` | `46.703 ± 1.089` | `55.573 ± 0.823` | `355.262 ± 3.674` | `51.120 ± 0.224` | `23.25 MiB` | alpha差は小さい |
| `beta_c=0.95, alpha=1e-2` | `46.703 ± 1.089` | `55.573 ± 0.823` | `355.262 ± 3.674` | `51.526 ± 0.372` | `23.25 MiB` | 大きいfloorはやや悪化 |
| `beta_c=0.99, alpha=1e-4` | `46.703 ± 1.089` | `55.573 ± 0.823` | `355.262 ± 3.674` | `50.842 ± 0.341` | `23.25 MiB` | betaを高めると改善傾向 |
| `beta_c=0.99, alpha=1e-3` | `46.703 ± 1.089` | `55.573 ± 0.823` | `355.262 ± 3.674` | `50.849 ± 0.343` | `23.25 MiB` | 次段階の暫定候補 |
| `beta_c=0.99, alpha=1e-2` | `46.703 ± 1.089` | `55.573 ± 0.823` | `355.262 ± 3.674` | `50.752 ± 0.190` | `23.25 MiB` | 最小lossだがalpha差は小さい |

confidence variantは`AdamW-LRSF`より約`4.0〜4.8` loss良い一方、`AdamW-SF`より
約`4.0〜4.8` loss悪い。最終snapshotのconfidence平均は全条件で約`0.01〜0.02`に
留まり、低rank座標の多くは`m̂²`よりinnovation varianceが大きい。したがって、次の
統合版では`beta_c=0.99, alpha=1e-3`を再現性重視の初期値とし、`alpha=1e-2`は候補として
残す。ただしこの差は縮小runでの傾向であり、長期runでの採用判断は保留する。

<a id="confidence-lrsf-prototype"></a>
#### 初期prototype: confidence付きLRSFの統合

`AdamW-LR-EMA-Conf`のconfidence stateとLRSFのSchedule-Free deltaを統合し、rank=`8`の
初期prototype `AdamW-LR-EMA-Conf-LRSF`を実装した。`m/c`は投影勾配の信頼度、`delta`は
train/eval軌跡の差分を表すため、品質差の原因を分けて調べられるよう各stateに独立した
projectionを使った。現在のstate・fallback・refresh契約は[optimizer一覧](../optimizers.md)を参照。

評価ではAdamW-SFを品質基準とし、validation loss、persistent state、step時間を比較した。
初回結果は次節に記録する。短期比較で統合版のloss悪化が見られたため、既定optimizerには
採用せず、APOLLO系との比較でconfidence正規化とSchedule-Free deltaの寄与を切り分けた。

##### 初回GPU比較結果

上記条件（rank=`8`、`beta_c=0.99`、`alpha=1e-3`、TinyStories、CUDA/BF16、3 seed、
実効32 step）で、`AdamW-SF`、`AdamW-LRSF`、`AdamW-LR-EMA-Conf`、
`AdamW-LR-EMA-Conf-LRSF`をpaired比較した。

| optimizer | validation loss mean±std | persistent state | host step seconds |
|---|---:|---:|---:|
| AdamW-SF | `46.703 ± 1.089` | `692.62 MiB` | `58.65 ms` |
| AdamW-LRSF | `55.573 ± 0.823` | `358.70 MiB` | `38.98 ms` |
| AdamW-LR-EMA-Conf | `50.849 ± 0.343` | `23.25 MiB` | `56.18 ms` |
| AdamW-LR-EMA-Conf-LRSF | `85.118 ± 5.759` | `35.59 MiB` | `64.04 ms` |

統合版のstate構成は、confidence単体にLRSF deltaを加えたサイズとなり、低rank stateのみで
成立した。一方、同一LR・Schedule-Free設定ではvalidation lossが大きく悪化したため、
現段階で「confidenceとLRSFを単純に足せばよい」とは言えない。LRSF deltaがconfidenceで
正規化された`u_t`を蓄積すること、またはSchedule-Freeの`sf_beta1`/LRがこの更新スケールに
適合していないことが候補である。このrunは初期32 stepのscreeningであり、長期性能の結論
ではないが、次の優先課題を「統合stateの完成」から「悪化要因の分離」に変更する根拠になる。

なお、初回実行ではBF16 fallbackの`lerp_` dtype不一致が検出された。補間時だけ`z`をFP32
へcastする修正と、統合版の`lr_ema_step`を読むconfidence診断修正を行い、修正版の全ケース
が`passed`になった。これらはpersistent stateのdtypeやサイズを変更しない。

### APOLLO比較とsensitivity

<a id="apollo-confidence-comparison"></a>
#### APOLLO比較とAPOLLO型confidence variant

`AdamW-LR-EMA-Conf-LRSF`の品質悪化がconfidence正規化そのものによるのか、Schedule-Free
deltaの追加によるのかを分けるため、既存の`APOLLO`を比較対象へ加える。さらに
`APOLLO-Conf`を実装した。これはSchedule-Free deltaを持たず、APOLLOの更新経路を維持した
まま、latent second stateの意味だけを次のように置き換えるvariantである。

```text
g_lr       = project(P, gradient)
m_t        = EMA(g_lr)
r_t        = g_lr - m_{t-1}
c_t        = EMA(r_t^2)
u_lr       = m_hat_t / sqrt(c_hat_t + alpha * m_hat_t^2)
scaling    = channel_norm(u_lr) / channel_norm(g_lr)
update     = scaling * full_gradient
```

stateはAPOLLOと同じ`projection`、latent `exp_avg`、latent `exp_avg_sq`で、
`exp_avg_sq`をinnovation varianceとして利用する。`APOLLO`との差は二次統計の定義、
`AdamW-LR-EMA-Conf`との差はfull-gradientへのchannel-wise scalingかdecoded updateか、
`AdamW-LR-EMA-Conf-LRSF`との差はSchedule-Free deltaの有無となる。

同じTinyStories・rank・LR・seedを使い、`AdamW-SF`、`AdamW-LRSF`、`AdamW-LR-EMA-Conf`、
`AdamW-LR-EMA-Conf-LRSF`、`APOLLO`、`APOLLO-Conf`の6-way縮小比較を実施した。state bytes、
step時間、更新normも記録し、次節で比較結果と更新normの診断を示す。

##### APOLLO比較の初回結果

上記の6-way比較を、TinyStories・CUDA/BF16・rank=`8`・learning rate=`3e-4`・
seed=`0,1,2`・実効32 stepで実行した。confidenceの診断と更新normを有効にしたscreening
のため、step時間は最終的な速度比較ではなく、state bytesは永続optimizer stateとして
読む。

| optimizer | validation loss mean±std | persistent state | host step seconds mean±std | update norm mean | update norm variance |
| --- | ---: | ---: | ---: | ---: | ---: |
| AdamW-SF | `46.703 ± 1.089` | `692.62 MiB` | `50.72 ± 2.15 ms` | `0.571` | `0.133` |
| AdamW-LRSF | `55.573 ± 0.823` | `358.70 MiB` | `31.66 ± 11.10 ms` | `0.463` | `0.129` |
| AdamW-LR-EMA-Conf | `50.849 ± 0.343` | `23.25 MiB` | `44.55 ± 5.78 ms` | `1.037` | `0.156` |
| AdamW-LR-EMA-Conf-LRSF | `85.118 ± 5.759` | `35.59 MiB` | `53.74 ± 1.27 ms` | `0.406` | `0.126` |
| APOLLO | `341.179 ± 3.651` | `23.25 MiB` | `49.76 ± 2.13 ms` | `0.0178` | `0.00339` |
| APOLLO-Conf | `340.890 ± 3.691` | `23.25 MiB` | `51.69 ± 0.89 ms` | `0.0181` | `0.00349` |

APOLLO-ConfはAPOLLOに対して3 seedすべてで改善し、平均差は`-0.289`だった。しかし、
改善幅は小さく、両者のstateは`24,383,302 bytes`で完全に同じである。したがって、
今回のrunでは「innovation varianceを使っても追加メモリなしで動く」ことは確認できたが、
confidence variantの品質優位性はまだ示せない。

重要な診断は更新normである。APOLLO系は最初の数step後に更新normが約`0.0026〜0.0037`
へ落ち、平均でも`0.018`に留まった。一方、`AdamW-LR-EMA-Conf`は平均`1.037`である。
この更新スケールの縮退が、APOLLO/APOLLO-Confのvalidation loss=`340`前後を説明する有力な
候補であり、現段階でAPOLLOの低品質をconfidence設計の失敗とは解釈しない。

この更新norm縮退を切り分けるため、APOLLOとAPOLLO-Confのlearning rate、`apollo_scale`、
norm-growth limiter有効/無効をpaired sweepで比較した。候補条件では診断なしのstep時間とpeak VRAMも
後続runで測定したが、APOLLO-Confの既定値採用や`APOLLO-CAME-LRSF`との品質比較は保留した。

##### 更新norm縮退の切り分け結果

`verify/launchers/run_text_lm_apollo_scale_sweep.sh`を、TinyStories・CUDA/BF16・rank=`8`・
seed=`0,1,2`・train/eval token=`4096/512`・実効10 stepで実行した。全12セル、各6ケースが
`passed`となった。

| optimizer | learning rate | scale | limiter | validation loss mean±std | update norm mean±std |
| --- | ---: | ---: | --- | ---: | ---: |
| APOLLO | `3e-3` | `0.5` | off | `64.264 ± 3.142` | `1.318 ± 0.014` |
| APOLLO-Conf | `3e-3` | `0.5` | off | `65.017 ± 1.480` | `1.358 ± 0.019` |
| APOLLO | `3e-3` | `1.0` | off | `65.633 ± 5.037` | `1.788 ± 0.018` |
| APOLLO-Conf | `3e-3` | `1.0` | off | `67.675 ± 2.191` | `1.838 ± 0.023` |
| APOLLO | `1e-3` | `1.0` | off | `79.721 ± 3.105` | `0.703 ± 0.013` |
| APOLLO-Conf | `1e-3` | `1.0` | off | `75.670 ± 3.458` | `0.744 ± 0.011` |

limiterを有効にした同条件では、`3e-3/0.5`のAPOLLOが`114.915`、`3e-3/1.0`が`94.519`、
`1e-3/1.0`が`224.540`となった。`3e-4/1.0`では`337.590`まで悪化している。このため、
前回の`lr=3e-4`比較で観測した更新norm=`0.018`は、APOLLOのstate容量不足ではなく、
norm-growth limiterが初期更新を過度に抑制した結果である可能性が高い。

scaleは単純な単調軸ではない。`3e-3`ではscale=`0.5`がAPOLLO/APOLLO-Confとも最良だが、
APOLLOの更新normはscale=`1.0`の方が大きい。したがって、更新normを大きくすること自体を
目的にせず、parameter lossと更新normの両方で候補を選ぶ。なお、全条件でstateは約`23.25 MiB`
で、scaleやlimiterによるstate増加はない。

今回の実験から、APOLLO-ConfがAPOLLOを一貫して改善するとは言えない。後続runでは候補の
`APOLLO lr=3e-3/scale=0.5/limiter=off`、`APOLLO-Conf lr=3e-3/scale=0.5/limiter=off`、
およびconfidence側で比較的良かった`1e-3/scale=1.0/limiter=off`を、実効32〜100 stepへ
延長する。診断なしrunで速度・peak VRAMを測り、長期条件でlimiter無効が安全か確認するまで
既定値は変更しない。

##### APOLLO候補の診断なし延長結果

候補sweepを、TinyStories・CUDA/BF16・rank=`8`・seed=`0,1,2`・train/eval token=`16384/1024`・
実効32 step・`limiter=off`で再実行した。診断を無効にしたため、ここでのstep時間とpeak VRAMは
前回の診断付きsweepより速度比較に適している。全8セル、各6ケースが`passed`となった。

| optimizer | learning rate | scale | validation loss mean±std | persistent state | peak allocated | peak reserved | host step seconds mean±std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| APOLLO | `3e-3` | `0.5` | `48.500 ± 0.280` | `23.25 MiB` | `2.94 GiB` | `3.42 GiB` | `44.17 ± 3.59 ms` |
| APOLLO-Conf | `3e-3` | `0.5` | `49.189 ± 1.866` | `23.25 MiB` | `2.94 GiB` | `3.42 GiB` | `50.41 ± 0.12 ms` |
| APOLLO | `3e-3` | `1.0` | `48.504 ± 2.020` | `23.25 MiB` | `2.94 GiB` | `3.42 GiB` | `52.00 ± 0.30 ms` |
| APOLLO-Conf | `3e-3` | `1.0` | `48.994 ± 0.456` | `23.25 MiB` | `2.94 GiB` | `3.42 GiB` | `50.23 ± 3.16 ms` |
| APOLLO | `1e-3` | `1.0` | `52.281 ± 0.945` | `23.25 MiB` | `2.94 GiB` | `3.42 GiB` | `46.74 ± 1.90 ms` |
| APOLLO-Conf | `1e-3` | `1.0` | `51.431 ± 0.795` | `23.25 MiB` | `2.94 GiB` | `3.42 GiB` | `47.96 ± 1.85 ms` |

APOLLOの最良条件は短期sweepと同じ`3e-3/0.5`で、前回の同条件AdamW-SF（validation loss
`46.703 ± 1.089`）との差は約`1.80`まで縮まった。`3e-3/1.0`との差は平均`0.004`であり、
scale=`0.5`を優位と確定するほどではない。APOLLO-Confは`1e-3/1.0`でAPOLLOより平均`0.850`
改善したが、`3e-3`ではAPOLLOより平均`0.489〜0.689`悪く、confidenceの一貫した優位性は
確認できない。stateは全条件で`24,382,784 bytes`（約`23.25 MiB`）だった。

続く比較では`3e-3/0.5`と`3e-3/1.0`を実効100 stepへ延長し、さらに同一token budgetで
AdamW系との診断なし比較を行った。今回の32 step結果だけでは、APOLLO-Confまたはlimiter無効を
既定化しない。

##### APOLLO候補の実効100 step結果

候補sweepを、TinyStories・CUDA/BF16・rank=`8`・seed=`0,1,2`・train/eval token=`51200/1024`・
実効100 step・`limiter=off`・診断なしで実行した。全4セル、各6ケースが`passed`となった。

| optimizer | scale | validation loss mean±std | persistent state | peak allocated | peak reserved | host step seconds mean±std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| APOLLO | `1.0` | `28.790 ± 0.406` | `23.25 MiB` | `2.94 GiB` | `3.42 GiB` | `46.71 ± 2.23 ms` |
| APOLLO-Conf | `0.5` | `32.724 ± 0.864` | `23.25 MiB` | `2.94 GiB` | `3.42 GiB` | `51.48 ± 0.70 ms` |
| APOLLO | `0.5` | `34.377 ± 0.492` | `23.25 MiB` | `2.94 GiB` | `3.42 GiB` | `47.26 ± 0.64 ms` |
| APOLLO-Conf | `1.0` | `29.229 ± 1.278` | `23.25 MiB` | `2.94 GiB` | `3.42 GiB` | `50.26 ± 1.49 ms` |

APOLLOではscale=`1.0`がscale=`0.5`より`5.59` loss良く、100 stepでは短期32 stepの順位が
再現しなかった。APOLLO-Confはscale=`0.5`でAPOLLO比`1.65`改善したが、scale=`1.0`では
`0.44`悪化し、confidenceの効果はscale依存だった。APOLLO-Confのstep時間はAPOLLOより
概ね`3〜10%`長い一方、stateとpeak VRAMは同じである。

この結果はAPOLLOの更新スケール縮退が長期条件で解消されることを支持するが、AdamW-SFや
AdamW-LRSFとは同一runの比較ではない。続く診断なしbaselineで、同じtoken budget・seed・モデルを
使い、品質・state・peak VRAM・step時間を比較した。既定値は変更しない。

##### AdamW系とAPOLLO系の診断なしfair comparison

上記の比較を、TinyStories・Qwen tokenizer・CUDA/BF16・rank=`8`・seed=`0,1,2`・
train/eval token=`51200/1024`・実効100 step・診断なしで実行した。AdamW系は候補LR=`3e-4`、
APOLLO系は更新norm縮退を避ける候補LR=`3e-3`、`limiter=off`、`scale=1.0`を使用した。
全4 optimizer、12ケースが`passed`となった。

| optimizer | learning rate | validation loss mean±std | persistent state | peak allocated | peak reserved | host step time mean±std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW-SF | `3e-4` | `36.081 ± 1.454` | `692.62 MiB` | `4.079 GiB` | `4.896 GiB` | `76.79 ± 3.04 ms` |
| AdamW-LRSF | `3e-4` | `43.688 ± 0.873` | `358.70 MiB` | `3.269 GiB` | `4.896 GiB` | `48.42 ± 15.28 ms` |
| APOLLO | `3e-3` | `29.730 ± 0.418` | `23.25 MiB` | `2.942 GiB` | `3.418 GiB` | `47.47 ± 2.86 ms` |
| APOLLO-Conf | `3e-3` | `30.377 ± 0.866` | `23.25 MiB` | `2.942 GiB` | `3.418 GiB` | `49.93 ± 0.69 ms` |

今回の候補条件ではAPOLLOが最良loss、最小state、最小peak VRAMとなった。APOLLO-Confは
APOLLOより平均`0.646` loss悪化し、step時間は約`5.2%`増加した。したがって、現時点で
APOLLO-Confを既定化する根拠はない。一方、APOLLO系とAdamW系でLRが異なるため、lossの
絶対値によるoptimizer本体の優劣は確定できない。AdamW系の同一LR候補を`3e-3`にも揃え、
APOLLO系を`3e-4`にも揃えるmatched-LR比較を追加で行う。stateとpeak VRAMの差は、今回の
条件でも実装上のメモリ差として利用できる。続くmatched-LR比較では、AdamW系とAPOLLO系の候補LRを
`3e-3`および`3e-4`に揃えて評価した。

matched-LRのうちLR=`3e-3`を、同じTinyStories・Qwen tokenizer・CUDA/BF16・rank=`8`・
seed=`0,1,2`・train/eval token=`51200/1024`・実効100 step・診断なしで実行した。全4 optimizer、
12ケースが`passed`となった。

| optimizer | validation loss mean±std | persistent state | peak allocated | peak reserved | host step time mean±std |
| --- | ---: | ---: | ---: | ---: | ---: |
| AdamW-SF | `15.657 ± 1.304` | `692.62 MiB` | `4.079 GiB` | `4.896 GiB` | `72.27 ± 2.31 ms` |
| AdamW-LRSF | `34.701 ± 1.396` | `358.70 MiB` | `3.269 GiB` | `4.896 GiB` | `39.45 ± 2.92 ms` |
| APOLLO | `29.730 ± 0.418` | `23.25 MiB` | `2.942 GiB` | `3.418 GiB` | `46.41 ± 3.13 ms` |
| APOLLO-Conf | `30.377 ± 0.866` | `23.25 MiB` | `2.942 GiB` | `3.418 GiB` | `50.30 ± 2.65 ms` |

LR=`3e-3`ではAdamW-SFがloss最良、APOLLOがstateとpeak VRAM最小、AdamW-LRSFがstep時間
最短だった。APOLLO-ConfはAPOLLOより平均`0.646` loss悪化し、step時間も約`8.4%`長かった。
APOLLOのメモリ優位は確認できるが、品質とのトレードオフは明確である。LR=`3e-4`の
matched比較は未実施のため、LR差に依存しない品質傾向はまだ確定しない。

続けてLR=`3e-4`でも同じ条件を実行し、全4 optimizer、12ケースが`passed`となった。

| learning rate | optimizer | validation loss mean±std | persistent state | peak allocated | peak reserved | host step time mean±std |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `3e-4` | AdamW-SF | `36.081 ± 1.454` | `692.62 MiB` | `4.079 GiB` | `4.896 GiB` | `70.77 ± 1.84 ms` |
| `3e-4` | AdamW-LRSF | `43.688 ± 0.873` | `358.70 MiB` | `3.269 GiB` | `4.896 GiB` | `40.62 ± 2.58 ms` |
| `3e-4` | APOLLO | `52.606 ± 2.568` | `23.25 MiB` | `2.942 GiB` | `3.418 GiB` | `45.74 ± 1.83 ms` |
| `3e-4` | APOLLO-Conf | `49.147 ± 1.244` | `23.25 MiB` | `2.942 GiB` | `3.418 GiB` | `51.51 ± 2.70 ms` |

同一LRで比較すると、AdamW-SFはLR=`3e-3`と`3e-4`の両方で最良lossだった。一方、
APOLLOはLR=`3e-3`で`29.730`まで改善し、AdamW-LRSFを上回ったが、AdamW-SFには届かなかった。
APOLLO-ConfはLR=`3e-4`ではAPOLLOより`3.459`改善したが、LR=`3e-3`では`0.646`悪化し、
confidenceの効果はLR依存だった。APOLLO系のstateは両LRで`23.25 MiB`、peak allocated/reservedは
`2.942/3.418 GiB`で変わらない。step時間も同一optimizer内で大きく変わらず、品質差は主にLRと
更新式によるものと解釈する。

この結果により、候補LRを分けた実用比較と両LRのmatched比較が揃った。ただし、これはrank=`8`・
実効100 stepの短期評価であり、長期学習やrank感度を確定するものではない。後続の縮小sweepでは、
AdamW-SFを品質基準、APOLLOをstate効率基準としてrank・scale・confidence設定を比較した。

##### APOLLO rank・scale縮小sweep

APOLLO/APOLLO-Confをrank=`4,8,16`、LR=`3e-4,3e-3`、scale=`0.5,1.0`、
`limiter=off`で、TinyStories・CUDA/BF16・seed=`0,1,2`・実効32 step・診断なしで比較した。
全24セルが`passed`となった。

| rank | learning rate | scale | APOLLO loss | APOLLO-Conf loss | state |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | `3e-3` | `0.5` | `49.158 ± 0.306` | `48.742 ± 0.712` | `11.68 MiB` |
| 4 | `3e-3` | `1.0` | `48.004 ± 1.747` | `48.404 ± 2.313` | `11.68 MiB` |
| 8 | `3e-3` | `0.5` | `48.500 ± 0.280` | `49.189 ± 1.866` | `23.25 MiB` |
| 8 | `3e-3` | `1.0` | `48.504 ± 2.020` | `48.994 ± 0.456` | `23.25 MiB` |
| 16 | `3e-3` | `0.5` | `47.719 ± 0.999` | `47.642 ± 1.487` | `46.41 MiB` |
| 16 | `3e-3` | `1.0` | `50.648 ± 1.549` | `49.010 ± 3.536` | `46.41 MiB` |
| 4 | `3e-4` | `1.0` | `224.597 ± 8.783` | `170.993 ± 8.294` | `11.68 MiB` |
| 8 | `3e-4` | `1.0` | `147.980 ± 16.920` | `94.053 ± 7.112` | `23.25 MiB` |
| 16 | `3e-4` | `1.0` | `83.992 ± 4.102` | `67.563 ± 2.669` | `46.41 MiB` |

短期ではLR=`3e-3`・rank=`16`・scale=`0.5`が最良で、APOLLO-Confが僅かにAPOLLOを上回った。
一方、stateはrank=`4`で`11.68 MiB`、rank=`8`で`23.25 MiB`、rank=`16`で`46.41 MiB`とほぼ
rankに比例する。メモリ効率と品質のバランスではrank=`4`・LR=`3e-3`・scale=`1.0`が候補であり、
最高短期lossを優先する場合はrank=`16`・scale=`0.5`が候補となる。LR=`3e-4`では全rankで
lossが大きく、現行のAPOLLO系には低すぎる可能性がある。実効32 stepの短期結果を受け、
rank=`4`/`16`の候補を実効100 stepへ延長した。

##### APOLLO rank候補の実効100 step結果

短期候補を、LR=`3e-3`・limiter無効・TinyStories・CUDA/BF16・seed=`0,1,2`・
train/eval token=`51200/1024`・診断なし・実効100 stepへ延長した。

| rank | scale | optimizer | validation loss mean±std | persistent state | peak allocated | peak reserved | host step time mean±std |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | `1.0` | APOLLO | `30.950 ± 0.728` | `11.68 MiB` | `2.931 GiB` | `3.414 GiB` | `37.48 ± 0.70 ms` |
| 4 | `1.0` | APOLLO-Conf | `30.412 ± 1.009` | `11.68 MiB` | `2.931 GiB` | `3.414 GiB` | `39.95 ± 0.90 ms` |
| 16 | `0.5` | APOLLO | `33.458 ± 0.729` | `46.41 MiB` | `2.964 GiB` | `3.426 GiB` | `45.81 ± 1.25 ms` |
| 16 | `0.5` | APOLLO-Conf | `32.197 ± 0.665` | `46.41 MiB` | `2.964 GiB` | `3.426 GiB` | `50.66 ± 0.47 ms` |

実効100 stepではrank=`4`がAPOLLO/APOLLO-Confともrank=`16`よりlossが良く、stateは約75%
少なかった。rank=`4`・scale=`1.0`のAPOLLO-Confがloss=`30.412`で最良、APOLLOも`30.950`
だった。rank=`4`はrank=`16`よりpeak allocatedが約`33 MiB`、step時間が約`18%`短い。
APOLLO-Confはrank=`4`でAPOLLOより`0.538`改善したが、step時間は約`6.6%`長い。このため、
現時点の実用候補をrank=`4`・LR=`3e-3`・scale=`1.0`へ絞り込む。ただし、AdamW-SFの同条件
loss=`15.657`には届いておらず、既定値変更ではなくstate効率を重視する実験候補として扱う。

<a id="apollo-conf-sensitivity"></a>
#### APOLLO-Conf confidence beta/alpha sensitivity

APOLLO-Confの候補をrank=`4`・LR=`3e-3`・scale=`1.0`・limiter無効・TinyStories・CUDA/BF16・
seed=`0,1,2`・train/eval token=`16384/1024`・実効32 stepで比較した。全6セル、18ケースが
`passed`となった。各セルでAPOLLOをpaired baselineとして再実行し、APOLLO-Confとの差を計測した。

| confidence beta | alpha | APOLLO-Conf loss mean±std | paired loss Δ | improved seeds | step Δ vs APOLLO |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.95` | `1e-4` | `46.181 ± 0.638` | `-1.113 ± 1.267` | `3/3` | `+3.411 ms` |
| `0.95` | `1e-3` | `46.235 ± 0.548` | `-1.059 ± 1.469` | `3/3` | `+3.909 ms` |
| `0.95` | `1e-2` | `47.074 ± 2.028` | `-0.220 ± 0.378` | `2/3` | `+2.472 ms` |
| `0.99` | `1e-4` | `47.702 ± 2.327` | `+0.409 ± 2.618` | `2/3` | `+0.891 ms` |
| `0.99` | `1e-3` | `47.368 ± 2.137` | `+0.074 ± 1.241` | `2/3` | `+3.798 ms` |
| `0.99` | `1e-2` | `46.620 ± 0.477` | `-0.673 ± 1.462` | `2/3` | `+3.304 ms` |

短期条件ではbeta=`0.95`が安定して改善し、alpha=`1e-4`が最良、alpha=`1e-3`がほぼ同等だった。
beta=`0.99`ではalphaにより改善と悪化が入れ替わり、confidenceの効果はbeta/alpha依存である。
全条件でstateは`11.68 MiB`で変わらず、APOLLO-Confのstep時間はAPOLLOよりおおむね数ms長い。
したがって、延長候補をbeta=`0.95`・alpha=`1e-4,1e-3`に絞り、100 stepで再現性を確認した。

beta=`0.95`・alpha=`1e-4,1e-3`を、rank=`4`・LR=`3e-3`・scale=`1.0`・limiter無効・
TinyStories・CUDA/BF16・seed=`0,1,2`・train/eval token=`51200/1024`・実効100 stepへ延長した。
両条件とも6ケースが`passed`となった。

| beta | alpha | APOLLO loss mean±std | APOLLO-Conf loss mean±std | paired loss Δ | improved seeds | state | paired step Δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.95` | `1e-4` | `30.950 ± 0.728` | `31.062 ± 1.084` | `+0.112 ± 0.434` | `1/3` | `11.68 MiB` | `+4.718 ms` |
| `0.95` | `1e-3` | `30.950 ± 0.728` | `31.063 ± 0.194` | `+0.113 ± 0.683` | `1/3` | `11.68 MiB` | `+3.159 ms` |

短期32 stepで観測したbeta=`0.95`の改善は100 stepでは維持されず、alpha=`1e-4,1e-3`とも
APOLLOを平均約`0.112` loss下回った。APOLLO-Confのstep時間は約`3.2〜4.7 ms`長く、stateは
同じだった。このためconfidence正規化は有望な補助機構ではあるが、現時点でAPOLLOの既定更新へ
組み込む根拠はない。その後はconfidence単体の調整からrank=`4` APOLLOのLR/scale設計へ検討対象を
移し、短期・100 step比較を行った。

<a id="apollo-rank-scale-sweep"></a>
#### APOLLO rank=4 LR・scale縮小sweep

rank=`4`のAPOLLO/APOLLO-ConfをLR=`1e-3,2e-3,3e-3,5e-3`、scale=`0.75,1.0`、
`limiter=off`、TinyStories・CUDA/BF16・seed=`0,1,2`・train/eval token=`16384/1024`・
実効32 step・診断なしで比較した。全16 optimizer条件が`passed`となった。

| learning rate | scale | APOLLO loss | APOLLO-Conf loss |
| ---: | ---: | ---: | ---: |
| `1e-3` | `0.75` | `57.201 ± 1.709` | `54.856 ± 1.093` |
| `1e-3` | `1.0` | `54.821 ± 1.532` | `53.269 ± 1.162` |
| `2e-3` | `0.75` | `49.418 ± 0.728` | `48.534 ± 0.693` |
| `2e-3` | `1.0` | `48.304 ± 1.150` | `47.909 ± 1.358` |
| `3e-3` | `0.75` | `47.457 ± 0.962` | `47.943 ± 0.290` |
| `3e-3` | `1.0` | `47.294 ± 1.904` | `47.368 ± 2.137` |
| `5e-3` | `0.75` | `48.563 ± 3.637` | `47.344 ± 1.734` |
| `5e-3` | `1.0` | `48.196 ± 4.022` | `48.853 ± 4.062` |

短期ではAPOLLO本体はLR=`3e-3`・scale=`1.0`が最良で、APOLLO-ConfはLR=`5e-3`・
scale=`0.75`が最良だった。ただし、同じrank・LR・scaleで比較した場合、APOLLO-Confの
改善は条件依存であり、rank=`4`・LR=`3e-3`・scale=`1.0`ではほぼ同等だった。stateは
全条件で`11.68 MiB`で、LR/scaleによるstate増加はない。LR=`1e-3`は明らかに弱く、
rank=`4`ではLR=`2e-3〜5e-3`を候補範囲とした。続いてAPOLLO-Confの短期最良条件を実効100 stepへ
延長した。

APOLLO-Confの短期最良条件を、rank=`4`・LR=`5e-3`・scale=`0.75`・limiter無効・
TinyStories・CUDA/BF16・seed=`0,1,2`・train/eval token=`51200/1024`・実効100 step・
診断なしへ延長した。両optimizerの6ケースが`passed`となった。

| optimizer | validation loss mean±std | persistent state | peak allocated | peak reserved | host step time mean±std |
| --- | ---: | ---: | ---: | ---: | ---: |
| APOLLO | `27.764 ± 1.314` | `11.68 MiB` | `2.931 GiB` | `3.414 GiB` | `39.04 ± 0.74 ms` |
| APOLLO-Conf | `27.293 ± 0.951` | `11.68 MiB` | `2.931 GiB` | `3.414 GiB` | `43.13 ± 1.08 ms` |

100 stepでもAPOLLO-ConfがAPOLLOを平均`0.470`改善し、3 seedすべてで改善した。step時間は
約`4.1 ms`（約`10.5%`）増加したが、stateとpeak VRAMは同じだった。APOLLO本体も、前候補の
LR=`3e-3`・scale=`1.0`に比べてlossが改善したため、rank=`4`・LR=`5e-3`・scale=`0.75`を
現時点の第一候補とする。ただしAdamW-SFのloss=`15.657`との差は残っており、既定値変更では
なく低state optimizerの候補条件として扱う。

<a id="apollo-adamw-sf-300-step"></a>
#### AdamW-SFとの実効300 step比較

AdamW-SFを、APOLLO第一候補と同じTinyStories・CUDA/BF16・seed=`0,1,2`・
train/eval token=`153600/1024`・実効300 step・診断なし・LR=`5e-3`で実行した。

| optimizer | validation loss mean±std | persistent state | peak allocated | peak reserved | host step time mean±std |
| --- | ---: | ---: | ---: | ---: | ---: |
| AdamW-SF | `11.482 ± 0.235` | `692.62 MiB` | `4.079 GiB` | `4.896 GiB` | `60.10 ± 2.33 ms` |
| APOLLO | `7.381 ± 0.277` | `11.68 MiB` | `2.931 GiB` | `3.414 GiB` | `37.98 ± 2.30 ms` |
| APOLLO-Conf | `7.114 ± 0.336` | `11.68 MiB` | `2.931 GiB` | `3.414 GiB` | `39.92 ± 4.03 ms` |

この条件ではAPOLLO-Confが最良lossで、AdamW-SFを`4.369`下回った。APOLLOはAdamW-SFに
対してstateを約`98.3%`、peak allocatedを約`28.2%`、peak reservedを約`30.3%`削減し、
step時間も約`36.8%`短かった。APOLLO-ConfはAPOLLOよりさらに`0.267`改善したが、step時間は
約`5.1%`長い。これはrank=`4`・LR=`5e-3`の短期長期比較であり、AdamW-SFのLR最適化や
異なるタスクへの一般化を示すものではない。現時点では、APOLLO-Confを既定化せず、APOLLOを
低state候補、AdamW-SFを品質基準として扱う。

AdamW-SFのLR=`3e-3`も同じ実効300 step条件で追加測定した。lossは`13.262 ± 0.509`、
stateは`692.62 MiB`、peak allocated/reservedは`4.079/4.896 GiB`、step時間は
`59.14 ± 3.36 ms`だった。LR=`5e-3`のloss=`11.482 ± 0.235`の方が良く、今回の
TinyStories条件ではAdamW-SFの候補LRを`5e-3`とする。これによりAPOLLO-Confのloss=`7.114`
はAdamW-SFのLR=`3e-3,5e-3`の両方を上回るが、rank=`4`・実効300 stepに限定した結果である。

<a id="apollo-rank4-300-step"></a>
#### APOLLO rank=4 第一候補の実効300 step結果

rank=`4`・LR=`5e-3`・scale=`0.75`・limiter無効のAPOLLO/APOLLO-Confを、TinyStories・CUDA/BF16・
seed=`0,1,2`・train/eval token=`153600/1024`・実効300 step・診断なしで比較した。両optimizerの
6ケースが`passed`となった。

結果表は[AdamW-SFとの実効300 step比較](#apollo-adamw-sf-300-step)に示した。

100 step時のloss（APOLLO=`27.764`、APOLLO-Conf=`27.293`）から両者とも改善し、APOLLO-Confは
300 stepでもAPOLLOを平均`0.267`改善した。全caseが有限値で完走し、発散は観測されなかった。
ただし診断なしrunのため、step単位の振動やtrajectory curvatureはこの比較では測定していない。
stateとpeak VRAMは100 step時と同じで、APOLLO-Confのstep時間は約`1.9 ms`（約`5.1%`）長い。
同条件のtrajectory診断結果は次節に記録した。

<a id="apollo-trajectory-diagnostic"></a>
#### Initial 3 epoch trajectory diagnostic result

上記の診断runを、TinyStories・CUDA/BF16・rank=`4`・LR=`5e-3`・scale=`0.75`・limiter無効・
seed=`0,1,2`・3 epoch × 100 step（実効300 step）で完了した。各optimizerの3 seedと3個の
周期validation点が揃っており、全9ケースが`passed`となった。

| optimizer | validation loss mean±std | persistent state | normalized roughness | validation loss Δ² abs mean |
| --- | ---: | ---: | ---: | ---: |
| APOLLO | `7.437 ± 0.327` | `11.68 MiB` | `2.220` | `1.595` |
| APOLLO-Conf | `7.147 ± 0.345` | `11.68 MiB` | `2.156` | `1.997` |
| AdamW-SF | `9.767 ± 0.298` | `692.62 MiB` | `2.050` | `2.068` |

この初回結果では、APOLLO系のvalidation loss 2階差分はAdamW-SFより小さく、lossの観測系列は
比較的滑らかだった。一方、effective updateのnormalized roughnessはAdamW-SFが最小であり、
「loss曲率が小さいこと」と「parameter update軌跡が滑らかなこと」は一致しなかった。したがって、
低state化がtrajectory smoothingを直接もたらす、または曲率が品質を因果的に説明するとはまだ結論
できない。次は同一optimizer内で、refresh方式・rank・LRを変えた介入比較を行い、loss曲率と
update roughnessのどちらが品質差と再現性よく対応するかを確認する。
