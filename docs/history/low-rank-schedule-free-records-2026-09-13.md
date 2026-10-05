# APOLLO-SPR and Low-Rank ScheduleFree research records

記録日: 2026-09-13

この文書は、当時の実装段階、測定結果、比較計画、未決定事項を保存した研究履歴です。現在の実装状態や推奨条件を示しません。数式と設計上の基準は[設計文書](../low-rank-schedule-free-design.md)、現行の実装契約はコードと[`optimizers.md`](../optimizers.md)を参照してください。

元のrun JSON等は公開treeに含めず、この文書には解釈に必要な条件、集計値、制約を残しています。

## 実装段階

### 1. CAME-SF full oracle（実装済み）

`CAMESF`はCAME更新を保ってfull-size `h`を持つ参照実装で、低メモリ化ではなく
Schedule-Free式とtrain/eval/checkpoint動作のoracle比較に使う。

### 2. CAME-LRSF（固定projection版を実装済み）

`CAME-LRSF`はCAMEの更新stateを保ち、Schedule-Free差分`h`だけを固定projection上の`H`で
近似する比較baselineである。CAME本体の更新順序を維持し、低rank経路が不利なparameterは
CAMEへfallbackする。現行のAPI・dtype・checkpoint契約は[`optimizers.md`](../optimizers.md)、
数式上の設計根拠は[設計snapshot](../low-rank-schedule-free-design.md)を参照する。
最初の10-step CPU ImageAE probeは次節の複数optimizer比較にまとめた。

### 3. APOLLO-SPR

APOLLOの更新projection `R_update`にPA/PBとsmooth weightを導入し、LRSFとは分けてmomentとの
相互作用を調べた。basis変更ではprojectionだけでなくmomentもtransportし、smooth期間中は
stateが二系統になるため、低メモリ化とは別の評価軸とした。

`M >= N`で係数を右側へ持つ場合、隣接projection間の座標変換を

```text
T = P_current^T P_next (P_next^T P_next + lambda I)^-1
m_next = m_current T
v_next = v_current (T.square())
```

とする。`m`はfirst moment、`v`はdiagonal second momentである。`v`はfull covariance
を持たないため、`v_next`は対角共分散を維持する近似であり、完全なbasis changeと
同値ではない。`M < N`では左側の対応する式を使う。したがってAPOLLO-SPRは、
projection blendだけでなく、moment transport誤差も独立に記録する。

### 4. APOLLO-CAME-LRSF（固定projection版を実装済み）

`APOLLOCAMELRSF`はAPOLLO-CAMEの低rank更新stateとLRSF差分`H`を組み合わせる。
`R_update`と`R_delta`は独立して保持し、fallback parameterではLRSFを適用しない。
checkpoint復元後の次step一致をunit testで確認した。APIと復元契約は現行仕様を参照する。

同一CPU ImageAE probe（FP32、batch=2、warmup=2、steps=10、rank=4、latent=4、
bottleneck=16）では、CAMEのfinal loss/state/optimizer stepが
`0.0477765`/`298600` bytes/`4.98 ms`、CAME-SFが
`0.0666551`/`430820` bytes/`5.61 ms`、CAME-LRSFが
`0.0620128`/`335640` bytes/`6.23 ms`、APOLLO-CAMEが
`0.0760455`/`130680` bytes/`6.66 ms`、APOLLO-CAME-LRSFが
`0.0636235`/`167608` bytes/`8.57 ms`だった。短いCPU probeのため、収束品質やGPU速度の
結論ではなく、state構造と実行経路の確認値として扱う。

同じ条件でCAME-LRSFのrank sweep（FP32、batch=2、warmup=2、steps=30）を行うと、
rank=1/4/8のfinal lossはそれぞれ`0.0411252`/`0.0392572`/`0.0257266`、
stateは`308608`/`335640`/`346056` bytesだった。rank=8ほどCAME-SF fullに近づくが、
rank=1はstate増分を最小化できる。100 stepの予備収束では、CAME-SFが`0.0213559`、
CAME-LRSF rank=1/4/8が`0.0039867`/`0.0037307`/`0.0017049`となった。
これはCAME-SF fullの理論的oracle性と低ランク近似の実用的な収束差を示すが、
CPUの小型synthetic probeであり、実データ品質やCUDA性能の結論ではない。

weight decayは`0`/`1e-3`/`1e-2`で別途測定した。50 stepの小型CPU probeでは、
CAMEのfinal lossは`0.00205699`/`0.00205699`/`0.00205667`、CAME-SFは
`0.04399030`/`0.04399035`/`0.04399085`、CAME-LRSF rank=4は
`0.02054246`/`0.02054255`/`0.02054330`だった。state bytesは各optimizer内で
不変だった。この範囲ではdecoupled CAME fallbackとSchedule-Free方向への加算の差は
小さく、実データ・より長い学習で再確認する。

実データの初回確認として、ローカルCIFAR-10からtrain/validation各512枚を取り出し、
epochごとの固定permutationをoptimizer間で共有して3 epoch学習した。CAME / CAME-SF /
CAME-LRSF rank=4 / APOLLO-CAME-LRSF rank=4のvalidation lossは、それぞれ
`0.0135198` / `0.0295460` / `0.0219063` / `0.0216251`だった。persistent stateは
`298600` / `430820` / `335640` / `167548` bytes、optimizer stepは
`5.46` / `5.61` / `7.43` / `7.98` msだった。これは実画像での初回sanity checkであり、
512枚・3 epoch・CPUのため、CIFAR-10全体の品質順位やGPU性能の結論には使わない。

同じCIFAR-10 subsetでrank=1/4/8/16を比較したところ、CAME-LRSFのvalidation lossは
`0.0218625`/`0.0219063`/`0.0188302`/`0.0148759`、APOLLO-CAME-LRSFは
`0.0399103`/`0.0216251`/`0.0177086`/`0.0134767`だった。一方、LRSF backend数は
それぞれCAME-LRSFで`16/13/9/3`、APOLLO-CAME-LRSFで`16/13/9/2`であり、rank=16の
state減少は一部パラメータがfull-rank fallbackへ移行した影響を含む。rank増加による
改善とfallback構成の変化を分離する必要があるため、現時点ではrank=8または16を既定値に
変更せず、現行rank=4を維持する。これは`docs/apollo-experiment-records.md`で定義した
「低rank制約」と「projection refresh由来の探索性」を別対照にする方針とも整合する。

`R_delta`の固定projectionとsmooth refreshも同じsubset・初期parameter・固定permutationで
比較した。条件はtrain/validation各512枚、3 epoch、batch=16、CPU FP32、rank=4、
refresh interval/window=8、smoothstepである。固定条件のvalidation lossは
CAME-LRSF=`0.0219063`、APOLLO-CAME-LRSF=`0.0216251`だった。smooth条件では
それぞれ`0.0216158`、`0.0225928`となり、11回のrefresh eventが発生した。CAME-LRSFでは
わずかな改善、APOLLO-CAME-LRSFでは悪化であり、現時点でsmooth refreshを既定化する根拠は
ない。再実行時のoptimizer stepは固定条件で`9.63`/`11.55` ms、smooth条件で
`10.21`/`9.25` msだったが、CPUの実行揺れが大きく、速度差の結論には使わない。
smooth条件では各eventにrefresh直前のloss、同一batchの更新後loss、次stepの観測lossを
記録する。CAMEのvalidation lossは両条件で`0.0135198`と不変だった。この比較はCPU・小subsetの
挙動確認であり、GPUの実運用速度や長期収束の結論ではない。

refresh interval/windowの予備sweepも同じ条件で行った。validation lossは次の通りである。

| R_delta policy | CAME-LRSF | APOLLO-CAME-LRSF | events |
| --- | ---: | ---: | ---: |
| fixed (frozen, mode=none) | 0.0219063 | 0.0216251 | 0 |
| smooth, interval=8, window=8 | 0.0216158 | 0.0225928 | 11 |
| smooth, interval=16, window=4 | 0.0215628 | 0.0222982 | 5 |
| smooth, interval=16, window=16 | 0.0215691 | 0.0219515 | 5 |
| hard, interval=16 | 0.0216242 | 0.0224778 | 5 |
| smooth, interval=32, window=16 | 0.0220515 | 0.0218351 | 2 |

この小規模条件では、`interval=16/window=16`または`interval=32/window=16`が
`APOLLO-CAME-LRSF`の候補であり、interval=8より悪化が小さい。ただし3 epoch・512枚・CPUの
予備測定であり、smooth refreshの既定値は変更しない。

候補を5 epochへ延長すると、fixed / smooth(16,16) / smooth(32,16)のvalidation lossは、
CAME-LRSFで`0.0173513` / `0.0175113` / `0.0169034`、APOLLO-CAME-LRSFで
`0.0182026` / `0.0193572` / `0.0189410`となった。`smooth(32,16)`はCAME-LRSFでは
改善した一方、APOLLO-CAME-LRSFではfixedを上回らなかった。したがって、refreshはLRSF
全体の既定機能ではなく、まずCAME-LRSF単体の候補として長期検証する。

stochastic選択も同じinterval=16/window=16、rank=4、CPU FP32条件で比較した。
3 epochではfixed / smoothstep / stochasticのvalidation lossが、CAME-LRSFで
`0.0219063` / `0.0215691` / `0.0214635`、APOLLO-CAME-LRSFで
`0.0216251` / `0.0219515` / `0.0219033`だった。5 epochではCAME-LRSFが
`0.0173513` / `0.0175113` / `0.0173794`、APOLLO-CAME-LRSFが
`0.0182026` / `0.0193572` / `0.0193095`となった。stochasticはsmoothstepより
両系統で改善したが、fixedを安定して上回る結果ではない。従って現時点では既定mixを
変更せず、branch-level noiseを利用した探索候補として扱う。CPU step時間は
揺れが大きいため、速度比較の結論には使わない。

double buffer EMAも同じ条件で比較した。3 epochではCAME-LRSFが
`0.0215281`、APOLLO-CAME-LRSFが`0.0218883`、5 epochではそれぞれ
`0.0174004`、`0.0192599`だった。EMAはsmoothstepより両系統で改善し、
5 epochのAPOLLO-CAME-LRSFではstochasticの`0.0193095`も僅かに下回ったが、
fixedの`0.0182026`には届かなかった。現時点ではEMAも既定mixにせず、
smoothstep/stochasticと併せてrefresh候補として評価する。

固定projection（`fixed/frozen`, `mode=none`）で実データ比較を10 epochへ延長した。
CIFAR-10 train/validation各512枚、CPU FP32、batch=16、rank=4、learning rate=`2e-4`、
weight decay=`0`、同一seed・epoch permutationの条件である。最終validation lossは
`CAME=0.0090911`、`CAME-SF=0.0153335`、`CAME-LRSF=0.0127685`、
`APOLLO-CAME-LRSF=0.0153793`となった。

この条件では、full hidden deltaを持つ`CAME-SF`が必ずしもCAMEを上回らず、
Schedule-Free差分を低rank化した`CAME-LRSF`の方が`CAME-SF`より良かった。
一方、persistent stateは`CAME=298600` bytesに対して`CAME-SF=430820`、
`CAME-LRSF=335640`、`APOLLO-CAME-LRSF=167548` bytesで、
`APOLLO-CAME-LRSF`はCAME比で約43.9%削減できた。平均optimizer step時間は順に
`5.39`、`5.96`、`6.88`、`7.25` msであり、低メモリ化にはprojection・delta適用の計算コストが残る。
これはCPU・小subset・単一seedの長期傾向であり、GPU性能や既定値変更の根拠にはしない。

同じ固定/frozen条件を5 epoch・3 seedへ広げたところ、最終validation lossの平均±標準偏差は
次の通りだった。

| optimizer | mean validation loss | std | state bytes |
| --- | ---: | ---: | ---: |
| CAME | 0.0111172 | 0.0005629 | 298600 |
| CAME-SF | 0.0250808 | 0.0020080 | 430820 |
| CAME-LRSF | 0.0156091 | 0.0012913 | 335640 |
| APOLLO-CAME-LRSF | 0.0178171 | 0.0004545 | 167548 |

3 seedすべてでCAMEが最良、CAME-SFが最も悪い順位になり、今回の小型ImageAE条件では
順位の逆転は観測されなかった。`APOLLO-CAME-LRSF`は`CAME-LRSF`よりvalidation lossが
平均で約14.1%高い一方、stateはCAME比で約43.9%少ない。これはCPU FP32・固定subset・
5 epochの候補選別結果であり、GPU性能や一般的なSchedule-Free優位性を示すものではない。
seed別の測定値は前掲の平均・標準偏差に集約した。

同じ3 seed・5 epoch条件で、`smooth` interval/window=`16/16`の`fixed/frozen`対照を
`smoothstep`と`ema`で比較した。最終validation lossの平均±標準偏差は次の通りである。

| R_delta mix | CAME-LRSF | APOLLO-CAME-LRSF |
| --- | ---: | ---: |
| fixed/frozen | 0.0156091 ± 0.0012913 | 0.0178171 ± 0.0004545 |
| smoothstep | 0.0157362 ± 0.0012772 | 0.0188628 ± 0.0004450 |
| ema | 0.0156926 ± 0.0012293 | 0.0187822 ± 0.0004401 |

`ema`は`smoothstep`よりCAME-LRSFで約0.28%、APOLLO-CAME-LRSFで約0.43%改善したが、
fixed/frozenよりはそれぞれ約0.53%、約5.42%悪かった。したがって、EMAはspikeを
抑える候補として実装・保存するが、現在の固定subset条件では既定mixにしない。

GPU実行では、`verify/launchers/run_lrsf_gpu_validation.sh`を使い、NVIDIA GeForce RTX 3080 Ti
（driver=`616.56`、VRAM=`12288 MiB`）、BF16、CIFAR-10各512枚、5 epoch、batch=8、
rank=4、latent=`16`、bottleneck=`256`、downsample stages=`3`、
refresh interval/window=`64/32`で比較した。最終validation lossは次の通りである。

| R_delta policy | CAME-LRSF | APOLLO-CAME-LRSF |
| --- | ---: | ---: |
| fixed/frozen | 0.0633700 | 0.0139154 |
| hard | 0.0633121 | 0.0139629 |
| smoothstep | 0.0633192 | 0.0137297 |
| ema | 0.0633203 | 0.0137443 |
| stochastic | 0.0633148 | 0.0137485 |
| orthogonal | 0.0632899 | 0.0135975 |

この単一seedのGPU条件では、`CAME-LRSF`は全方式の差が小さく、orthogonalがfixed比で
約0.13%改善した。`APOLLO-CAME-LRSF`ではorthogonalがfixed比で約2.28%改善し、
smoothstep、EMA、stochasticも約1.20--1.33%改善した。一方、hardは約0.34%悪化した。
平均optimizer step時間はfixedから、CAME-LRSFで`27.61` ms、hard=`28.12` ms、
smoothstep=`30.71` ms、EMA=`29.84` ms、stochastic=`28.53` ms、orthogonal=`36.81` ms、
APOLLO-CAME-LRSFでfixed=`26.66` ms、hard=`28.07` ms、smoothstep=`28.70` ms、
EMA=`30.67` ms、stochastic=`26.83` ms、orthogonal=`40.15` msだった。orthogonalは品質候補
である一方、step時間がCAME-LRSFで約33%、APOLLO-CAME-LRSFで約51%増える。

optimizer stateはCAME-LRSFが`47279618` bytes、APOLLO-CAME-LRSFが`1933530` bytesで、
後者はCAME-LRSF比で約95.9%削減された。ただし、このprobeはCUDA allocatorのpeak
allocated/reservedも記録するよう更新した。fixed/frozenのpeak allocatedは
CAME-LRSF=`103762432` bytes、APOLLO-CAME-LRSF=`51217920` bytesだった。smoothstep、
EMA、stochasticではstateのpeakがそれぞれCAME-LRSF=`47724978` bytes、
APOLLO-CAME-LRSF=`2378890` bytesとなり、定常stateから`445360` bytes増加した。
allocatorのpeak allocatedは両系統でfixed比約0.5--0.9%の増加に収まり、reservedは
`121634816` bytesで共通だった。したがって、double-bufferの追加stateは存在するが、
この条件ではallocator peakへの影響は限定的だった。より大きなモデル・rank・batchでは
別途再確認が必要である。

同じGPU条件でseed=1,2を追加し、seed=0を含む3 seedの平均を比較した。
`APOLLO-CAME-LRSF`の最終validation lossはfixed=`0.0138273 ± 0.0002066`、
hard=`0.0139173 ± 0.0002292`、smoothstep=`0.0137558 ± 0.0002502`、
EMA=`0.0137570 ± 0.0002470`、stochastic=`0.0137592 ± 0.0002482`、
orthogonal=`0.0137486 ± 0.0002064`だった。orthogonalはfixed比で約0.57%改善したが、
smoothstep、EMA、stochasticとの差は約0.01--0.06%であり、標準偏差と比較して小さい。
`CAME-LRSF`ではsmoothstep=`0.0620549 ± 0.0019382`、stochastic=`0.0626706 ± 0.0011224`が
fixed=`0.0631850 ± 0.0001412`を下回ったが、seed=1の改善が大きく、再現性は限定的である。

従って、現時点の推奨は`fixed/frozen`を既定として維持し、探索時の候補にsmoothstep、EMA、
stochastic、orthogonalを残すことである。orthogonalはAPOLLO-CAME-LRSFの品質候補だが、
step時間が約22%--50%増えるため、品質だけで既定化しない。seed=1,2の測定値は前掲の
平均・標準偏差に集約した。

rank sweepも同じGPU条件でseed=0,1,2、fixed/frozen、rank=1/4/8/16を比較した。
`APOLLO-CAME-LRSF`の最終validation lossはrank=1=`0.2143185 ± 0.0950438`、
rank=4=`0.0138273 ± 0.0002066`、rank=8=`0.0135811 ± 0.0011461`、
rank=16=`0.0148701 ± 0.0001393`だった。rank=8はrank=4より平均で約1.78%良いが、
seed間のばらつきが大きく、stateは`1.93 MB`から`3.62 MB`へ増える。rank=16はstateが
`6.93 MB`まで増える一方、rank=8より悪化した。rank=1はstateが`0.67 MB`まで減るが、
全seedで収束不良になった。

`CAME-LRSF`ではrank=1/8の一部seedで改善が見られたものの、標準偏差が大きく、rank=4の
`0.0631850 ± 0.0001412`が安定した。今回の実用候補は、品質・state・再現性のバランスから
rank=4を維持する。rank=16もLRSF backendを含み、全体がCAMEへfull-rank fallbackしたわけでは
ないが、低rank近似のstate削減効果は薄くなる。

続けて、rank依存のrefresh効果を確認するため、同じGPU条件でrank=4/8、fixed/frozen、
smoothstep、EMA、orthogonalを3 seedで比較した。12本のJSONはすべて`passed`であり、
最終validation lossの平均±標準偏差は次の通りである。

| optimizer | rank | fixed/frozen | smoothstep | EMA | orthogonal |
| --- | ---: | ---: | ---: | ---: | ---: |
| APOLLO-CAME-LRSF | 4 | 0.0138273 ± 0.0002066 | 0.0137558 ± 0.0002502 | 0.0137570 ± 0.0002470 | 0.0137486 ± 0.0002064 |
| APOLLO-CAME-LRSF | 8 | 0.0135811 ± 0.0011461 | 0.0134655 ± 0.0013291 | 0.0134647 ± 0.0013229 | 0.0134932 ± 0.0011906 |
| CAME-LRSF | 4 | 0.0631850 ± 0.0001412 | 0.0620549 ± 0.0019382 | 0.0634429 ± 0.0001696 | 0.0631645 ± 0.0001014 |
| CAME-LRSF | 8 | 0.0576214 ± 0.0075949 | 0.0631960 ± 0.0002788 | 0.0631723 ± 0.0002475 | 0.0582106 ± 0.0067492 |

`APOLLO-CAME-LRSF`ではrank=4のrefresh差はfixed比で約0.51--0.57%、rank=8では
約0.65--0.86%の改善だった。ただしrank=8はfixedを含めてseed間標準偏差がrank=4の
約5--6倍であり、EMAとsmoothstepの差も小さい。orthogonalはrank=4で平均step時間が
`25.23` msから`30.89` ms、rank=8で`24.52` msから`31.02` msとなり、約22--27%増加した。

`CAME-LRSF`ではrank=4のsmoothstepだけがfixed比で約1.79%良かったが、seed=1の
`0.0593165`が主因で標準偏差も大きい。rank=8のfixedもseed=0の`0.0468811`により
平均が押し下げられており、smoothstep/EMAの約`0.0632`という安定値を上回る根拠には
ならない。したがって、両optimizerとも既定refreshはfixed/frozen、実用rankは4を維持し、
EMA/smoothstepは探索候補、orthogonalは速度を許容できる場合の候補とする。

メモリ面では、smoothstep/EMAのdouble bufferにより、APOLLO-CAME-LRSFのpeak stateは
rank=4で`1,933,530` bytesから`2,378,890` bytes、rank=8で`3,622,106` bytesから
`4,512,826` bytesへ増えた。一方、peak allocatedはrank=4で約0.94%、rank=8で約1.82%の
増加に留まり、peak reservedは全方式で`121,634,816` bytesだった。追加stateと品質差を
考慮すると、現時点でsmooth refreshを既定化する根拠はない。


### APOLLOのloss-directed rotation確認とLRSFへの適用方針

APOLLO本体のorthogonal refreshには、`direction="loss_directed"`が追加されている。
実装は実損失を直接評価するものではなく、現在gradientを`G`、射影行列を`R`としたときの
次の射影勾配エネルギーを増やす一次近似である。

```text
E(R) = 1/2 ||G R||_F^2       (rows >= cols)
E(R) = 1/2 ||R G||_F^2       (rows < cols)
```

`rotate_orthogonal_projection`は、`G^T G R`または`R G G^T`をStiefel接空間へ射影し、
微小step後にreduced QRで直交基底へ戻す。APOLLOの更新経路では現在gradientを使い、
`R_update`の低rank momentとsmooth中のnext momentを同じ座標変換でtransportする。
したがって、これは「loss-directed」という名前の実損失最適化ではなく、gradient-energy-directed
な探索proxyである。gradientがない場合にrandomへ暗黙fallbackせずエラーにする点、seedと
`orthogonal_refresh_count`からresume後の系列を再現する点も既存契約である。専用unit testでは
射影gradient energyの増加、矩形shape、直交性、APOLLO/APOLLO-CAMEの現在gradient利用を確認している。

この思想はLRSFにも技術的には既に接続されている。`CAMELRSF`と`APOLLOCAMELRSF`は共有の
`OrthogonalRefreshPolicy`と`rotate_projection_state`を使い、現在は`parameter.grad`をsignalとして
`R_delta`を回転する。回転後は`lrsf_delta`、およびsmooth中の`refresh_next_delta`をtransport
するため、次の低レベル指定は動作契約上可能である。

```python
CAMELRSF(
    params,
    orthogonal_refresh={
        "rate": 0.005,
        "direction": "loss_directed",
    },
)

APOLLOCAMELRSF(
    params,
    orthogonal_refresh={"rate": 0.005, "direction": "loss_directed"},
    # これはR_update側とは別の設定
    apollo_orthogonal_refresh={"rate": 0.005},
)
```

ただし、APOLLOの`R_update`とLRSFの`R_delta`ではproxyの意味が異なる。APOLLOでは
`R_update`がgradientの低rank momentを直接決めるため、`||G R_update||`を増やす目的と
更新stateの探索が比較的整合する。一方LRSFの`R_delta`は、CAMEまたはAPOLLO-CAMEの主更新を
置き換えず、Schedule-Freeの累積hidden deltaを表現するbasisである。従ってraw gradientの
射影energyだけを最大化すると、hidden deltaの保持ではなく瞬間的なgradientノイズへbasisを
追従させ、delta transport誤差、update norm variance、loss spikeを増やす可能性がある。
特に`APOLLOCAMELRSF`では`R_update`と`R_delta`が別物なので、`R_update`用のsignalをそのまま
`R_delta`へ流用しない。

#### LRSF向けの推奨variant

LRSFでは`R_update`と`R_delta`の役割が異なるため、signalも分けて検証した。候補は
`gradient`（APOLLO互換の射影gradient energy）、`effective_update`（CAME系optimizerが
実際に使う更新方向）、`loss_lowering`（現在のhidden delta係数`D`を固定した局所proxy）である。
最後の方式は実損失を直接評価せず、`D=0`ではbasis選択の信号を持たない。`R_delta`回転後は
低rank delta stateを新basisへtransportする。これらはLRSF専用の探索方式であり、APOLLO本体の
`R_update`とは独立に扱う。数式・更新契約の詳細は[設計snapshot](../low-rank-schedule-free-design.md)
と現行の[optimizer契約](../optimizers.md)を参照。

#### 導入判断と検証順序

採用判断ではfixed/frozenを基準にrandom orthogonalとも比較し、複数seedでのloss、update norm
variance、recovery、step時間、state/VRAMを確認する方針とした。以下のGPU記録ではgradientと
effective updateについて3 ratesを測定した。loss-loweringとrandom orthogonalの同率比較、
projection/delta transport相対誤差はこの記録時点で未計測であり、採否を保留した。

#### loss-directed signalの予備GPU比較

2026-09-13、RTX 3080 Ti・BF16・CIFAR-10固定subset（train/validation各512枚、5 epoch）、
rank=4、orthogonal rate=0.01、3 seedで比較した。`fixed/frozen`を基準にした最終validation
lossの平均±標準偏差は次の通りである。

| optimizer | fixed/frozen | loss-directed / gradient | loss-directed / effective_update |
| --- | ---: | ---: | ---: |
| APOLLO-CAME-LRSF | 0.0138273 ± 0.0002066 | 0.0138073 ± 0.0001653 | 0.0141466 ± 0.0003139 |
| CAME-LRSF | 0.0631849 ± 0.0001412 | 0.0546802 ± 0.0123989 | 0.0634438 ± 0.0002236 |

`APOLLO-CAME-LRSF`ではgradient版はfixed比約0.14%の僅かな改善に留まり、
`effective_update`版は約2.31%悪化した。`CAME-LRSF`のgradient版は平均だけを見ると約13.5%
改善したが、seed=1の`0.0371462`が主因で、seed=0/2は`0.0635709`/`0.0633237`だった。
したがって、gradient版の改善も現時点では再現性のある効果とは扱わない。effective update版は
全seedでfixedと同程度または悪く、この条件では「loss-directed」のproxyとして不適切な可能性が
高い。実損失ではなく、更新エネルギーの大きい方向を強化しているため、LRSFの`R_delta`に対して
有効な探索方向と一致しないと考えられる。

orthogonalのoptimizer step時間はfixed比で、`APOLLO-CAME-LRSF`が`29.73` msから`46.39` msへ
約56%、`CAME-LRSF`が`26.97` msから`41.80` msへ約55%増加した。orthogonal-onlyのため、
persistent state、peak state、peak allocated/reservedはfixedと変わらなかった。なお、このprobe
ではprojection energy、delta transport誤差、update norm分散、refresh後のrecovery stepsは
記録していないため、これらについての結論は保留する。

rate=0.001/0.005についても同じ条件で追加計測した。fixedに対する最終validation lossの
相対差、update norm varianceの比、refresh eventから計算したrecovery stepsは次の通りである。

| signal | rate | optimizer | validation loss relative to fixed | update variance ratio | recovery steps |
| --- | ---: | --- | ---: | ---: | ---: |
| gradient | 0.001 | APOLLO-CAME-LRSF | -4.51% | 0.997 | 3.985 |
| gradient | 0.005 | APOLLO-CAME-LRSF | -2.69% | 0.999 | 3.883 |
| effective_update | 0.001 | APOLLO-CAME-LRSF | -1.53% | 0.999 | 3.803 |
| effective_update | 0.005 | APOLLO-CAME-LRSF | -2.56% | 0.999 | 3.871 |
| gradient | 0.001 | CAME-LRSF | +0.02% | 0.999 | 4.270 |
| gradient | 0.005 | CAME-LRSF | -7.16% | 0.997 | 4.576 |
| effective_update | 0.001 | CAME-LRSF | +0.05% | 1.000 | 4.277 |
| effective_update | 0.005 | CAME-LRSF | +0.17% | 1.003 | 4.249 |

APOLLO-CAME-LRSFでは、rate=0.001/0.005の両signalが改善したが、rate=0.01では
`effective_update`が約2.31%悪化し、単調なrate依存性はない。CAME-LRSFではeffective updateの
差は0.2%未満で、gradient rate=0.005の改善はseed=1の`0.0494067`に依存する。したがって、
loss-directed signalの効果はoptimizerとrateの組み合わせに依存し、現時点で既定化できない。

update norm varianceの比は全条件でほぼ1.0だった。つまり、今回のloss差はparameter updateの
分散制御によるものではなく、低rank deltaが追従するprojection subspaceの違いによる可能性が
高い。recoveryは約3.8--4.6 stepsだったが、orthogonalは毎step発生するため、これは各stepの
観測lossをpre-refresh lossへ戻す単純指標であり、独立したrefresh spikeの因果的recoveryとは
解釈しない。

計測runでは全320 stepsでorthogonal eventが発生し、stateとpeak VRAMはfixedから変化しなかった。
optimizer step時間は条件によりfixed比約16--43%増加した。次はrandom orthogonalとの同率比較と、
projection/deltaのtransport相対誤差を追加し、改善がloss-directed固有かを切り分ける。

## 初期計画の記録範囲

当時の比較は、Schedule-Free full oracleと低rank版、APOLLO-SPR、`R_delta` refreshの
効果を順に分離する計画だった。固定projection、rank sweep、weight decay、CPU/GPU probeの
実測値と採否理由は上記に記録した。初期計画にあったshape別の追加計測やTriton融合検討は
実施済みとは扱わない。現行の実装状態と検証方法は[optimizer契約](../optimizers.md)を参照。
