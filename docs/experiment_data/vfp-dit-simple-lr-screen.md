# VFP-DiT Simple learning-rate screen

実施日: 2026-09-24

> **履歴記録:** 廃止した`vfp_dit_simple`実装を使った1 seedの短期screenです。数値はこの実装・条件での観測であり、現行の[`vfp_dit`](../../vfp_dit/)の性能や推奨設定を示しません。

## 条件

- 初期重み: `vfp_dit_simple.train_quality-512-w1024-d24-f4-t4-10x1024-w4-observe1000-seed42-resized_20260924T074809Z_abca2a68` のstep 2048 checkpoint
- 各条件: 512 optimizer steps、512 train samples、validation 256 samples（T2I/TI2I各128）
- 共通設定: seed 42、512px、width 1024、depth 24、APOLLO rank 32、BF16、4 workers
- optimizer stateはcheckpointから復元せず、重みだけを読み込んで各条件で新規作成
- 全条件でconstant LRを使い、短い比較期間にcosine decayが入り込まないようにした
- validation split・サンプリング順・初期ノイズseedは共通

## 結果

| LR | train loss | validation total | T2I | TI2I | train peak VRAM |
|---:|---:|---:|---:|---:|---:|
| `1e-4` | 1.3364 | 1.2898 | 1.1858 | 1.3938 | 4.92 / 5.49 GiB allocated/reserved |
| `3e-4` | 1.3608 | 1.2392 | 1.1801 | 1.2983 | 4.92 / 5.49 GiB allocated/reserved |
| `1e-3` | 1.4212 | 1.1991 | 1.1680 | 1.2302 | 4.92 / 5.49 GiB allocated/reserved |

共通初期checkpointのvalidationはtotal `1.2674`（T2I `1.1898`、TI2I `1.3451`）。この基準と比べ、`1e-4`はtotalが悪化し、`3e-4`は`0.0283`改善、`1e-3`は`0.0684`改善した。短期では高いLRほどvalidationが良く、特にTI2Iの改善が大きい。

各LRの固定seedサンプルPNGは公開アーカイブ整理時に削除した。画像から算出して記録したLR間の画素相関`0.89`〜`0.92`、MAE `0.089`〜`0.109`は集計値として残しているが、元画像は再確認できない。

## 解釈と制限

この比較は、今回の停滞に学習率が関わっているという仮説を支持する。元の10-epoch cosine runでは後半LRがほぼ0になっており、短期比較ではconstant `1e-3`が最良だった。ただし、512 samples・1 seedのscreenなので、`1e-3`を本番の既定値に決める根拠には不足する。そこで同じ初期checkpointから`1e-3` cosineを5 epochs継続する追試を行った（結果はFollow-up runを参照）。

学習・validation・checkpoint保存は3条件とも完了した。3条件目のsampler CLIは、画像PNGを保存した後、並行更新中の`generate_samples.py`にあるsampling metadata引数の不整合でJSON sidecar保存時に終了コード1となった。PNGは生成後に公開用archiveから削除した。sampler側の共同作業ファイルはこの比較では変更していない。

## Follow-up run

短期screenを受け、同じstep 2048 checkpointから`1e-3` cosine decay・`min_lr_ratio=0.1`・5 epochs × 1024 samplesを開始した。各epochでvalidation 256 samples、1024 optimizer stepsごとに同一prompt・同一noise seedの画像を保存する。run:
`vfp_dit_simple.train_lr-followup-1e-3-cosine-floor10pct-5x1024-seed42_20260924T142148Z_25a038cd` (raw archive removed)

epoch 1時点のvalidationはtotal `1.1755`（T2I `1.1521`、TI2I `1.1989`）で、初期checkpointの`1.2674`および512-step constant-LR screenの全条件を下回った。epoch 1時点のLRは`9.14e-4`。これは途中経過で、完了時の値を以下に記録する。

5 epoch・5120 optimizer stepsまで完了した。最終validationはtotal `1.1033`（T2I `1.0881`、TI2I `1.1184`）、最終epochのtrain lossは`1.1075`だった。最終epochのvalidationは256件で、run全体を通じてfiniteだった。step 1024, 2048, 3072, 4096, 5120の固定promptサンプル画像はraw artifact削除時に削除した。同一seed・同一noiseで初期checkpointから継続し、validationは初期`1.2674`から`0.1641`改善した。1 seedのscreenなので、再現性や画像品質の確定値ではない。

## Stage2 weights-only continuation

最終`1e-3` cosine checkpointから重みだけを読み、APOLLO stateを新規作成して`3e-4` cosine・floor `0.1`・5 epochs × 1024 samplesで継続した。validationはepoch 1〜5で`1.1308`、`1.1125`、`1.1031`、`1.0987`、`1.0969`（最終T2I `1.0820`、TI2I `1.1118`）と一貫して改善し、source checkpointの`1.1033`を`0.0064`下回った。step 1024/2048/3072の固定prompt・seed画像は高周波ノイズが支配的で、ボートの構造は判別できなかった。step 2048から3072への画素相関は`0.993`、MAEは`6.87/255`で、観測画像上の変化は小さい。これは勾配消失を直接示すものではない。1 seed・256 validation例の結果であり、品質改善の根拠にはならない。run directory、画像、sidecar、step-level metricsは公開用archiveから削除し、集計値だけを本資料に残した。

## Velocity magnitude diagnostic

最終checkpointを使い、固定したvalidation splitからT2I/TI2I各16例を取り、各画像につき4つの等幅timestep区間から1点ずつサンプルした。condition dropoutを切り、trainingと同じ`target = noise - clean_latent`で計算した。各区間のprediction RMSは`0.388`〜`0.478`、target RMSは`1.128`〜`1.152`で、prediction/target RMS比は`0.34`〜`0.42`だった。Cosineは`0.31`〜`0.40`で、zero predictionのMSE `1.273`〜`1.328`に対して実測MSEは`1.072`〜`1.202`だった。

この測定では出力はゼロではなく、教師方向と正の相関があり、zero predictionよりMSEを下げている。一律スカラーを掛ける場合の最適係数は各condition/timestep区間でおよそ`0.87`〜`1.15`だったため、prediction RMSがtarget RMSより小さいことだけから、出力振幅が不足しているとは結論できない。32例の診断なので全体品質の保証にはならない。既存のadapter ablationも、Transformer/FFN/Linear間のvalidation差が小さく、attentionを除くことでノイズ問題が解決する根拠を示していない。

同じ最終checkpoint・prompt・seed `42`・`flow_match_euler`・flow shift `1`で、Eulerのstep数とCFG scaleを分けて固定noise比較する6条件を計測した。条件はCFG 1 / 30 steps、CFG 2 / 30 steps、CFG 4 / 15・30・60 steps、CFG 7.5 / 30 steps。PNGとJSON sidecarはraw artifact削除時に削除した。同じnoiseでの画素MAEはCFG 1→2が`6.96/255`、1→4が`21.56/255`、1→7.5が`45.87/255`だった。CFG 4固定では15→30 stepsが`0.89/255`、30→60 stepsが`0.60/255`だった。今回の設定ではCFG scaleが出力に強く作用し、30から60へのstep増加は画素差が小さい。画素差は生成品質の評価ではなく、元画像も公開アーカイブには残っていない。

## Fixed-noise training progression

step 1024, 2048, 3072, 4096, 5120の固定prompt観測画像を画素比較した。隣接checkpoint間のMAEは`9.60`, `10.31`, `7.47`, `3.25` /255、step 1024と5120のMAEは`24.52` /255（pixel correlation `0.922`）だった。同じ初期noiseなので高い相関は予想されるが、出力画像は学習中ずっと同一ではなく、最終epochでは隣接checkpoint間の変化が小さくなっていた。これは画素差の測定であり、知覚品質や残留noiseの程度はcontact sheetで目視評価する。

時系列contact sheetはraw artifact削除時に削除した。隣接step間の画素比較値のみをこの集計資料に残している。

step 5120の画像を目視すると、promptの「赤い木製ボート」は判別できず、高周波ノイズが支配的だった。stage2 epoch 1/2/3 (step 1024/2048/3072) の同一prompt・seed 42・Euler 30 steps・CFG 4画像も同様に構造が見えない。stage2のstep 2048→3072はMAE `6.87/255`、pixel correlation `0.993`で変化が小さい。lossが改善しても生成画像の構造は確認できないため、学習延長の前に固定checkpoint上でsamplerの時間方向・velocity出力・decodeを切り分けて診断する。
