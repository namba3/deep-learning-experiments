# LRTDO research summary

[English](lrtdo-research-summary.en.md) | [日本語](lrtdo-research-summary.md)

このcompact archiveは未実施の研究phase案・optimizer variant案・将来benchmark計画を圧縮し、診断、prototype、比較の記録とその条件を残しています。snapshotにある「current」「next」「candidate」は記録時点の表現であり、現在の作業指示ではありません。現行optimizer契約は[`../optimizers.md`](../optimizers.md)とコードを参照してください。

## 主題と結果の読み方

Low-Rank Trajectory Drift Optimization (LRTDO)は、Schedule-Free trajectory driftやoptimizer stateを低rank表現で扱う研究テーマ名です。APOLLOのgradient projection `R_update`とSchedule-Free差分projection `R_delta`は別の状態・役割として扱います。

このアーカイブは主にTinyStoriesでのSchedule-Free low-rank、confidence variant、APOLLO-Confの比較記録です。ImageAE/CIFAR-10におけるAPOLLO系optimizerとprojection refreshの集計結果は[APOLLO実験結果](../apollo-experiment-results.md)を参照してください。taskと実験条件が異なるため、両資料の数値を直接比較しません。

残した診断・比較結果には短いTinyStories probe、rank/refresh sweep、prototype比較とAPOLLO候補比較が含まれます。結果はtask、seed、token budget、step数、dtype、実装状態に依存します。初期rank診断にはparameter dtype不一致があり、state-bytes測定は無効と注記されています。少数seedや短期probeから一般的なoptimizer優位性を結論しないでください。文書末尾のAPOLLO候補比較もrank 4の限定条件で、AdamW-SFの品質baselineや別taskへ一般化する根拠ではありません。

## 記録内容

| 結果 | 本文 |
| --- | --- |
| 初回rank診断（dtype不一致の制約を含む） | [初回診断結果](lrtdo-research-results.md#diagnostic-initial) |
| EMAによるdelta rankの変化 | [EMA付き診断の再実行結果](lrtdo-research-results.md#diagnostic-ema-rerun) |
| rank・refresh interval比較 | [rank・refresh interval sweep](lrtdo-research-results.md#rank-refresh-sweep) |
| refresh時のtransport誤差とoverlap比較 | [長期refresh transport診断](lrtdo-research-results.md#refresh-transport) |
| projected-gradient EMAの基準比較 | [低rank projected-gradient EMA](lrtdo-research-results.md#projected-gradient-ema) |
| innovation variance confidence | [初回結果](lrtdo-research-results.md#innovation-variance-confidence) |
| confidence付きLRSF prototype | [初期prototype](lrtdo-research-results.md#confidence-lrsf-prototype) |
| APOLLOとconfidence variantの比較 | [APOLLO比較](lrtdo-research-results.md#apollo-confidence-comparison) |
| confidence beta/alphaの感度比較 | [APOLLO-Conf sensitivity](lrtdo-research-results.md#apollo-conf-sensitivity) |
| APOLLO rank/scale探索と100-step比較 | [rank=4 LR・scale sweep](lrtdo-research-results.md#apollo-rank-scale-sweep) |
| AdamW-SFとのmatched 300-step比較 | [matched comparison](lrtdo-research-results.md#apollo-adamw-sf-300-step)、[rank=4候補の300-step結果](lrtdo-research-results.md#apollo-rank4-300-step) |
| 300-step trajectory診断 | [trajectory診断](lrtdo-research-results.md#apollo-trajectory-diagnostic) |

各結果の集計値、条件、限界はリンク先に残しています。ローカルoutputのJSONやcheckpointは公開treeに含めず、再利用する集計結果だけを記録しています。
