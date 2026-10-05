# Schedule-Free系optimizer研究記録

このページは比較記録への索引です。現行の数式・state・checkpoint契約は[`optimizers.md`](optimizers.md)とコードを参照してください。研究結果は記録した条件に限られ、現在の推奨や実装仕様を示すものではありません。

| 確認したい内容 | 参照先 |
| --- | --- |
| LRTDO / Low-Rank Schedule-Freeの診断・prototype・refresh比較 | [研究要約](history/lrtdo-research-summary.md)から読み、測定条件と数値は[詳細記録](history/lrtdo-research-results.md)を参照 |
| CAME-SF / CAME-LRSF / APOLLO-SPRの初期実装段階・比較記録 | [2026-09-13の研究履歴](history/low-rank-schedule-free-records-2026-09-13.md)。後続のLRTDO診断結果とは別の実装・probe記録 |
| APOLLO-SFのstate量子化比較 | [Mini-ImageNet GQA実験結果](mini-imagenet-gqa-results.md#optimizer-and-apollo-sf-quantization-comparison)：10 epoch・3 seedではINT8-DeltaはAPOLLO-SFに近く、INT4-Deltaは低下。条件と数値はリンク先を参照 |
| Low-Rank Schedule-Freeの数式と当時の設計根拠 | [2026-09-13設計snapshot](low-rank-schedule-free-design.md)。現行動作はコードを確認 |
| optimizerの現行名称、CLI、fallback、state契約 | [Optimizer仕様](optimizers.md) |

各結果のモデル、seed、step数、dtype、比較条件はリンク先で確認してください。過去の「next」や「candidate」は当時の記録です。
