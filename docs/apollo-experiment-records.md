# APOLLO experiment record index

> Run report paths mentioned in the historical records refer to local generated output and are not included in the public tree. Aggregate measurements are retained in [APOLLO experiment results](apollo-experiment-results.md).

> **Document type:** Index to historical APOLLO experiment results and protocols. Proposals describe their recorded context, not current work instructions. See [`optimizers.md`](optimizers.md) and the code for the current optimizer contract.

## 概要と読み方

AdamWはfull-rankの一次・二次momentを保持する比較基準です。APOLLOは行列勾配を低rank空間へ射影して適応状態を減らす設計です。projection refreshや低rank近似が探索性に寄与する可能性は仮説であり、Flat Minimaへの到達やAdamW一般を上回る性能を示したものではありません。

| 観点 | AdamW | APOLLO |
|---|---|---|
| 主なstate | full-rankの一次・二次moment | 低rank空間の補助stateとprojection |
| state memory | parameter数に比例 | rankとparameter shapeに依存 |
| 更新 | full-rank adaptive update | 低rank projectionを介した近似update |
| Refresh | 通常は不要 | variantによってprojection basisを定期更新 |

メモリ比較では永続optimizer state、一時tensor、activation、peak allocated/reservedを分けます。ImageAE/CIFAR-10の固定条件で候補を測定しましたが、5 epochの結果から一般的な優位性は主張できません。refreshの因果効果、複数seedやdataset規模での再現性、探索指標との関係も未確定です。

実測値は[APOLLO experiment results](apollo-experiment-results.md)と[optimizer性能snapshot](history/optimizer-review-2026-09-12.md)、仮説と過去の比較設計は[experiment protocols](apollo-experiment-protocols.md)を参照してください。現行実装・CLI・既定値は[`optimizers.md`](optimizers.md)とコードが正です。

## 記録の構成

- [実験結果と集計値](apollo-experiment-results.md): 測定条件と結果、および記録時点の判断
- [仮説と比較protocol](apollo-experiment-protocols.md): 探索性の仮説、候補方式、過去の比較設計
