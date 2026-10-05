# APOLLO experiment hypotheses and protocol notes

> 履歴資料です。以下の仮説・protocol要件は現在の作業指示ではありません。実測値は[APOLLO実験結果](apollo-experiment-results.md)と[LRTDO研究記録](history/lrtdo-research-summary.md)を参照してください。現行optimizerの動作・CLIは[optimizer仕様](optimizers.md)とコードが正です。

## 記録された仮説

- **低rank更新の制約:** 更新をprojectionすると、parameter更新に使える方向が変わります。state量の削減だけでは、品質の維持や探索性の向上は示せません。
- **Projection refresh:** basisの交換・回転で表現される更新が変わります。loss spikeや回復、よりflatな領域への移動は仮説であり、近似誤差や不安定性を増やす可能性もあります。
- **Loss-directed orthogonal refresh:** 当時のprototypeは実際のlossを評価せず、勾配行列`G`とbasis`R`から、射影勾配energy`||G R||_F^2`を増やすStiefel接空間上のproxy方向を使いました。

  ```text
  R_next = exp(eta * Omega) R_current
  Omega^T = -Omega
  ```

  このproxyはloss上昇やflatness改善を保証しません。validation lossを最適化する方式とは説明できず、gradientがない場合はrandomへfallbackせずエラーとします。

## 結果の読み方

完了したCIFAR-10/APOLLO比較は[APOLLO実験結果](apollo-experiment-results.md)、Low-rank Schedule-Freeの診断・refresh transport測定は[LRTDO結果記録](history/lrtdo-research-results.md)を参照してください。各記録にはdataset、seed、token/step budget、dtype、実装上の制約が記載されています。

この資料に方式や候補が載っていても、実行済みとは限りません。完了は結果資料に測定値が記載された範囲だけで判断してください。小規模probeや少数seedのscreenから、optimizer全般の優劣、basin脱出、flat minimaを結論しないでください。

## 比較時に残す条件

refreshを比較するときは初期weight、data order、seed、rank、learning-rate policy、precision、学習budgetを固定し、refreshだけを変えるfrozen-basis対照を含めます。品質と計算資源は分けて記録します。

| 区分 | 記録項目 |
| --- | --- |
| 品質 | train/validation loss、seed数、平均・ばらつき、必要に応じcheckpoint再読込結果 |
| Refresh挙動 | mode/state policy、intervalまたはrotation rate、event数、update norm/方向、event前後のloss |
| 資源 | optimizer永続state量、peak allocated/reserved memory、optimizer step時間 |

値のhost転送やstepごとのdiagnostic記録は時間・memoryへ影響するため、主な資源比較とは分けます。異なるminibatchで測ったloss差はrefreshの因果効果を示しません。sharpnessやHessian指標は補助診断として扱い、flat-minimaの証拠とはしません。
