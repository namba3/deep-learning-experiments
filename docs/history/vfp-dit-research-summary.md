# VFP-DiT research history summary

このcompact archiveは旧設計案・未実施のablation計画・容量見積もりを圧縮し、単一seedのQwen tap screeningとv0.1.36〜v0.1.47の実装・検証記録を残しています。履歴の提案を現在の実装仕様として使わないでください。現行仕様は[architecture note](../architecture/image-latent-dit.md)、[VFP-DiT README](../../vfp_dit/README.md)、コードを参照してください。

## 記録された結果と制約

- Qwen tap `{6,10,12,14,18,final}`のscreeningはseed 42、各1,024 updatesの単一seed比較です。layer 6のtotal lossが最小でしたが全候補の差は`0.001352`で、先行する1-epoch screeningではlayer 10が最小でした。layer選択や下流生成品質を確立する結果ではありません。
- v0.1.45ではjoint-KV用の独立FP32 referenceを追加しました。数値・gradient・CUDA/BF16・runtime比較は未完了でした。
- v0.1.47では共通layer抽出後、CPUでforward・gradient・partial-mask等を比較したと記録されています。CUDA/BF16と実runtimeの確認は未完了です。
- v0.1.46のcondition-dropout telemetryはseed 1でepoch 1=`0.10742`、epoch 4=`0.09961`。これは設定確率0.1に対するtelemetry sanity checkで、生成品質の評価ではありません。

## 保存している本文

- [主要設計変更と採否理由（revision別）](VFP-DiT_Research_History_v0.1.47.md#主要な設計判断revision別)
- [Qwen tap screeningの条件・集計値・限界](VFP-DiT_Research_History_v0.1.47.md#recorded-experiment-preliminary-qwen-tap-screening)
- [v0.1.36〜v0.1.47の実装・検証履歴](VFP-DiT_Research_History_v0.1.47.md#implementation-and-validation-history)

除いた旧提案の詳細は、現行architecture文書と実装を参照してください。履歴内のrun pathは当時の記録であり、データや成果物は公開treeに含みません。
