# VFP-DiT historical research record v0.1.47

> Historical snapshot, latest recorded revision v0.1.47. This compact archive keeps recorded screening results and implementation/validation history. Earlier architecture proposals, unexecuted ablation plans, speculative scaling memos, and preflight checklists have been compressed out. The removed sections did not establish completed experiments. For current architecture and CLI behavior, use [`../vfp_dit/README.md`](../../vfp_dit/README.md), [`../architecture/image-latent-dit.md`](../architecture/image-latent-dit.md), and the code.

## 主要な設計判断（revision別）

次表は、旧版に散在していた主要な設計判断をrevision単位で圧縮したものです。元記録に信頼できる年月がないため、暦月は割り当てていません。「参照案に採用」は当時の仕様上の選択を指し、比較実験による優位性の確認を意味しません。測定記録の日付・条件は後続節に残しています。現在の仕様はREADME・architecture文書・コードで確認してください。

| revision | 変更・判断 | 理由と記録上の採否 |
| --- | --- | --- |
| v0.1.0 | VAE latentと空間整合したvisual featureをchannel方向に結合し、単一のMain DiTへ入力するVFP-DiTを定義。別個のDesigner–Predictor分解は参照案から外した。 | 研究の中心を「視覚特徴のchannel pre-fusion」に絞る簡素な構成を選択。設計仮説であり、分解案との品質比較結果はない。 |
| v0.1.1–v0.1.3 | noisy latentからteacher visual featureを予測するVFCB、MSE・direction・relational loss、timestep依存の重み付けを導入。teacher/alignerを固定し、teacherなしend-to-end branchをcontrolに追加。 | teacher targetの移動による混同を避け、teacher feature supervisionの寄与を切り分ける設計。これらは訓練計画上の選択で、品質比較による確定ではない。 |
| v0.1.5–v0.1.7 | static conditioning prefix内をjoint-bidirectionalにし、Qwen風block-causal maskはablationへ移した。 | condition prefixをtarget timestepから独立させ、prefix K/V再利用を保ちながらtarget queryが全conditionを参照できる構成を参照案に選択。mask比較での実証ではない。 |
| v0.1.17–v0.1.20 | attentionのpost-SDPA gateをfactorized `sigmoid × tanh`からsingle `SiLU`へ変更し、FFN residual gateをsigned `tanh`から`2 × sigmoid`へ変更。 | attention gateはsigned correctionを許す単一のwrite-strengthとして単純化。FFNの更新方向はSwiGLU側に任せ、gateはunit gainを中心とした正の書き込み量に限定する設計。activation/gateの比較結果による採否ではない。 |
| v0.1.22–v0.1.24 | GQAを効率上の参照実装に採用。visual tokenは`[512-d latent-derived ; 512-d feature-derived]`とし、旧`576 -> 1024` concat-then-project案をlegacy controlへ。latent側だけをliftし、2 branchを別契約でnormalize。 | GQAはK/V projection・cache・memory trafficを減らすための実装選択。latent liftingとbranch別normalizationは、latentとteacher featureの表現を入力境界で識別可能に保つための設計判断。いずれも品質上の優位性はこの履歴では実証されていない。 |
| v0.1.25–v0.1.27 | VFCBをQwen intermediate hiddenでconditionし、VFCB/Main DiTのtrainable adapterとmeta embeddingを分離。 | intermediate hiddenを共有sourceとして使いつつ、feature predictionとgenerationの学習役割を分け、VFCB単独pretrainingとstatic condition K/V再利用を可能にする設計。中間tap自体は後のscreeningでも選定に至っていない。 |
| v0.1.33–v0.1.35 | VFCBのsemantic cross-attentionを旧2/4 blockから4/4 blockへ拡張し、各blockでself/cross-attentionのsoftmaxとresidual writeを分離。`512 × 4`は最適値ではなくbring-up用referenceと再定義。 | joint-softmaxではvisual/semantic keyが同じattention probability massを奪い合うため、当時の参照案は別々のattention budgetを選択。これは設計上の理由であり、Topology A/Bの実測比較ではない。容量・tapの段階的探索も提案段階。 |
| v0.1.35–v0.1.41 | Qwen tap screeningとmatched multi-seed sweep用の診断・実行基盤を整備。 | 初期screeningではlayer 6が最低lossでしたが、候補間差は`0.001352`、先行する1-epoch screeningではlayer 10が最低。結果が再現せず、複数seedの完了集計もないため、tapは選定せずdynamic layer mixingも参照案にしなかった。実測条件は次節に記録。 |
| v0.1.42–v0.1.47 | projection costを減らすjoint-KV topologyを比較候補として再導入し、独立FP32 referenceを追加。shared layerを`core.layers`へ抽出。 | joint-KVは別softmaxという旧判断を置き換える採用決定ではなく、数値・gradient・runtime比較用の候補。CPUのreference比較は記録されたが、CUDA/BF16・runtime acceptanceが未完了なので既定Topology Aは変更せず。core抽出も構造整理であり、architecture採否を変えていない。 |

## Recorded experiment: preliminary Qwen tap screening

The following is a short, single-seed training-loss screening. It did not establish a selected Qwen layer or downstream generation quality.

### Preliminary implementation screening (not a layer-selection result)

The initial HF bring-up compared one-based decoder-block outputs at layers `{6, 10, 12, 14, 18}` and the post-final-RMSNorm `final` tap with the VFCB held at `512 x 4`. Every candidate used seed 42, batch size 1, 512 replacement-sampled examples per epoch for 2 epochs (1,024 updates), 512px resolution, BF16, APOLLO, the same teacher transform and 1:1 COCO/MultiEdit sampler weights. The sampler and DataLoader generators are reseeded by epoch, so candidate runs use the same sampled example order. The reported values are the final epoch's training averages:

| Qwen condition tap | Total loss | MSE | Direction | Relational |
|---:|---:|---:|---:|---:|
| 6 | 1.381191 | 0.878679 | 0.939674 | 0.765625 |
| 10 | 1.381793 | 0.878595 | 0.940666 | 0.765625 |
| 12 | 1.381675 | 0.878408 | 0.940655 | 0.765625 |
| 14 | 1.381565 | 0.878385 | 0.940509 | 0.765625 |
| 18 | 1.382543 | 0.878380 | 0.942000 | 0.765625 |
| `final` (post-final-norm) | 1.382013 | 0.878507 | 0.941079 | 0.765625 |

Layer 6 has the lowest aggregate loss in this run, but the full spread is only `0.001352`; layer 18 is highest and `final` is close to both. The initial one-epoch screening instead ranked layer 10 lowest, so the apparent winner did not reproduce. This short, single-seed training-loss comparison is only a pipeline/screening check. It does not establish a useful representation ranking or downstream generation quality. Do not lock the condition layer from these numbers. Held-out validation, timestep/SNR-binned diagnostics, sampled feature-statistics/representation metrics, and their checkpoint logging are now implemented. Formal Phase A still needs repeated-seed runs, throughput/VRAM measurements and downstream generation checks; condition-dropout degradation is now measured by a paired null-condition validation pass. The sampled metrics do not establish layer quality by themselves.

The matched multi-seed runner was documented at the time as `benchmarks/run_vfp_dit_qwen_layer_sweep.sh`; that launcher is no longer present in the repository. Its recorded defaults were taps `{6, 10, 12, 14, 18, final}`, seeds `{0,1,2}`, 10 epochs, 1,024 train samples/epoch, and 256 held-out validation samples. It was intended to launch one run per tap/seed and write a JSON/Markdown report with completed-run mean/std plus missing cells. This was an executable Phase A comparison harness at the time; no completed repeated-seed result was claimed in this snapshot.


A potentially important diagnostic is whether the best Qwen layer changes with noise level. For example, earlier layers may preserve more local/multimodal detail while deeper layers may supply stronger semantic abstraction. Such a result should first be recorded as a diagnostic; dynamic layer mixing is **not** introduced into the reference until a single-layer baseline is well established.

## Implementation and validation history

Historical terms such as “current” and “next” refer to this snapshot. Instrumentation and protocol descriptions are summarized here; they are not evidence of completed model-quality comparisons.

- **v0.1.36–v0.1.39:** added timestep/SNR-binned loss diagnostics, sampled feature statistics, paired null-condition validation, and throughput/allocator telemetry. These metrics support analysis but do not establish generation quality or select a Qwen layer.
- **v0.1.40–v0.1.41:** added matched Qwen-tap sweep infrastructure and recorded run-code provenance. No completed repeated-seed aggregate result was recorded; throughput from a run started before runtime changes was excluded from comparison.
- **v0.1.42–v0.1.44:** specified a joint-KV attention comparison and its acceptance checks. The comparison remained open; CUDA/BF16 correctness and runtime were unverified.
- **v0.1.45:** added an independent FP32 joint-KV reference. Numerical, gradient, CUDA/BF16, and runtime comparisons were still outstanding.
- **v0.1.46:** condition-dropout telemetry recorded `0.10742` in epoch 1 and `0.09961` in epoch 4 for tap 14, seed 1. This is a telemetry sanity check against configured probability `0.1`, not a quality result.
- **v0.1.47:** extracted shared layers under `core.layers` with compatibility re-exports. CPU tests compared the shared layer and VFP wrapper against explicit FP32 attention for forward values, gradients, masks, mass statistics, and imports. CUDA/BF16 and runtime checks remained open.
