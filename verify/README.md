# Runtime verification

ここは、通常の`pytest`に含めると重い、外部モデル・CUDA・実行時間に依存する検証を再実行可能な形で残す場所です。検証器は失敗したprobeのshapeと例外もJSONで出力します。

トップレベルの`verify/*.py`には、外部モデルやCUDAを使うruntime probeと、保存済みrun結果を集計するreport generatorがあります。`*_report.py`などのreport generatorは記録済み結果を読み取る後処理であり、それだけでは元の学習・性能測定を再検証しません。各測定の条件と集計結果は対応する`docs/`資料を参照してください。

### Offline report generators

以下は保存済みJSONやmetricsを読む後処理moduleです。学習・GPU計測を起動しません。

| Group | Modules | Related records |
| --- | --- | --- |
| APOLLO orthogonal, refresh, and variance-cap sweeps | `apollo_orthogonal_sweep_report.py`, `apollo_refresh_report.py`, `apollo_variance_cap_report.py` | [APOLLO results](../docs/apollo-experiment-results.md), [LRTDO recorded results](../docs/history/lrtdo-research-results.md) |
| LRSF GPU and refresh-recovery summaries | `lrsf_gpu_report.py`, `text_lm_lrsf_lr_refresh_recovery_report.py` | [Schedule-Free research records](../docs/history/low-rank-schedule-free-records-2026-09-13.md), [LRTDO recorded results](../docs/history/lrtdo-research-results.md) |
| Text-LM adapter and pretraining summaries | `text_lm_adapter_report.py`, `text_lm_pretraining_report.py` | [Adapter records](../docs/adapter-experiments/text-lm.md), [Pretraining comparison](../docs/text-lm-pretraining-comparison.md) |
| Text-LM optimizer, trajectory, residual, and APOLLO reports | `text_lm_optimizer_comparison_report.py`, `text_lm_optimizer_trajectory_diagnostics_report.py`, `text_lm_residual_approximation_report.py`, `text_lm_apollo_confidence_report.py`, `text_lm_apollo_scale_report.py` | [LRTDO recorded results](../docs/history/lrtdo-research-results.md) |

## Shell launcher index

検証手順は領域別のguideにまとめています。この一覧は`verify/launchers/`配下のscriptを収録しています。benchmark launcherは[`benchmarks/README.md`](../benchmarks/README.md)、package固有・archive launcherは各package READMEを参照してください。
次のshell launcherは、検証器の単発実行より大きなoptimizer・trajectory実験を起動します。多くはCUDAとTinyStories等のmodel/data cacheを使い、複数seed・複数条件では長時間かかります。通常のCPU testではなく、出力先を分けたうえで実行してください。既定の結果は `output/` 以下です（一部は環境変数で変更できます）。一覧への掲載は現在の推奨や実験の進行中を意味せず、再実行可能な検証presetを示します。完了状態・集計値は右欄の記録で判断し、launcherがあるだけで数値検証済みとは扱いません。

| 用途 | Launchers | 記録・状態の読み方 |
| --- | --- | --- |
| APOLLO projection/refresh/variance-cap実験 | [`run_apollo_orthogonal_sweep.sh`](launchers/run_apollo_orthogonal_sweep.sh), [`run_apollo_refresh_experiments.sh`](launchers/run_apollo_refresh_experiments.sh), [`run_apollo_variance_cap_sweep.sh`](launchers/run_apollo_variance_cap_sweep.sh) | [APOLLO集計](../docs/apollo-experiment-results.md)、[LRTDO記録](../docs/history/lrtdo-research-results.md)。記録時点と条件に限定した結果 |
| LRSF GPU validation | [`run_lrsf_gpu_validation.sh`](launchers/run_lrsf_gpu_validation.sh) | [Schedule-Free記録](../docs/history/low-rank-schedule-free-records-2026-09-13.md)、[LRTDO記録](../docs/history/lrtdo-research-results.md)。実行環境依存の検証preset |
| Text-LM APOLLO scale/state-rank実験 | [`run_text_lm_apollo_scale_sweep.sh`](launchers/run_text_lm_apollo_scale_sweep.sh), [`run_text_lm_apollo_state_rank_sweep.sh`](launchers/run_text_lm_apollo_state_rank_sweep.sh) | [LRTDO記録](../docs/history/lrtdo-research-results.md)。個別task/seedの研究結果 |
| Text-LM optimizer公平比較・residual approximation | [`run_text_lm_optimizer_fair_comparison.sh`](launchers/run_text_lm_optimizer_fair_comparison.sh), [`run_text_lm_residual_approximation_sweep.sh`](launchers/run_text_lm_residual_approximation_sweep.sh) | [LRTDO記録](../docs/history/lrtdo-research-results.md)。記載されたbudget内の比較 |
| Text-LM optimizer/state trajectory診断 | [`run_text_lm_optimizer_trajectory_diagnostics.sh`](launchers/run_text_lm_optimizer_trajectory_diagnostics.sh), [`run_text_lm_state_trajectory_pca.sh`](launchers/run_text_lm_state_trajectory_pca.sh), [`run_text_lm_causal_trajectory_pca_sweep.sh`](launchers/run_text_lm_causal_trajectory_pca_sweep.sh) | [LRTDO summary](../docs/history/lrtdo-research-summary.md)から対象・限界を確認。診断は品質比較ではありません |
| Text-LM LRSF learning-rate/refresh/recovery比較 | [`run_text_lm_lrsf_lr_refresh_diagnostics.sh`](launchers/run_text_lm_lrsf_lr_refresh_diagnostics.sh) (refresh挙動), [`run_text_lm_lrsf_lr_refresh_recovery.sh`](launchers/run_text_lm_lrsf_lr_refresh_recovery.sh) (recovery), [`run_text_lm_lrsf_lr_reset_sweep.sh`](launchers/run_text_lm_lrsf_lr_reset_sweep.sh) (rank/interval sweep), [`run_text_lm_lrsf_lr_speed_comparison.sh`](launchers/run_text_lm_lrsf_lr_speed_comparison.sh) (diagnosticなし速度比較), [`run_text_lm_lrsf_lr_trajectory_diagnostics.sh`](launchers/run_text_lm_lrsf_lr_trajectory_diagnostics.sh) (trajectory snapshot付きdiagnostic) | [Schedule-Free記録](../docs/history/low-rank-schedule-free-records-2026-09-13.md)、[LRTDO記録](../docs/history/lrtdo-research-results.md)。各protocolの個別条件を参照 |
| Text-LM confidence・Schedule-Free trajectory実験 | [`run_text_lm_lr_ema_confidence_sweep.sh`](launchers/run_text_lm_lr_ema_confidence_sweep.sh), [`run_text_lm_schedulefree_trajectory_curvature.sh`](launchers/run_text_lm_schedulefree_trajectory_curvature.sh), [`run_text_lm_schedulefree_trajectory_sweep.sh`](launchers/run_text_lm_schedulefree_trajectory_sweep.sh) | [LRTDO記録](../docs/history/lrtdo-research-results.md)。診断/trajectory結果であり一般的な推奨ではありません |

## Qwen Image VAE

ローカルcacheを使える環境では、次を実行します。

```bash
python3 -m verify.qwen_vae \
  --vae-model Qwen/Qwen-Image \
  --vae-dtype fp32
```

`--vae-model`にはHugging Face model idまたは`vae/`サブディレクトリを含むローカルモデルディレクトリを指定できます。`--device auto`はCUDAが使えるときCUDA、使えないときCPUを選びます。CPUでは`--vae-dtype bf16`を指定しても、実装上は互換性のためFP32になります。

既定では、次を確認します。

- `8x8`、`16x16`、`32x40`のencode shapeと有限値
- `image-size=256, bucket-step=32`で生成される全bucketの整数かつ一貫したlatent stride
- `16x16`、`32x40`のencode→decode shapeと有限値

境界入力だけを確認する場合:

```bash
python3 -m verify.qwen_vae \
  --vae-model Qwen/Qwen-Image \
  --probe-size 4x4 \
  --probe-size 8x8 \
  --skip-roundtrip
```

この例では`4x4`がVAEの畳み込み境界で失敗すること自体を記録します。そのため終了コードは非0になります。外部モデルのダウンロード、CPU/GPUの所要時間、BF16の実機精度は通常のunit testの合否に混ぜず、実行時のJSONと[repository audit snapshot](../docs/history/repository-audit-2026-09-12.md)で管理します。

## Qwen3.5 condition tap

Qwen3.5のinteger layer tapとfinal tapについて、早期終了の出力が全層forwardの対応出力と一致することを実モデルで確認します。既定はCPU、64×64のsynthetic source image、tap 6とfinalです。各tapでshape、mask、有限値、後続decoder blockのskip、final時のLM head skipもJSONへ記録します。referenceとして全層forwardを実行するため、早期終了forwardより時間がかかります。

~~~bash
HF_HUB_OFFLINE=1 PYTHONPATH=. python3 -m verify.qwen35_condition \
  --device cpu --dtype bf16 --tap 6 --tap final
~~~

別のtapは--tapを繰り返して指定します。CUDA上で実行する場合は--device cuda --dtype bf16を指定します。GPU学習中はCPUを使い、性能比較ではなくforward契約の検証として扱ってください。

VFP-DiTが必要とするT2I/TI2I双方のQwen hidden・mask・multimodal RoPE位置を実モデルで確認する場合は、次を実行します。プローブはT2Iとsynthetic source imageのTI2Iをencodeし、processorのattention maskとQwen画像gridから再構築した位置に一致することを確認します。

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=. python3 -m verify.vfp_dit_condition \
  --device cpu --dtype fp32 --tap final --image-size 64
```

VFP-DiTで使う元のQwen-Image VAE（2.1ではない方）のlatent正規化とsingleton-frame encode/decodeを確認する場合:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=. python3 -m verify.vfp_dit_vae \
  --device cpu --dtype fp32 --size 16x16 --size 32x40
```

実データと両モデルを通す最小training-step smokeは次で実行します。COCOとMultiEditから各1件を読み、tiny DiTでloss、全学習parameterのgradient、AdamW更新を確認します。これは配線確認用で、収束や既定optimizerの比較ではありません。

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. \
  python3 -m verify.vfp_dit_train_step --resolution 32 --threads 4
```

実Qwen3.5・元のQwen-Image VAEと未学習tiny DiTでT2I/TI2IのCFG samplerから画像tensorのdecodeまで通す場合:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=. python3 -m verify.vfp_dit_sampling \
  --resolution 32 --steps 2 --guidance-scale 2 --threads 4
```

このprobeはshape、有限値、出力範囲のみを確認します。tiny DiTは未学習なので、画像品質の結果ではありません。

## core kernelのTriton比較

CUDA環境では、core kernelのreference実装とTriton実装を、FP32またはBF16のforward/backward、peak allocated/reservedで比較できます。

```bash
python3 -m verify.core_kernels --dtype bf16
python3 -m verify.core_kernels --dtype fp32 --skip-backward
```

CUDAがない環境では`status=skipped`をJSON出力して終了コード2になります。未実行を成功扱いにしないため、CIの合否へ直接組み込む前に実行環境を確認してください。


## Runtime guide

- [モデル・VAE・kernel](model-runtime.md)
- [Optimizer runtime・convergence](optimizer-probes.md)
- [Adapter比較](adapter-comparisons.md)
- [ImageAE fixed-input/LRSF optimizer probe](image-ae-optimizer.md)
- [ImageAE CIFAR-10 optimizer comparison](image-ae-cifar10-optimizer.md)
