# Benchmarks and experiment launchers

このディレクトリには、再利用するbenchmark toolと、特定の実験条件を再現するshell launcherの両方があります。下記のlauncherは軽量なunit testではなく、trainingやsweepを起動します。実行前に各scriptの既定値と環境変数を確認し、必要なdataset/model、CUDA、出力先を用意してください。実行時間はhardware、cache、seed数、stepsにより大きく異なります。

## Python benchmark tools

次のPython moduleは個別launcherより再利用しやすい計測入口です。各moduleの`--help`でshape、device、dtype、warmupなどを確認してください。

| Tool | 対象 |
| --- | --- |
| [`benchmark_core_kernels.py`](benchmark_core_kernels.py) | core kernel backendのforward/backward時間とCUDA peak allocator使用量 |
| [`benchmark_dit_projection_fusion.py`](benchmark_dit_projection_fusion.py) | Q/K/VやSwiGLU projectionのseparate/packed Linear比較 |
| [`benchmark_gated_ffn.py`](benchmark_gated_ffn.py) | GatedFFNのPyTorch/Triton forward/backward比較 |
| [`benchmark_sdpa_attention.py`](benchmark_sdpa_attention.py) | PyTorch SDPAと明示的reference attentionの比較 |
| [`benchmark_text_lm_architectures.py`](benchmark_text_lm_architectures.py) | synthetic token batchによるdecoder architectureの時間・memory比較。language-model品質評価ではありません |

## Shell launcher index

すべてrepository rootから `bash <script>` で実行します。出力先は多くの場合 `output/` または各packageの `output/` で、環境変数により変更できるscriptもあります。出力dirを共有・上書きしないよう、比較ごとに別の場所を指定してください。

データセットをダウンロード・使用する前に、[データセットとモデルの出典・利用条件](../docs/data-model-provenance.md)を確認してください。特にTiny ImageNetのHub cardにはlicenseが示されておらず、利用条件は未確認です。launcherが用意されていることは、配布権や商用利用権の確認を意味しません。

| 実験 | Launcher |
| --- | --- |
| CIFAR-10 adapter budget比較 | [`run_cifar10_adapter_budget_comparison.sh`](launchers/run_cifar10_adapter_budget_comparison.sh) |
| CIFAR-10 adapter learning-rate sweep | [`run_cifar10_adapter_lr_sweep.sh`](launchers/run_cifar10_adapter_lr_sweep.sh) |
| CIFAR-10 RGLU-LoRA sweep | [`run_cifar10_rglu_lora_sweep.sh`](launchers/run_cifar10_rglu_lora_sweep.sh) |
| Tiny-ImageNet adapter比較 | [`run_tiny_imagenet_adapter_budget_comparison.sh`](launchers/run_tiny_imagenet_adapter_budget_comparison.sh), [`run_tiny_imagenet_adapter_long_comparison.sh`](launchers/run_tiny_imagenet_adapter_long_comparison.sh) |
| Mini-ImageNet GQA factorial comparison | [`run_mini_imagenet_gqa_comparison.sh`](launchers/run_mini_imagenet_gqa_comparison.sh) |
| Text-LM adapter baseline/budget/fixed-batch/learning-rate/sequence/instruction比較 | [`run_text_lm_adapter_base.sh`](launchers/run_text_lm_adapter_base.sh), [`run_text_lm_adapter_budget_comparison.sh`](launchers/run_text_lm_adapter_budget_comparison.sh), [`run_text_lm_adapter_fixed_batch_comparison.sh`](launchers/run_text_lm_adapter_fixed_batch_comparison.sh), [`run_text_lm_adapter_lr_sweep.sh`](launchers/run_text_lm_adapter_lr_sweep.sh), [`run_text_lm_adapter_sequence_comparison.sh`](launchers/run_text_lm_adapter_sequence_comparison.sh), [`run_text_lm_instruction_comparison.sh`](launchers/run_text_lm_instruction_comparison.sh) |
| Text-LM pretraining・optimizer・rank・refresh比較 | [`run_text_lm_pretraining_comparison.sh`](launchers/run_text_lm_pretraining_comparison.sh), [`run_text_lm_optimizer_comparison.sh`](launchers/run_text_lm_optimizer_comparison.sh), [`run_text_lm_optimizer_long_rank_comparison.sh`](launchers/run_text_lm_optimizer_long_rank_comparison.sh), [`run_text_lm_optimizer_lr_sweep.sh`](launchers/run_text_lm_optimizer_lr_sweep.sh), [`run_text_lm_optimizer_rank_sweep.sh`](launchers/run_text_lm_optimizer_rank_sweep.sh), [`run_text_lm_optimizer_refresh_comparison.sh`](launchers/run_text_lm_optimizer_refresh_comparison.sh) |
| Text-LM refresh smoke check | [`run_text_lm_optimizer_refresh_smoke.sh`](launchers/run_text_lm_optimizer_refresh_smoke.sh)（refresh経路の実行・有限値確認。品質・性能比較用ではありません） |
| VFP-DiT screen・learning-rate・memory・checkpoint・Ada scale比較 | [`run_vfp_dit_screen.sh`](launchers/run_vfp_dit_screen.sh), [`run_vfp_dit_lr_screen.sh`](launchers/run_vfp_dit_lr_screen.sh), [`run_vfp_dit_memory_profile.sh`](launchers/run_vfp_dit_memory_profile.sh), [`run_vfp_dit_checkpointing_comparison.sh`](launchers/run_vfp_dit_checkpointing_comparison.sh), [`run_vfp_dit_ada_scale_shift.sh`](launchers/run_vfp_dit_ada_scale_shift.sh) |

`run_vfp_dit_lr_screen.sh` requires an existing completed starting checkpoint; it has no repository-specific checkpoint default. Set `VFP_DIT_LR_SCREEN_INIT_CHECKPOINT` before running it:

```bash
INIT_CHECKPOINT="vfp_dit/output/runs/<RUN_ID>/checkpoints/checkpoint_latest.safetensors"
VFP_DIT_LR_SCREEN_INIT_CHECKPOINT="$INIT_CHECKPOINT" \
bash benchmarks/launchers/run_vfp_dit_lr_screen.sh
```

This launcher runs the current `vfp_dit.train` entrypoint. Use a completed
checkpoint compatible with the current VFP-DiT model configuration. The
historical [LR-screen results](../docs/experiment_data/vfp-dit-simple-lr-screen.md)
were produced by the retired `vfp_dit_simple` implementation from a 2,048-step
checkpoint; its checkpoint and implementation are not included in this
repository. Reusing a current `vfp_dit` checkpoint runs a new experiment and
does not reproduce or extend those historical measurements. Keep its results
separate from that record.

この一覧は`benchmarks/launchers/`配下のlauncherを収録しています。
`verify/launchers/`は[runtime verification guide](../verify/README.md)、package固有のlauncherは各package READMEを参照してください。
各benchmarks launcherは再利用可能な汎用CLIではなく、比較条件を再現する実験presetです。一覧への掲載は現在の推奨や実験の進行中を意味しません。過去の結果と採否は対応するpackage READMEまたは集計資料を参照してください。実験presetが存在することだけから、品質・性能の確認済みとは判断しないでください。
たとえば`run_text_lm_adapter_fixed_batch_comparison.sh`は共通sequence runnerへfixed-batch条件を渡し、
`run_tiny_imagenet_adapter_long_comparison.sh`はbudget runnerへ長期比較条件を渡します。
条件の違いを保つため別入口として維持しています。統合・削除する場合は、呼び出し元だけでなく各条件と過去の集計結果との対応を確認してください。
