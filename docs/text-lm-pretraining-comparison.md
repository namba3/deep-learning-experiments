# Text LM pretraining comparison

`text_lm`のdecoder architectureを、TinyStoriesの同一token budgetで短時間比較した記録です。実験結果のJSONは実行時に`TEXT_LM_OUTPUT_DIR`へ保存されます。

現在の新規実行で選択できるのは`naive`、`mhla3-gqa`、`looped`、`looped-hybrid`、`mhla3-gqa-looped-hybrid`です。以下に残る`shared-fixed` / `shared-variable`の行は過去比較の記録であり、アーカイブ実装としてのみ保持しています。

## CPU short probe

実行条件:

- dataset: `roneneldan/TinyStories`（HF cacheのArrow shardをoffline再利用）
- tokenizer: `Qwen/Qwen3.5-0.8B`
- device/dtype: CPU / FP32
- `embed_dim=32`, `num_layers=16`, `num_heads=4`, `kv_heads=2`
- train/eval: 4096 / 512 tokens
- 1 epoch、10 optimizer steps、batch size 1
- seed: `0,1,2`

21 runはすべて完走した。値は3 seedの平均で、`eval_loss_std`はseed間の標準偏差である。

| architecture | parameters | eval loss | eval loss std | CPU steps/s |
| --- | ---: | ---: | ---: | ---: |
| `mhla3-gqa-looped-hybrid` | 8,088,832 | 26.360 | 2.183 | 3.12 |
| `mhla3-gqa` | 8,138,944 | 26.668 | 1.075 | 2.78 |
| `looped-hybrid` | 8,072,616 | 26.697 | 1.568 | 3.70 |
| `naive` | 8,153,088 | 27.263 | 0.745 | 2.49 |
| `looped` | 7,951,908 | 27.788 | 0.870 | 3.01 |
| `shared-variable` | 8,795,144 | 27.820 | 1.114 | 1.87 |
| `shared-fixed` | 8,795,144 | 27.833 | 1.124 | 0.92 |

## Interpretation

このprobeでは`mhla3-gqa-looped-hybrid`のeval lossが最小だった。ただし、学習量は4096 tokens・10 steps、モデルも検証用の小型設定であるため、長期収束や実用的なモデル品質の順位とはみなさない。shared系はparameter数が多く、CPUでは特に遅かったが、GPU kernelとBF16では結果が変わる可能性がある。

CPUではCUDA peak memoryは計測されず、VRAM欄は比較不能である。また、TinyStoriesのstream先頭を使う短期probeなので、データ全体の品質評価にも使わない。

## OpenWebText preliminary probe

OpenWebTextでも同じ設定をseed=`0,1,2`・10 stepsで実行した。全21 runが完走し、streamingで取得したtrain先頭の4096 tokensと続くeval 512 tokensを使用した。値は3 seedの平均で、標準偏差は次の通りである。

| architecture | eval loss | eval loss std | CPU steps/s |
| --- | ---: | ---: | ---: |
| `mhla3-gqa-looped-hybrid` | 27.131 | 2.131 | 3.46 |
| `looped-hybrid` | 26.938 | 2.237 | 3.23 |
| `naive` | 27.014 | 1.579 | 3.40 |
| `shared-variable` | 27.323 | 1.630 | 1.82 |
| `shared-fixed` | 27.332 | 1.615 | 1.31 |
| `mhla3-gqa` | 27.452 | 0.186 | 2.90 |
| `looped` | 27.684 | 0.402 | 3.07 |

これは短期値であり、TinyStoriesとのデータセット優劣やarchitectureの一般的な品質順位を示さない。長い学習、GPU/BF16での再検証を残課題とする。

## BF16 CPU smoke

BF16の実行経路をTinyStoriesの全7構成で確認した。`embed_dim=32`、`num_layers=16`、seed=0、10 steps、train/eval=1024/128 tokensの条件で全runが完走し、全parameterのstorage dtypeはBF16、logitsはFP32だった。BF16 Linearが返すFP32 activationとBF16 RMSNorm weightの境界も警告なしで通過した。

この環境のCPUではBF16 kernelが高速とは限らないため、以下の速度はGPU性能の代替ではない。

| architecture | parameter storage | eval loss | CPU steps/s |
| --- | ---: | ---: | ---: |
| `looped` | 15.17 MiB | 28.268 | 3.92 |
| `naive` | 15.55 MiB | 27.382 | 3.83 |
| `looped-hybrid` | 15.40 MiB | 26.596 | 3.06 |
| `mhla3-gqa-looped-hybrid` | 15.43 MiB | 27.774 | 2.53 |
| `shared-fixed` | 16.78 MiB | 27.901 | 1.60 |
| `shared-variable` | 16.78 MiB | 27.889 | 1.47 |
| `mhla3-gqa` | 15.52 MiB | 26.642 | 1.34 |

## Distillation convergence probe

Qwen3.5-0.8Bのlogits蒸留を、`naive`、`looped-hybrid`、`mhla3-gqa-looped-hybrid`で比較した。TinyStories、CPU、BF16 student、FP32 logits、temperature=2.0、alpha=0.5、seed=0、50 steps、train/eval=8192/1024 tokensの条件である。

| architecture | eval loss | CPU steps/s |
| --- | ---: | ---: |
| `mhla3-gqa-looped-hybrid` | 26.438 | 2.64 |
| `looped-hybrid` | 27.009 | 2.79 |
| `naive` | 27.271 | 2.70 |

`mhla3-gqa-looped-hybrid`がこの条件では最小lossだった。ただし、studentは検証用の小型設定で、seed=0のみかつ50 stepsである。teacherのreference kernel fallback通知は残っているが、student側のRMSNorm dtype mismatch警告は発生していない。実用的な品質・速度・VRAMの判断には、CUDA/BF16、複数seed、長期学習が必要である。

## Candidate convergence probe

短期probeの次段として、TinyStories・CPU・FP32・3 seed・50 steps・train/eval=8192/1024 tokensで、7構成を比較した。モデル設定は`embed_dim=32`、`num_layers=16`、`num_heads=4`、`kv_heads=2`、batch size 1である。

| architecture | parameters | eval loss | eval loss std | CPU steps/s |
| --- | ---: | ---: | ---: | ---: |
| `naive` | 8,153,088 | 22.845 | 0.687 | 2.75 |
| `looped-hybrid` | 8,072,616 | 23.120 | 0.969 | 2.70 |
| `mhla3-gqa-looped-hybrid` | 8,088,832 | 23.671 | 0.163 | 2.04 |
| `mhla3-gqa` | 8,138,944 | 23.920 | 0.789 | 2.17 |
| `shared-fixed` | 8,795,144 | 25.142 | 0.606 | 1.01 |
| `looped` | 7,951,908 | 25.227 | 0.407 | 1.63 |
| `shared-variable` | 8,795,144 | 25.528 | 0.613 | 1.22 |

10 stepの短期probeで最小だった`mhla3-gqa-looped-hybrid`ではなく、50 stepでは`naive`が最小になった。したがって、現時点のCPU小型probeから構成の優劣を固定せず、GPU/BF16と十分なtoken budgetで再比較する。

## Candidate convergence probe: 100 steps

50 step probeの再現性確認として、現行の主要3構成をTinyStories・CPU・FP32・3 seed・100 steps・train/eval=16384/2048 tokensで比較した。モデル設定は前節と同じ`embed_dim=32`、`num_layers=16`、`num_heads=4`、`kv_heads=2`、batch size 1である。表はseed間の平均と標準偏差で、PPLはhard CEから算出した監視値である。

| architecture | parameters | eval loss | eval loss std | eval PPL | CPU steps/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `naive` | 8,153,088 | 20.663 | 0.559 | 1.03e9 | 2.14 |
| `looped-hybrid` | 8,072,616 | 20.734 | 0.363 | 1.05e9 | 1.98 |
| `mhla3-gqa-looped-hybrid` | 8,088,832 | 21.233 | 0.040 | 1.67e9 | 2.07 |

この条件では`naive`がeval lossとPPLの平均で最小だった。`looped-hybrid`との差はeval loss 0.071に留まり、seed間標準偏差の範囲も考慮すると明確な優位とは言い切れない。一方、`mhla3-gqa-looped-hybrid`はlossが高いもののseed間のばらつきは最小だった。これは小型CPU probeの結果であり、GPU kernel、BF16、十分な学習token数、実用規模モデルでの順位を示すものではない。

集計値と実行条件は、この比較節の結果とrun記録に示す。

## Sequence length sweep

`mhla3-gqa`の長文での挙動を確認する場合は、`TEXT_LM_MAX_SEQ_LENS`に複数のblock長を指定する。`max_seq_len`はrun名と`report.json`の集計キーに含まれるため、128・1024・4096の結果が混ざらない。

```bash
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
TEXT_LM_MAX_SEQ_LENS=128,1024,4096 \
TEXT_LM_ARCHITECTURES=naive,mhla3-gqa,looped-hybrid,mhla3-gqa-looped-hybrid \
TEXT_LM_DEVICE=cuda TEXT_LM_BF16=1 \
bash benchmarks/launchers/run_text_lm_pretraining_comparison.sh
```

系列長を変える場合は、同じoptimizer step数になるよう`max-train-tokens`と`max-eval-tokens`を十分大きく設定する必要がある。たとえば`4096`では、`16384` train tokensは4 blockにしかならないため、評価の分散が大きくなりやすい。このsweepでは`mhla3-gqa`を必ず含め、短文のlossだけで候補を落とさない。

## GPUでの再実行

Qwen tokenizerとTinyStoriesがcache済みのCUDA環境では、次で同じ比較を実行できる。

```bash
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
TEXT_LM_DEVICE=cuda TEXT_LM_BF16=1 \
TEXT_LM_DATASETS=roneneldan/TinyStories \
TEXT_LM_SEEDS=0,1,2 \
bash benchmarks/launchers/run_text_lm_pretraining_comparison.sh
```

OpenWebTextを加える場合は`TEXT_LM_DATASETS=roneneldan/TinyStories,Skylion007/openwebtext`とする。未取得のデータセットをoffline modeで指定すると失敗するため、先にcacheを用意する。

結果は`output/text-lm-pretraining-comparison/report.json`に保存され、parameter数、loss、step速度、CUDA peak allocated/reservedを構成ごとに集計する。

蒸留ありの比較では、次の環境変数を追加する。run名には`distill-logits`が付くため、通常比較と同じoutput directoryでもログ名が衝突しない。

```bash
TEXT_LM_DISTILL_MODE=logits \
TEXT_LM_TEACHER_MODEL=Qwen/Qwen3.5-0.8B \
TEXT_LM_DISTILL_TEMPERATURE=2.0 \
TEXT_LM_DISTILL_ALPHA=0.5 \
bash benchmarks/launchers/run_text_lm_pretraining_comparison.sh
```

上記設定はdry-runでconfigへの反映を確認し、TinyStories・`naive`・1 stepのCPU smokeではQwen3.5 teacherのロード、BF16 student、`report.json`生成まで確認した。
