# `vfp_dit_runtime/`

VFP-DiTの学習entrypointが共有するdataset、Qwen feature抽出、training helperを置くパッケージです。独立した学習CLIではありません。モデル仕様と起動方法は[`vfp_dit/README.md`](../vfp_dit/README.md)を参照してください。

## Modules

- `data.py`: tensor manifestから学習例を読むdataset
- `hf_data.py`: Hugging Face画像・テキストsourceをVFP-DiT用のrecordへ正規化
- `qwen35.py`: Qwen3.5のsemantic・visual feature抽出
- `training.py`: VFP-DiT entrypointで共有する学習、checkpoint、validation helper

ここにある処理はVFP-DiTのデータ・encoder・training runtimeです。汎用model layerは[`core/`](../core/README.md)、一般的な学習run管理は[`runtime/`](../runtime/README.md)、flow samplerは[`flow_sampling/`](../flow_sampling/README.md)を参照してください。
