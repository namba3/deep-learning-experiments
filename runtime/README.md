# `runtime/`

複数の学習entrypointが共有する実行時補助機能を置くパッケージです。モデル層は`core/`、optimizerと学習率制御は`optimizers/`が担当します。このパッケージ自体に学習CLIはありません。

## 機能

- `preflight.py`, `validation.py`: 学習前の設定確認とdataset/model検証結果の整形
- `run.py`, `checkpoint.py`, `config.py`: run metadata、event記録、checkpointと設定の保存・復元
- `sampler.py`, `data.py`: 再開可能なsamplerとDataLoader workerのseed設定
- `progress.py`, `metrics.py`: 共通進捗表示とmetrics event
- `device.py`, `memory.py`, `profiling.py`: device選択、memory cleanup、任意の時間計測
- `signal.py`: Ctrl-C時の協調停止

学習entrypointから利用する共有APIです。変更時は呼び出し元のCLI、checkpoint/resume契約、既存run artifactへの影響を確認してください。
