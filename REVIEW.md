# コードレビューガイドライン

この文書は、AIコードレビューエージェントがこのリポジトリの設計・数値仕様・検証範囲を踏まえて、実用的な指摘を行うための基準です。通常の実装・検証作業には [`AGENTS.md`](AGENTS.md) を適用してください。

## Review Principles

- 数学的正しさ、セキュリティ、信頼性を優先し、実際に起きる不具合や回帰を指摘する。
- 差分とその呼び出し元・呼び出し先を確認し、問題の根拠と再現条件を示す。
- 推測だけのリスク、スタイルの好み、実益の乏しい改善案をfindingにしない。
- 実装と文書に記録された設計意図・実験上の制約を確認してから判断する。
- 実行していないテストやGPU検証を、確認済みの根拠として扱わない。

## Review Priorities

優先順位は次のとおりです。

1. **Correctness** — 数式、shape、軸の意味、mask、勾配、学習・samplingの時間方向
2. **Security** — 外部入力・checkpoint・外部データ連携に起因する具体的な危険
3. **Reliability** — 例外、有限値、checkpoint保存・再開、割り込み時の整合性
4. **Concurrency and resource safety** — DataLoader、非同期処理、CUDA memory、worker/thread終了
5. **Performance** — 実測可能なstep時間、kernel時間、peak allocated/reserved、optimizer state
6. **Compatibility** — CLI既定値、checkpoint metadata/state、sampler、既存import経路
7. **Maintainability** — 共通APIの契約、実験設定・結果・説明の整合性

## Project-Specific Review Guidelines

### 構成と変更範囲

- `core/`は共有model/tensor処理、`runtime/`は学習時の進捗・memory・device等、`optimizers/`はoptimizerとschedulerを担当します。共通処理がtrain scriptに依存していないか、実験固有処理が共有層へ漏れていないか確認します。
- MNIST/CIFAR-10等の基本実験、text LM、image AE、image-latent DiT、VFP-DiTは異なる契約を持ちます。該当するパッケージREADMEと`docs/architecture/`の仕様を参照し、別実験の仮定を持ち込まないでください。
- `docs/history/`や実験レポートは特定時点の記録です。現行コードやCLIと食い違う場合、履歴文書だけから現在の挙動を断定しないでください。

### 数値・shape・勾配

- 画像は通常`(B,C,H,W)`、tokenは`(B,T,D)`、attention内部は`Q=(B,Hq,T,Dh)`・`K/V=(B,Hkv,S,Dh)`です。reshape/permute後も各軸の意味、flatten順、復元順が一致するか確認します。
- GQAでは`Hq % Hkv == 0`、head分割では`D % num_heads == 0`を確認します。2D RoPEの座標pair、Q/Kのsequence length、maskのbatch/query/key軸、全masked行の処理を個別に確認します。
- 矩形画像、最小サイズ、stride/window境界、非倍数解像度、channel数やhead数の境界でdownsample/upsample、token数、mask offsetが保たれるか確認します。
- BF16/FP32の境界ではparameterとactivationのdtype、reduction/accumulation dtype、device、contiguous条件を確認します。出力がfiniteであることだけで、数値等価性や勾配の正しさを結論しないでください。
- Flow Matching/Rectified Flowでは、`x_t`、velocity target、samplerの時間方向を同じ定義で追います。現在のimage-latent設計では`x_t=(1-t)x0+t x1`、`v=x1-x0`を使い、samplingはnoiseの`t=1`からdataの`t=0`へ進みます。別solverやwarpを変更する場合は対応する設計資料も確認します。

### Reference、Triton、optimizer

- Triton kernelは独立した数値実装として扱います。PyTorch/referenceとのforward値・backward勾配、mask、境界shape、dtype、stride/contiguous、fallbackを確認します。CUDA/Triton未実行のCPU結果からGPU一致や高速化を推定しないでください。
- kernel最適化やoptimizer backend変更は、更新順序、FP32等のaccumulation、state、weight decay、projection/refresh、1D fallbackを追います。選択されたbackendがmoment/state更新からparameter更新まで二重適用・部分適用になっていないか確認します。
- 性能改善の主張には同条件のreference比較が必要です。forward単独ではなくbackward、初回compile/autotune、step時間、peak allocated/reserved、state memoryを分け、dtype・batch・矩形shapeを照合します。測定のない特殊kernel追加を性能改善として扱いません。

### 学習状態、checkpoint、互換性

- architectureや学習挙動の変更ではCLIの既定値、保存metadata、resume時の設定検証、sampler/RNG復元、optimizer/scheduler stateをセットで確認します。model weightだけの`--init-checkpoint`とoptimizer等を復元する`--resume`を混同しないでください。
- strictなstate dictロードやnetwork/config version確認を弱めて、不一致を黙って許容していないか確認します。互換性を意図的に変える場合は、旧checkpointを拒否する条件と初期化checkpointへの移行経路が明確か確認します。
- samplerを途中再開する場合、保存位置がsample単位かbatch単位か、epoch境界・worker数・乱数状態と整合するか確認します。割り込み時は現在のoptimizer stepと保存stateの境界が一致している必要があります。
- CLI変更は`--help`、metadata、resume時の既定値継承、関連generate/export/launcher scriptとの整合性を確認します。歴史的なimport pathを互換再exportしている場合は維持を確認します。

### 外部データ、checkpoint、再現性

- Hugging Face等のdataset/model IDやローカルrecordを扱う変更では、ロード対象、cache/path、例外時の挙動を確認します。取得先や利用条件を新たに広げる場合、コードライセンスが外部データ・modelの条件を置き換えると仮定しないでください。必要に応じて[`docs/data-model-provenance.md`](docs/data-model-provenance.md)を参照します。
- `safetensors`のmodel weightと、`torch.load(..., weights_only=False)`を使うtraining-state sidecarは安全性が同一ではありません。信頼できないcheckpointを読み込める経路が追加・拡大されていないか、入力の出所を踏まえて具体的に評価します。
- 再現性・品質の比較ではseedだけでなくdataset split、model/dataset revision、設定、初期checkpoint、sampler、実行数が揃っているか確認します。異なる条件の結果からoptimizerやarchitecture単独の効果を主張していないか見ます。
- run出力・生成画像・checkpointを追跡対象へ加える変更では、個人情報、ローカル絶対パス、credential、外部素材の利用条件を確認します。実際の危険や追跡差分がある場合に限り指摘します。

### 検証範囲の読み方

- CIの標準suiteはCPU unit/integration、compileall、Pyright、Ruffです。既定CIに含まれないCUDA/Triton・性能・実データ/model downloadの挙動は別途検証が必要です。
- Pyrightの対象は主に`core/`、`optimizers/`、`tests/unit/`です。学習script全体の型安全性がCIで保証されているとは扱いません。
- CPUテストはGPU kernelの数値一致や速度を保証しません。compileallは構文確認であり、import/runtime成功の証拠ではありません。findingは変更の影響する検証層に合わせてください。

## Findings and Severity

既存の明示的なレビュー出力規約がない場合、次の優先度を使います。

- **P0 Critical** — 広範なデータ損失・破壊、秘密情報の重大な漏えい、主要機能を安全に使えない重大な欠陥。通常の利用条件で緊急対応が必要。
- **P1 High** — 主要な学習・生成経路で再現する誤った結果、checkpoint破損/誤復元、深刻なsecurity/reliability問題。回避策が限られる。
- **P2 Medium** — 限定された設定・入力で起きる実質的な不具合、数値/互換性回帰、妥当性が裏付けられた性能問題。
- **P3 Low** — 影響範囲が小さく、正しさを直ちに損なわない保守性・文書整合性の問題。具体的な利用影響がある場合に限る。

findingは原則として一件一問題とし、変更差分内の最小の関連箇所を示してください。ファイルと行番号、問題と根拠、発生条件、影響、可能な修正方針を簡潔に含めます。優先度を影響に合わせ、重複指摘や単なる要約を加えません。裏付けが足りない点はfindingと断定せず、確認事項として不確実性を明示します。

## Review Exclusions

- Ruff等で機械的に検出できる軽微なformat/lint指摘。ただしCI失敗や実動作への影響を伴う場合を除きます。
- 変更から到達可能性や発生条件を示せない潜在的リスク。
- 個人的な命名・書式・抽象化の好み。
- 変更差分と関係しない既存問題、重複する指摘、利用影響のないリファクタリング提案。
- 実測のない速度・VRAM改善要求や、実験仮説だけを理由とする品質断定。

## Review Behavior

- レビュー中にソースコードを変更しません。
- 問題の説明と改善案を分け、改善案を出す場合は原因に対応させます。
- 差分だけで判断できない場合は、呼び出し元/先、関連テスト、設定、設計資料を確認します。
- テストやGPU実行ができない場合は、その制約と結論への影響を明記します。
- findingがなければ、簡潔に問題を確認できなかった旨と、確認範囲上の重要な制約だけを返します。
