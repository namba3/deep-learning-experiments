# Repository Guidelines

このリポジトリは、AI・深層学習への理解を深めるため、PyTorchでモデルアーキテクチャや学習手法を実装・検証する実験用コードです。画像モデル、autoencoder、Transformer、optimizerなどを題材にしています。

## 作業方針

- テンソル計算は単なる実装詳細ではなく、モデルの数学的仕様そのものとして扱う。shape、reshape、transpose、broadcast、reductionの各境界で、式上の添字と実際の要素対応が一致することを確認する。
- 数式を変更・最適化する場合は、まずreference式を明記し、最適化後の式が同値か近似かを分類する。同値でない場合は、許容誤差・近似対象・学習への影響を記録する。
- Tritonは高速化の補助実装ではなく、独立した数値実装として扱う。PyTorch/reference、Triton、fallbackの更新順序・accumulation dtype・mask・境界条件を一致させる。
- 変更の評価軸は、数学的妥当性、メモリ効率と性能、再利用可能性、保守性の4点とする。性能だけを理由に数値仕様や検証可能性を犠牲にしない。
- 既存の学習挙動を変える修正は、CLIのデフォルト、checkpoint metadata、sampler復元処理を確認してから行う。
- 数値計算の変更では、テンソル形状、dtype、device、broadcast、mask、勾配経路を確認する。
- `image_gen` と `image_ae` の共有可能な処理は、必要に応じて `core` へ抽出する。ただし、coreから大きな学習スクリプトへ依存しない。
- 共通化では、単にコード行数を減らすのではなく、入力契約、出力契約、dtype/device契約を明示した小さなAPIを優先する。
- メモリ最適化では、peak allocated/reserved、保存activation、optimizer state、再計算による速度低下を分けて測定する。VRAM削減だけでなく、step時間と数値結果も比較する。
- kernel最適化では、reference実装を維持し、最適化版を差し替え可能にする。実測で効果が確認できない特殊kernelは増やさない。
- Triton化した処理は通常のPyTorch実装と別実装として扱い、数値、勾配、mask、境界shape、dtype、contiguous条件、fallback動作を個別に検証する。Tritonが速くても、referenceとの一致が確認できるまで標準backendにしない。
- Triton kernelでは、forwardだけでなくbackward、autotune初回遅延、workspace/peak VRAM、非対応入力時のエラーまたはfallbackを記録する。
- CUDA/Tritonを実行できない環境では、CPUでのshape・forward/backward・数値等価性を確認し、GPUで未検証であることを明記する。
- 既存のユーザー変更を上書きしない。変更範囲は依頼対象に限定する。
- Gitへの書き込み権限があり、作業ツリーとindexの状態を確認したうえで安全に対象変更だけを選べる場合は、依頼された作業を完了後にcommitする。無関係な既存変更はcommitに含めず、権限不足やcommit失敗時は理由を報告する。

## 公開前の個人情報・ローカル情報

- 公開対象のソース、設定、ログ、画像メタデータ、文書、ファイル名に、個人名、ユーザー名、メールアドレス、認証情報、ホスト名、個人を特定できるIDを含めない。
- Unixのhome/mount path、macOSのuser directory、Windowsのuser profileなど環境固有の絶対パスを、追跡ファイルや公開文書に記録しない。必要なパスはプロジェクト相対パスか、`<LOCAL_PATH>` のような汎用placeholderで表す。
- 学習ログ、生成config、benchmark出力を保存・追加する前に、個人情報、秘密情報、絶対パスを確認して除去する。調査結果を文書化するときも、検出した秘密値そのものを出力しない。
- 公開前に追跡対象全体をテキスト検索し、上記の情報とcredentialらしい値を確認する。バイナリ成果物を含める場合は、metadataや埋め込み文字列も確認し、公開に不要なものは含めない。
- Git履歴には削除後の情報が残る。公開用リポジトリを作る場合、履歴・tag・remoteを含めて公開対象を確認し、個人情報を含む既存履歴をそのまま公開しない。新規履歴のcommit identityにも公開用メールアドレスを使う。
- `.gitignore` でdataset、checkpoint、run output、cache、credential fileを除外し、公開に必要な小さな再現用例だけを明示的に追跡する。

## レビュー観点

各変更では、次の質問に答えられる状態にする。

1. 数式、shape、dtype、mask、勾配は意図した仕様と一致しているか。
2. peak VRAM、step時間、kernel時間、optimizer stateのどれが改善または悪化したか。
3. 他のtrain scriptやsamplerから再利用できるAPIになっているか。
4. checkpoint、resume、CLI、テスト、ドキュメントを含めて保守できるか。

## 検証コマンド

基本的な検証には次を使用する。

```bash
PYTHONPATH=. python3 -m pytest -q
bash scripts/compile_python.sh
python3 -m pyright
ruff check .
git diff --check
```

CUDA/Tritonの変更では、利用可能な環境で次も実行する。

```bash
python3 -m image_gen.validate_mhla
python3 -m benchmarks.benchmark_core_kernels
```

ベンチマークは、forwardだけでなくbackward、peak allocated/reserved、dtype、batch、矩形解像度を記録する。

## 数値実装の注意

- shapeは「実行できるか」だけでなく、各軸の意味が保たれているかで検証する。主要な標準形は、画像を`(B,C,H,W)`、token列を`(B,T,D)`、attention内部を`Q=(B,Hq,T,Dh)`・`K/V=(B,Hkv,S,Dh)`とする。
- `reshape`・`view`・`transpose`・`permute`の前後では、変換前後のshapeと軸の意味をコメントまたはassertで明示する。見かけ上の要素数が一致することだけを、shapeの正しさの根拠にしない。
- downsample/upsampleでは、入力の`H/W`、stride、kernel、paddingから出力空間サイズを確認する。特に可変解像度・矩形画像・window/blockサイズ未満・strideの境界を対象にする。
- token化と復元では、`(B,C,H,W) -> (B,H*W,C) -> (B,C,H,W)`のtoken数と空間順序が一致することを検証する。image/text/context tokenを連結する場合は、各token数とmaskのoffsetを同時にassertする。
- attentionでは、`num_heads % kv_heads == 0`、`D % num_heads == 0`、RoPEのhead dimension制約、Q/K/Vのsequence length、maskのbatch・query・key軸を個別に検証する。
- GroupNormのgroup数、Convの入力channel、Linearの最終次元、catする軸を明示し、channel数1・奇数channel・head数と割り切れない値を境界テストに含める。
- custom kernelへ渡す直前に、shapeだけでなくdtype、device、stride/contiguous、maskの型を確認する。Tritonがflat pointer indexingを使う場合、non-contiguous入力を許容しないか、kernel前に明示的にcontiguous化する。
- shapeテストはbatch=1/2以上、正方形/矩形、最小サイズ、window/blockの倍数/非倍数、text token数1/最大長、部分mask/全maskを含める。各段階のshape、最終出力shape、finite性をassertする。
- 2D RoPEは座標軸ごとの偶奇ペアを独立に回転する。half-split方式とpair-wise方式を混在させない。
- GQAは `Q=(B,Hq,T,D)`、`K/V=(B,Hkv,S,D)` を基本とし、`Hq % Hkv == 0` を検証する。
- 空text maskではsoftmaxの全masked行が発生しないよう、safe maskまたは有効tokenを用意する。
- BF16/FP32の境界では、呼び出し先parameterのdtypeとactivationを一致させる。
- Flow Matchingの変更では、`x_t`、velocity target、samplerの時間方向が同じ定義になっていることを確認する。
- kernel最適化では、naive/reference版とのforward値、backward勾配、mask、矩形shapeの比較を先に行う。
- optimizerでは、更新前後のparameter、moment、preconditioner、projection、weight decayの適用順序を確認する。stateの要素数削減と、step中の一時tensor・reductionコスト・数値近似を別々に評価する。

## checkpointと設定

- architecture変更時はmetadataの保存項目とsampler側の復元項目を同時に更新する。
- `strict=True` のstate dictロードを維持し、構成不一致を早期に検出する。
- CLIの引数を追加・変更した場合は、help、metadata、resume、generate scriptの整合性を確認する。
- optimizerの新しいbackendや更新式を追加した場合は、通常のparameter、1D fallback、BF16、複数parameter group、state保存・復元を対象にする。

## ドキュメント

- 文書の分類と全体索引: [`docs/README.md`](docs/README.md)
- image-latent DiTの設計仕様: [`docs/architecture/image-latent-dit.md`](docs/architecture/image-latent-dit.md)
- 実装監査の2026-09-12 snapshot: [`docs/history/repository-audit-2026-09-12.md`](docs/history/repository-audit-2026-09-12.md)
- 性能レビューの2026-09-12 snapshot: [`docs/history/performance-review-2026-09-12.md`](docs/history/performance-review-2026-09-12.md)
- 作業中の優先タスクやセッション引き継ぎは、公開文書に含めずローカルのignore対象領域で管理する。
