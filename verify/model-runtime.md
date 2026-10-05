# Runtime probes: models and kernels

実モデル・VAE・kernelのruntime確認手順です。外部モデルやCUDAが必要な項目は、その前提と未実行時の扱いを各節に記載しています。

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
