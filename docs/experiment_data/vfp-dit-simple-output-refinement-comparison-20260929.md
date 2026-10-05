# VFP-DiT output-refinement comparison (2026-09-29)

> **Historical result:** This single-seed, short screen used the retired `vfp_dit_simple` implementation. Its measurements do not describe the current [`vfp_dit`](../../vfp_dit/) architecture or establish a current recommendation.

## Results

| Variant | Refinement blocks | Conditioning | Train loss | Validation loss (32) | T2I val (14) | TI2I val (18) | Peak allocated | Peak reserved | Train samples/s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Depth4 | 4 | none | **0.636741** | **0.463533** | **0.441921** | **0.480342** | 6.760 GiB | 7.414 GiB | 0.911 |
| Cross-attn 2 | 2 | cross-attention | 0.635910 | 0.479307 | 0.454079 | 0.498928 | **6.583 GiB** | 10.518 GiB | **0.950** |
| Cross-attn 4 | 4 | cross-attention | 0.717993 | 0.615525 | 0.567608 | 0.652793 | 6.868 GiB | 10.811 GiB | 0.747 |
| Depth8 | 8 | none | 0.674457 | 0.505746 | 0.470348 | 0.533277 | 7.234 GiB | 8.070 GiB | 0.915 |
| Front16 + depth8 | 8 | none | 0.701156 | 0.517950 | 0.483255 | 0.544936 | **6.015 GiB** | **6.852 GiB** | 0.967 |

All runs used 512px, batch size 1, one timestep per image, 256 training examples, 32 validation examples, seed 42, gradient checkpointing, and the same initialization checkpoint. The cross-attn 4 and depth4 runs both use 4 refinement blocks, making them the closest comparison of conditioning mode. The cross-attn 2 run changes both conditioning and depth. Depth8 keeps conditioning off and raises refinement depth from four to eight. Front16 + depth8 also keeps width 1024 and initialization fixed while reducing front DiT depth from 24 to 16.

Cleanup intervals varied: depth4 used GC=100/cache=0, cross-attn 2 used GC=100/cache=10, and cross-attn 4 and depth8 used GC=10/cache=10. These do not change the intended model equation but can affect runtime and allocator behavior.

## Interpretation

Cross-attn 4 validation loss is 0.151992 above depth4 without conditioning, about 32.8% relative to the latter. Training loss is also higher by 0.081252. Compared with cross-attn 2, cross-attn 4 has validation loss higher by 0.136218 and training loss higher by 0.082083. Its T2I and TI2I validation losses are both worse. This short, single-seed screen suggests that adding the cross-attention conditioning path to the 4-block refinement did not help at 256 updates. The extra randomly initialized blocks and cross-attention projections may need more updates, so this is not a long-run architecture verdict.

Cross-attn 4 peak allocated VRAM was 6.868 GiB, 0.109 GiB above depth4/no-conditioning and 0.286 GiB above cross-attn 2. Peak reserved was 10.811 GiB, 3.396 GiB above depth4/no-conditioning and 0.293 GiB above cross-attn 2. On a 12 GiB GPU, that reserved peak leaves less allocator headroom. Reserved memory includes allocator-held blocks and is not identical to live tensor allocation, but it is relevant to subsequent allocations and perceived VRAM pressure.

Cross-attn 4 trained at 0.747 samples/s, about 21.3% slower than cross-attn 2 and 18.0% slower than depth4/no-conditioning. With one run per variant, treat throughput and quality differences as screening results. A controlled follow-up should hold block count, GC/cache intervals, and initialization constant while changing only conditioning mode, then use multiple seeds or a longer schedule if the short comparison merits it.

Depth8 without conditioning has validation loss 0.042213 above depth4 without conditioning, about 9.1% relative to depth4, and training loss is 0.037716 higher. Peak allocated/reserved VRAM rises by 0.474/0.656 GiB. Training throughput is nearly unchanged (0.915 vs 0.911 samples/s). At 256 updates, adding four more full-resolution blocks did not improve validation loss; the extra depth also uses more memory. This remains a short, single-seed screen and does not establish that depth8 would lose after longer training.

Reducing the front DiT from 24 to 16 layers while retaining eight 1024-wide, unconditioned refinement blocks raised validation loss from 0.505746 to 0.517950 (+0.012204, about 2.4%); training loss rose by 0.026699. T2I and TI2I validation losses each rose by about 0.013 and 0.012. Peak allocated/reserved memory fell by 1.219 GiB in both measures (7.234/8.070 to 6.015/6.852 GiB), while throughput increased from 0.915 to 0.967 samples/s (+5.7%). In this 256-update screen, front16 + depth8 therefore trades a small loss increase for a clear memory reduction and modest speed gain. This is not yet evidence that the reduced front depth is equivalent after a longer schedule or across seeds.

## Run evidence

The run IDs below identify the conditions summarized above. Per-run raw configs, metrics, logs, and other artifacts were removed from the public archive; the comparison table and interpretation in this document are the retained aggregate record.

- Depth8 without conditioning: `vfp_dit_simple.train_output-refine-depth8_20260929T133744Z_513cc9c2` (raw archive removed)
- Front16 + depth8 without conditioning: `vfp_dit_simple.train_output-refine-depth8-front16_20260929T135631Z_f202ab27` (raw archive removed)
- Cross-attn 4: `vfp_dit_simple.train_output-refine-depth4-cross-attn_20260929T132632Z_9fb04adf` (raw archive removed)
- Cross-attn 2: `vfp_dit_simple.train_output-refine-cross-attn_20260929T132104Z_e05d65a8` (raw archive removed)
- Depth4 without conditioning: `vfp_dit_simple.train_output-refine-depth4_20260929T124040Z_22fe1246` (raw archive removed)
- Earlier stopped cross-attn attempt: `vfp_dit_simple.train_output-refine-cross-attn_20260929T124720Z_359d618b` (raw archive removed), stopped at step 90/256 due to reported VRAM pressure.
