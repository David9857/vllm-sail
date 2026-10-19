# Loading-time FP8 conversion

The PPU plugin registers `fp8-to-int8` and `fp8-to-bf16`. Both read the original
FP8 checkpoint and convert weights once after loading. INT8 execution uses the
same PPU Dense and MoE backends as offline INT8 weights; activations are quantized
dynamically per token during inference.

## Select a path

```bash
vllm serve /path/to/fp8-model --quantization fp8-to-int8 --dtype bfloat16
vllm serve /path/to/fp8-model --quantization fp8-to-bf16 --dtype bfloat16
```

The Python API accepts the same names through `LLM(quantization=..., dtype="bfloat16")`.
The source must declare `quant_method: fp8`, `weight_block_size: [128, 128]`,
E4M3 weights, and dynamic activations. Other source formats fail validation.
Use explicit BF16 activation dtype; automatic dtype follows upstream model metadata
and can reject a checkpoint that declares FP16.

FP8-to-BF16 decodes the checkpoint; it cannot recover precision lost when the
original floating weights were quantized. It also increases resident weight
memory. FP8-to-INT8 introduces a further quantization step and must be evaluated
on the intended model and input protocol.

## Defaults and compatibility

Installing the plugin does not automatically convert every FP8 checkpoint.
Explicit quantization arguments take precedence over environment policies;
`--quantization fp8` retains native FP8 behavior.

With no explicit quantization argument, `VLLM_SAIL_FP8_LOAD_POLICY` selects
`native`, `bf16`, or `int8`. Unset or empty retains native behavior. The corresponding
`VLLM_PPU_FP8_LOAD_POLICY` alias is accepted; SAIL takes precedence.

If no load policy is set, an explicitly set `VLLM_SAIL_ENABLE_FP8_TO_INT8`
(or legacy `VLLM_PPU_ENABLE_FP8_TO_INT8`) selects INT8 for `1` and BF16 for `0`.
Leaving the legacy variable unset does not change the default. Unlike older
in-tree deployments, this switch never overrides an explicit quantization argument.

## Layer selection and storage

Checkpoint `ignored_layers` / `modules_to_not_convert` are translated through the
model's weight mapper. Fused QKV and gate/up groups must have consistent selection.
The port retains floating-layer safeguards for embeddings, LM heads, visual
towers, routers and selected hybrid-model projections. Weight-loader checks reject
an FP8 tensor assigned to a BF16 layer or a floating tensor assigned to an FP8
layer before implicit dtype casting can corrupt it.

Storage metadata is authoritative. `mix_layers` and `scales_path` are rejected:
neither can safely invent missing source-format information or an implemented
scale-cache contract. Checkpoint exclusions must describe actual floating storage.

## Parallelism and limitations

FP8 fused boundaries and TP shards must align to the checkpoint's 128-element
blocks. Independent MoE gate/up tail blocks retain independent scale grids.
INT8 input-dimension shards use MAX reduction of row maxima across TP before
quantization, matching full-row scale selection rather than independently scaling
each local shard. This can differ from older in-tree TP conversion behavior.

MoE conversion rejects padded dimensions, flattened DP/PCP tensor parallel groups
for INT8 conversion, and EPLB. Preprocessed-weight loading is not supported.
Reloading converted weights is not an established workflow; restart and load the
original checkpoint when changing quantization paths. Backend-specific limits
still apply.

These are implementation paths, not a qualification of all models, TP/EP modes,
compiled execution or device/SDK combinations. Device tests and full-model accuracy
and performance measurements are required for each deployment. CPU tests and
skipped device tests are not evidence of PPU runtime correctness.

## BF16 source checkpoints

`bf16-to-int8` is intentionally not registered. A BF16 checkpoint does not specify
which layers to quantize or retain in floating precision. A future entry point
requires an explicit, validated recipe; it must not infer one from a directory
name or silently quantize all Linear layers. FP8 conversion entry points reject
BF16 checkpoints.

## Verification

The dependency-free tests run in the normal CPU merge gate:

```bash
python -m pytest tests/ut/test_online_conversion.py tests/ut/test_envs.py -q
```

Tensor-only reference tests require torch, without vLLM or a device:

```bash
python -m pytest tests/tensor/test_online_conversion_math.py -q
```

With vLLM 0.31 installed, CPU loading-contract tests exercise its real parameter
classes and check the MoE kernel call interface (the device kernel is mocked):

```bash
python -m pytest tests/integration/test_online_conversion_loading.py -q
```

On an installed SAIL environment, run the Dense conversion gate:

```bash
python -m pytest tests/e2e/test_online_conversion_device.py -q
```

Then compare converted weights/scales, prefill/decode logits, complete generation
and stop reasons against fixed references. Keep checkpoint revision, prompt IDs,
chat template, sampling, SDK, backend, TP/EP and execution mode fixed. Run an A/A
repeatability check before attributing differences to conversion. Record cold-start
time and peak memory separately from steady-state inference performance.
