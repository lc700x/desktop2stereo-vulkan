# macOS Core ML Depth Models and Local Viewer Report (2026-11-28)

## Scope

This work deploys the DepthPro2, MoGe-3, and FlashDepth families through
accelerator-only Core ML inference on Apple Silicon, keeps Windows/ROCm/MPS
numeric semantics aligned, and records depth, Half-SBS, and Local Viewer
evidence.

The validated matrix covers:

- DepthPro2-S, DepthPro2-B, DepthPro2-L, and DepthPro2-H.
- MoGe3-L and MoGe3-G.
- FlashDepth-S, FlashDepth-L, and FlashDepth-Full.
- Distill-Any-Depth-Small at 336 and Distill-Any-Depth-Base at 518 as the
  visual and latency references.

## Numeric Contract

All providers normalize to the same D2S response used by Distill-Any-Depth:
near is high and far is low, followed by the shared 2nd/98th percentile
normalization.

| Family | Official output | D2S conversion |
|---|---|---|
| DepthPro2 | `inverse_depth` | unchanged |
| FlashDepth | ReLU inverse/disparity | unchanged |
| MoGe-3 | metric Z depth | `1 / clamp(depth, min=0.005)` |

Masked MoGe pixels use the official infinite-Z meaning, so they become far
instead of incorrectly appearing as near foreground. The same conversion is
used by Core ML, PyTorch MPS, CUDA, ROCm, TensorRT, and ONNX paths.

DepthPro2 also needs its focal-normalized inverse-depth response kept as-is.
FlashDepth's official output is already inverse depth/ReLU disparity. The
conversion therefore happens only where the source output is metric Z; no
second normalization is applied inside a model wrapper.

## Inference Backend

- Apple Silicon, macOS 15+.
- Core ML fixed-shape `.mlpackage` artifacts cached under `models/coreml/`.
- GPU hard gate through `MLComputePlan`: every non-constant operation must
  support and prefer GPU. CPU-only operations fail the model.
- New-family `auto`/`all` requests are normalized to GPU; explicit ANE or CPU
  requests are rejected.
- Local Viewer resolves compute units by model family before launch, so the
  legacy monitor-mode ANE default cannot be applied to the new GPU-only models.
- No fallback from the new families to PyTorch MPS or CPU.
- The native Core ML/Metal bridge accepts model-specific normalization and
  retains the IOSurface-to-Metal input path.
- Core MLTools 9 cannot reliably compile these transformer graphs as FP16;
  the new family artifacts use FP32 GPU compute for deterministic output.
- The Local Viewer path accepts 3D and 4D Core ML outputs and creates
  contiguous `MLMultiArray` strides for the actual output rank. This preserves
  the existing Metal zero-copy path for the new model families.
- The new families use a strict native path. If the native bridge cannot
  produce an accelerator-backed frame, the provider fails closed instead of
  silently switching to PyTorch MPS or CPU.

## Resolution Profiles

| Family | Resolution list | Reason |
|---|---|---|
| DepthPro2 | 320, 352, 384, 448, 512, 640, 768 | DINOv3 patch 16 and DepthPro2 input multiple 32 |
| MoGe-3 | 294, 336, 392, 448, 518 | DINOv2 patch 14 |
| FlashDepth | 294, 336, 392, 448, 518 | DINOv2 patch 14; S/L are trained at 518x518 |

Evidence matrix profiles:

- DepthPro2: 320 and 512.
- MoGe-3 / FlashDepth: 336 and 518.
- Distill baselines: Small at 336 and Base at 518.

The alignment follows the official implementations:

- [Depth Pro 2](https://github.com/apple-aiml-research/ml-depth-pro-2)
  pads to multiples of 32 and uses DINOv3 patch-16 backbones. Its FOV
  preprocessing caps the short side at 512 while preserving aspect ratio.
  The relevant vendored constraints are in
  `stereo_runtime/model_impl/depth_pro_2/model.py` and
  `stereo_runtime/model_impl/depth_pro_2/network/fov.py`.
- [MoGe](https://github.com/Microsoft/MoGe) uses DINOv2 ViT patch-14
  backbones and supports flexible aspect ratios. The vendored encoder asserts
  divisibility by 14.
- [FlashDepth](https://github.com/Eyeline-Labs/FlashDepth) reports training
  FlashDepth-S/L at 518x518 and uses DINOv2 patch-14 encoders. Full is trained
  at a higher resolution, while 518 remains the evaluation/live profile used
  in this matrix.

The GUI and registry now derive alignment from the same family constants, so
Windows TensorRT/ONNX and macOS do not independently round resolutions.

## Windows and Cross-Platform Parity

- TensorRT native and ORT constructors receive the selected depth resolution.
- The shared depth preprocessor derives patch alignment from the model family.
- New-family resize is bilinear without antialiasing, matching the official
  inference setup.
- MoGe's masked depth is represented as infinite Z before the shared
  reciprocal conversion, including the ONNX export wrapper.
- DepthPro2 uses BF16 on MPS where supported and FP32 Core ML graphs with the
  same inverse-depth output semantics.

## Validation Results

The full evidence root is `outputs/macos_coreml_matrix/`; committed summaries
and contact sheets are under `docs/evidence/20261128-macos-coreml-depth/`.

- 20/20 online depth runs completed successfully.
- Every new-family Core ML artifact reports zero CPU-only operators and 100%
  GPU-preferred operations under `MLComputePlan`.
- 9/9 Local Viewer runs used `depth_backend=coreml`, `fallback=false`,
  `coreml_input_shared=1`, and `coreml_output_zero_copy=1`.
- All generated depth maps were finite and in the canonical `[0, 1]` range.
- Contact sheets compare each new model at 336 against Distill-Small at 336,
  and at 518 against Distill-Base at 518.

Representative Local Viewer p50/p95 latency on the validated M1:

| Model | Resolution | Present FPS | Content FPS | Latency p50/p95 ms | Depth p95 ms |
|---|---:|---:|---:|---:|---:|
| DepthPro2-S | 512 | 64.7 | 20.9 | 57.8 / 84.4 | 45.4 |
| DepthPro2-H | 512 | 52.8 | 1.0 | 1000.2 / 1064.5 | 1018.2 |
| FlashDepth-S | 518 | 51.6 | 15.4 | 65.1 / 90.6 | 49.7 |
| FlashDepth-L | 518 | 51.7 | 2.5 | 421.3 / 467.9 | 405.6 |
| MoGe3-L | 336 | 51.5 | 0.5 | 2027.7 / 2272.2 | 2243.5 |
| MoGe3-G | 336 | 50.7 | 0.2 | 7482.0 / 7981.6 | 7966.2 |

Present FPS is display presentation throughput and includes reuse while a
new depth frame is in flight. Content FPS is the rate of fresh depth results,
so it is the relevant live-model throughput number.

## Reproduce

```bash
src/python3/bin/python scripts/benchmark/run_macos_coreml_matrix.py \
  --media outputs/macos_coreml_matrix/media/depth_models_60fps_loop.mp4 \
  --output-root outputs/macos_coreml_matrix \
  --compute-units gpu \
  --viewer-seconds 30
```

The runner writes offline samples, Local Viewer logs, contact sheets,
`report.json`, `report.csv`, and `report.md` under the output root. Generated
models and evidence remain untracked.

## Evidence

Generated summary evidence is copied to
`docs/evidence/20261128-macos-coreml-depth/` after a complete run. The report
distinguishes successful artifacts from failures caused by disk, memory, or
unsupported graph operations; it never silently substitutes CPU inference.

The Flet GUI rendered blank in this desktop session, so the macOS Core ML reset
was exercised through the same `DEFAULTS` and reset implementation used by the
button rather than a physical click. Runtime and Local Viewer behavior were
validated directly.
