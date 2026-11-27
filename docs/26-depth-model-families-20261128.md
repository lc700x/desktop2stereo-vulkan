# Depth model families implementation and performance report — 2026-11-28

## 1. Scope

This change adds the requested model families to the runtime and GUI while keeping the existing
Distill-Any-Depth baselines available:

| Family | Variants | Hugging Face source | License |
|---|---|---|---|
| DepthPro2 | S, B, L, H | `lc700x/DepthPro2` mirror of Apple checkpoints | Apple research-only |
| MoGe-3 | L, G | `Ruicheng/moge-3-vitl`, `Ruicheng/moge-3-vitg` | MIT |
| FlashDepth | S, L, Full | `Eyeline-Labs/FlashDepth` | Apache-2.0 |

The GUI model list now contains 46 entries. DepthPro2 is shown but gated by an explicit
research-license acknowledgement. The acknowledgement is checked in the model-selection handler
before a DepthPro2 download or run is allowed.

## 2. Runtime implementation

### Registry and downloads

- All nine variants have explicit `DepthModelSpec` entries, artifact names, weight files,
  output semantics, license metadata, and compatibility profiles.
- DepthPro2 requires both the selected checkpoint and the shared FOV checkpoint before the cache is
  considered complete.
- FlashDepth shared-repository checkpoints resolve through `flashdepth-s/`, `flashdepth-l/`, and
  `flashdepth/` subfolders, including Hugging Face snapshot cache layouts.
- Shared-repository artifact prefixes prevent DepthPro2 and FlashDepth variants from overwriting
  each other's ONNX/TensorRT/MIGraphX files.
- New families do not silently fall back to CPU in the automatic GPU chain.

### PyTorch providers

- `DepthPro2Provider` uses the official `DepthPro2.infer()` inverse-depth output on device.
- `MoGe3Provider` runs the non-refiner compatibility path and recovers metric Z depth with a
  batched, device-resident 2×2 solver. The upstream NumPy focal solver is not used in the hot path.
- `FlashDepthProvider` runs the still-image DPT decoder without the optional Mamba temporal module.
  FlashDepth-Full retains hybrid teacher fusion but uses an empty temporal-layer list.
- All three providers report `zero_copy_output=True` and `host_readback_bytes=0` for their inference
  results. Non-finite model values are repaired by the shared depth postprocessor before DIBR.

### Compiled backends

- ONNX wrappers now exist for DepthPro2, MoGe-3, and FlashDepth.
- Native TensorRT and MIGraphX share the same canonical `pixel_values -> predicted depth` graph
  contract as the existing providers.
- TensorRT/MIGraphX provider reports carry the compatibility profile and zero-copy metadata.
- The benchmark script was rewritten to use the current provider API instead of the removed
  `depth_trt_native_provider` and `depth_trt_provider` modules.

### Automatic compatibility profiles

For GPUs below 10 GiB, the provider caps selected large variants to a conservative grid while
keeping the profile visible in telemetry. Examples:

- DepthPro2-H/H → 294
- DepthPro2-L → 336
- DepthPro2-B → 448
- MoGe3-G → 336
- FlashDepth-L/Full → 448

Turing-class NVIDIA GPUs use FP16 autocast for DepthPro2; Ampere and newer use its trained BF16
path.

## 3. Hardware and test frame

- GPU 1: AMD Radeon RX 9060 XT, ROCm 7.16 / PyTorch 2.11
- GPU 2: NVIDIA RTX 2070 8 GB, CUDA 12.8 / PyTorch 2.11
- Source: `P1-【4K 120FPS HDR10】文字大战.mp4`, frame at 75 seconds, 3840×1920.
- Benchmark inference grid: 518×518, with the shared D2S depth normalization and upsample to the
  benchmark output size.
- All timings are medians of warm iterations. Evidence files are in
  [`docs/evidence/20261128-depth-models/`](evidence/20261128-depth-models/).

## 4. RX 9060 XT results

### 4.1 PyTorch ROCm, 518×518

| Model | Median total | P90 total | Model stage median | Profile |
|---|---:|---:|---:|---|
| Distill-Any-Depth-Small | 27.61 ms | 27.83 ms | 7.83 ms | baseline |
| Distill-Any-Depth-Base | 56.32 ms | 57.18 ms | 55.25 ms | baseline |
| DepthPro2-S | 38.46 ms | 38.83 ms | 37.26 ms | `onnx-inverse-depth` |
| MoGe3-L | 920.10 ms | 929.15 ms | 919.24 ms | `spatial-refine-off` |
| FlashDepth-S | 18.18 ms | 18.28 ms | 17.39 ms | `spatial-temporal-off` |
| FlashDepth-Full | 93.94 ms | 95.93 ms | 92.99 ms | `spatial-temporal-off` |

### 4.2 MIGraphX ROCm, FlashDepth-S

| Backend | Median total | P90 total | Model stage median |
|---|---:|---:|---:|
| PyTorch ROCm | 18.18 ms | 18.28 ms | 17.39 ms |
| MIGraphX ROCm | **4.13 ms** | 5.07 ms | **0.86 ms** |

MIGraphX is about **4.4× faster** by median end-to-end provider time and about **20× faster** in
the model-only stage. The first build took about 5m45s and the cached graph load is about 1.1s.
The provider reported `zero_copy_output=True` and `host_readback_bytes=0`.

The 1080p post-upsample evidence is also stored in the evidence directory. Its ranking is not used
above because the model input grid remains fixed and the first-run/upsample caches differ.

## 5. RTX 2070 results

### 5.1 PyTorch CUDA, 518×518

| Model | Median total | P90 total | Model stage median | Profile |
|---|---:|---:|---:|---|
| Distill-Any-Depth-Small | 49.59 ms | 109.28 ms | 6.09 ms | baseline |
| Distill-Any-Depth-Base | 31.31 ms | 31.47 ms | 30.36 ms | baseline |
| DepthPro2-S | 90.55 ms | 184.04 ms | 87.95 ms | `onnx-inverse-depth` |
| MoGe3-L | 294.57 ms | 295.91 ms | 294.01 ms | `spatial-refine-off` |
| FlashDepth-S | 23.81 ms | 23.89 ms | 23.24 ms | `spatial-temporal-off` |
| FlashDepth-Full | 146.87 ms | 316.89 ms | 145.83 ms | `spatial-temporal-off@448` |

### 5.2 TensorRT native

| Model | PyTorch median | TensorRT median | Speedup | TensorRT model stage |
|---|---:|---:|---:|---:|
| FlashDepth-S | 23.81 ms | **5.42 ms** | **4.4×** | 0.95 ms |
| DepthPro2-S | 90.55 ms | **33.92 ms** | **2.7×** | 1.47 ms |
| MoGe3-L | 294.57 ms | **178.58 ms** | **1.65×** | 2.74 ms |

The first TensorRT build took roughly 1–4 minutes depending on graph size. Cached engine load was
around 0.2s for FlashDepth-S. All three compiled providers reported zero host readback.

## 6. Quality gates

The parity check compares the compiled backend against the PyTorch reference on the same artifact
grid. The supplied person frame was finite in every tested backend.

| Model / backend | Finite | Spearman ρ | Normalized MAE | Gate |
|---|---:|---:|---:|---|
| FlashDepth-S / MIGraphX | yes | 0.999996 | 0.000367 | pass |
| FlashDepth-S / TensorRT | yes | 0.999989 | 0.000584 | pass |
| DepthPro2-S / TensorRT | yes | 0.999995 | 0.000647 | pass |
| MoGe3-L / TensorRT | yes | 0.999973 | 0.000619 | pass |

The required quality gate was finite depth 100%, Spearman ≥ 0.98, and normalized MAE ≤ 0.03.
All tested compiled graphs pass with substantial margin. The shared postprocessor also guarantees
finite normalized depth for NaN/Inf inputs, including the sky/Inf cases that DepthPro2 can emit.

## 7. Half-SBS visual review

The contact sheet shows the same person frame after DIBR synthesis for all six primary benchmark
models:

[Depth model Half-SBS contact sheet](evidence/20261128-depth-models/half_sbs_contact_sheet.jpg)

Each image is a 960×540 Half-SBS result. The left/right views contain different parallax warps, not
a duplicated left image. Full individual samples and the source frame are in
`outputs/depth-model-eval-20261128/samples-75s/` on the test host and the contact sheet is committed
as evidence.

## 8. Regression coverage

- Targeted depth registry/artifact/semantics/new-family tests: **137 passed**.
- GUI/config/adapter regression subset: **205 passed, 4 skipped**.
- Full suite with `PYTHONUTF8=1`: **3252 passed, 88 skipped, 12 failed**. The 12 failures are
  the existing macOS Metal/CoreML viewer tests and the POSIX installer shell-layout assertion; no
  new depth-provider, registry, adapter, GUI, or artifact tests failed.

## 9. Remaining limitations

- The HF mirror upload was implemented in
  `scripts/tools/mirror_depthpro2_hf.py`, but this workstation has no Hugging Face login. The script
  intentionally refuses to invent credentials and instructs the operator to run `hf auth login`.
  Once authenticated it creates the public gated `lc700x/DepthPro2` repository, writes the original
  license and SHA-256 manifest, and uploads the five Apple checkpoints.
- Performance was measured on representative S/L/Full variants. The B/L/H DepthPro2 variants and
  MoGe3-G use the same provider and export paths and are covered by registry/configuration tests,
  but their multi-GB weights were not downloaded for a full timing sweep in this run.
- The compatibility profiles intentionally disable MoGe sparse refinement and FlashDepth temporal
  Mamba. Full-fidelity profiles remain available where their optional dependencies are installed.
- New-family CoreML conversion is not claimed. macOS uses the PyTorch MPS path; CoreML remains
  limited to model families with an existing CoreML artifact.
