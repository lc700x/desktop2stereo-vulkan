from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import threading
import time

import torch

from ...depth_onnx_provider import ModelOnnxPreprocessor, _dtype_from_onnx_name, _input_size_from_artifact_name, default_onnx_path
from ...depth_provider import (
    DISTILL_ANY_DEPTH_BASE_MODEL_ID,
    DISTILL_ANY_DEPTH_BASE_NAME,
    DISTILL_ANY_DEPTH_PATCH_SIZE,
    DISTILL_ANY_DEPTH_BASE_RESOLUTION,
    ZIPDEPTH_PATCH_SIZE,
    _is_zipdepth_model,
    _model_input_size_for_model,
    DepthProfileResult,
    DepthProviderConfig,
    DepthProviderInfo,
    _prepare_accelerated_artifacts,
    _normalize_depth,
    default_lab_cache_dir,
)
from ...depth_upsample import DepthUpsampleMode, upsample_depth
from ...output import ensure_b1hw, ensure_bchw, match_depth
from ...progress import activity_progress, emit_progress_event, status_write, write_bytes_with_progress
from .tensorrt_ort import ensure_tensorrt_dll_path


def default_distill_base_native_trt_path(cache_dir: str | Path | None = None) -> Path:
    cache = Path(cache_dir) if cache_dir is not None else default_lab_cache_dir()
    return cache / "models--lc700x--Distill-Any-Depth-Base-hf" / "model_fp16_294x518.trt"


def default_native_tensorrt_engine_path(cache_dir: str | Path | None = None) -> Path:
    return default_distill_base_native_trt_path(cache_dir)


def _compact_artifact_path(path: str | Path) -> str:
    artifact = Path(path)
    parent_name = artifact.parent.name
    return f"{parent_name}/{artifact.name}" if parent_name else artifact.name


def _infer_model_metadata_from_paths(
    onnx_path: Path,
    engine_path: Path,
    *,
    model_id: str,
    model_name: str,
) -> tuple[str, str]:
    if model_id != DISTILL_ANY_DEPTH_BASE_MODEL_ID or model_name != DISTILL_ANY_DEPTH_BASE_NAME:
        return model_id, model_name

    for path in (onnx_path, engine_path):
        for part in path.parts:
            if part.startswith("models--") and "Distill-Any-Depth" in part:
                inferred_id = part.removeprefix("models--").replace("--", "/")
                inferred_name = inferred_id.rsplit("/", 1)[-1].replace("-hf", "")
                return inferred_id, inferred_name
    return model_id, model_name


@dataclass
class _NativeTensorRtExecutionSlot:
    index: int
    context: object
    runtime: object | None = None
    engine: object | None = None
    input_tensor: torch.Tensor | None = None
    output_buffers: dict[str, torch.Tensor] = field(default_factory=dict)
    graph_input: torch.Tensor | None = None
    graph_output_name: str | None = None
    graph: torch.cuda.CUDAGraph | None = None
    available_event: object | None = None
    stream: object | None = None
    reserved: bool = False
    last_acquire_wait_ms: float = 0.0


class NativeTensorRtEngine:
    def __init__(
        self,
        engine_path: str | Path,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        execution_slot_count: int = 1,
    ) -> None:
        self.engine_path = Path(engine_path)
        self.device = torch.device(device)
        self.dtype = dtype
        self._slots: list[_NativeTensorRtExecutionSlot] = []
        self._next_slot_index = 0
        self._slot_lock = threading.Lock()
        if self.device.type != "cuda":
            raise RuntimeError("Native TensorRT engine requires CUDA")
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine file not found: {self.engine_path}")

        ensure_tensorrt_dll_path()
        import tensorrt as trt

        with self.engine_path.open("rb") as file:
            self._serialized_engine = file.read()
        self._trt = trt
        self._logger = trt.Logger(trt.Logger.ERROR)

        first_slot = self._create_execution_slot(0)
        self._slots.append(first_slot)
        requested_slots = min(3, max(1, int(execution_slot_count)))
        for slot_index in range(1, requested_slots):
            try:
                self._slots.append(self._create_execution_slot(slot_index))
            except Exception as exc:
                print(
                    f"[TensorRT] execution slot {slot_index + 1}/{requested_slots} unavailable; "
                    f"using {len(self._slots)} slot(s): {type(exc).__name__}: {exc}",
                    flush=True,
                )
                break
        # Preserve the legacy attribute for diagnostics and integrations that
        # inspect the primary context. Runtime execution uses each slot's engine.
        self.runtime = first_slot.runtime
        self.engine = first_slot.engine
        self.context = first_slot.context

        self.input_names: list[str] = []
        self.output_names: list[str] = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        if not self.input_names:
            raise RuntimeError("TensorRT engine has no input tensors")
        if not self.output_names:
            raise RuntimeError("TensorRT engine has no output tensors")

    def _create_execution_slot(self, index: int) -> _NativeTensorRtExecutionSlot:
        # Myelin keeps graph state inside a deserialized engine. Slots must not
        # share that engine when enqueueV3 work can overlap on CUDA streams.
        runtime = self._trt.Runtime(self._logger)
        engine = runtime.deserialize_cuda_engine(self._serialized_engine)
        if engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {self.engine_path}")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("failed to create TensorRT execution context")
        return _NativeTensorRtExecutionSlot(
            index,
            context,
            runtime=runtime,
            engine=engine,
            stream=torch.cuda.Stream(device=self.device),
        )

    @property
    def pipeline_slot_count(self) -> int:
        return len(self._slots)

    @property
    def input_shape(self) -> tuple[int, ...]:
        input_name = self.input_names[0]
        return tuple(int(dim) for dim in self.engine.get_tensor_shape(input_name))

    @property
    def input_image_size(self) -> tuple[int, int] | None:
        shape = self.input_shape
        if len(shape) != 4 or any(dim < 1 for dim in shape):
            return None
        return int(shape[-2]), int(shape[-1])

    @staticmethod
    def _torch_dtype_from_trt(dtype) -> torch.dtype:
        import tensorrt as trt

        mapping = {
            trt.float16: torch.float16,
            trt.float32: torch.float32,
            trt.int32: torch.int32,
            trt.int8: torch.int8,
            trt.bool: torch.bool,
        }
        if hasattr(trt, "bfloat16"):
            mapping[trt.bfloat16] = torch.bfloat16
        return mapping.get(dtype, torch.float32)

    @staticmethod
    def _event_ready(event: object | None) -> bool:
        if event is None:
            return True
        query = getattr(event, "query", None)
        if not callable(query):
            return False
        try:
            return bool(query())
        except Exception:
            return False

    def _acquire_slot(self) -> _NativeTensorRtExecutionSlot:
        # A context may be used by exactly one host thread at a time. The CUDA
        # completion event alone is insufficient because it was previously
        # recorded only after enqueueV3 had already begun.
        slot_lock = getattr(self, "_slot_lock", None)
        if slot_lock is None:
            slot_lock = threading.Lock()
            self._slot_lock = slot_lock
        wait_start = time.perf_counter()
        while True:
            wait_event = None
            with slot_lock:
                slot_count = len(self._slots)
                for offset in range(slot_count):
                    index = (self._next_slot_index + offset) % slot_count
                    slot = self._slots[index]
                    if slot.reserved:
                        continue
                    if self._event_ready(slot.available_event):
                        slot.available_event = None
                        slot.reserved = True
                        slot.last_acquire_wait_ms = (time.perf_counter() - wait_start) * 1000.0
                        self._next_slot_index = (index + 1) % slot_count
                        return slot
                    if wait_event is None:
                        wait_event = slot.available_event
            # A completed future can reach the scheduler before its final CUDA
            # event is signalled. Wait for an unreserved slot rather than
            # treating normal GPU backpressure as a provider failure. Reserved
            # contexts are never synchronised or reassigned here.
            synchronize = getattr(wait_event, "synchronize", None)
            if callable(synchronize):
                synchronize()
            else:
                time.sleep(0.0005)

    def slot_acquire_wait_ms(self, slot_index: int | None) -> float:
        if slot_index is None:
            return 0.0
        try:
            return max(0.0, float(self._slots[int(slot_index)].last_acquire_wait_ms))
        except (IndexError, TypeError, ValueError, AttributeError):
            return 0.0

    def _bind_input_output(
        self,
        tensor: torch.Tensor,
        slot: _NativeTensorRtExecutionSlot,
    ) -> dict[str, torch.Tensor]:
        tensor = tensor.contiguous().to(device=self.device, dtype=self.dtype)
        # execute_async_v3 does not retain the Python Tensor object. Keep the
        # actual converted allocation alive until this slot is acquired again
        # after its completion event has fired.
        slot.input_tensor = tensor
        input_name = self.input_names[0]
        slot.context.set_input_shape(input_name, tuple(tensor.shape))
        slot.context.set_tensor_address(input_name, tensor.data_ptr())

        outputs: dict[str, torch.Tensor] = {}
        for name in self.output_names:
            shape = tuple(int(dim) for dim in slot.context.get_tensor_shape(name))
            if slot.engine is None:
                raise RuntimeError("TensorRT execution slot has no engine")
            output_dtype = self._torch_dtype_from_trt(slot.engine.get_tensor_dtype(name))
            output = slot.output_buffers.get(name)
            if output is None or tuple(output.shape) != shape or output.dtype != output_dtype:
                output = torch.empty(shape, device=self.device, dtype=output_dtype)
                slot.output_buffers[name] = output
            outputs[name] = output
            slot.context.set_tensor_address(name, output.data_ptr())
        return outputs

    def _execute(self, slot: _NativeTensorRtExecutionSlot) -> None:
        stream = slot.stream or torch.cuda.current_stream(self.device)
        ok = slot.context.execute_async_v3(stream_handle=stream.cuda_stream)
        if ok is False:
            raise RuntimeError("TensorRT execute_async_v3 failed")

    def __call__(self, tensor: torch.Tensor, *, synchronize: bool = True) -> torch.Tensor:
        output, _slot_index = self.run_with_slot(tensor, synchronize=synchronize)
        return output

    def run_with_slot(
        self,
        tensor: torch.Tensor,
        *,
        synchronize: bool = True,
    ) -> tuple[torch.Tensor, int]:
        slot = self._acquire_slot()
        try:
            stream = slot.stream or torch.cuda.current_stream(self.device)
            caller_stream = torch.cuda.current_stream(self.device)
            input_ready = torch.cuda.Event(blocking=False)
            input_ready.record(caller_stream)
            with torch.cuda.stream(stream):
                stream.wait_event(input_ready)
                outputs = self._bind_input_output(tensor, slot)
                self._execute(slot)
                completion = torch.cuda.Event(blocking=False)
                completion.record(stream)
            # Make the result visible to the caller's stream without a host-side
            # synchronize. Post-processing can proceed after this dependency.
            caller_stream.wait_event(completion)
            if synchronize:
                caller_stream.synchronize()

            if "predicted_depth" in outputs:
                return outputs["predicted_depth"], slot.index
            return outputs[self.output_names[0]], slot.index
        except Exception:
            self.mark_slot_available_after_current_stream(slot.index)
            raise

    def _capture_graph_for_slot(
        self,
        input_shape: tuple[int, ...],
        slot: _NativeTensorRtExecutionSlot,
    ) -> None:
        if slot.graph is not None:
            return
        if self.device.type != "cuda":
            raise RuntimeError("CUDA graph requires CUDA")
        static_input = torch.zeros(input_shape, device=self.device, dtype=self.dtype)
        outputs = self._bind_input_output(static_input, slot)
        output_name = "predicted_depth" if "predicted_depth" in outputs else self.output_names[0]

        with torch.cuda.stream(slot.stream):
            for _ in range(3):
                self._execute(slot)
        torch.cuda.synchronize(self.device)

        graph = torch.cuda.CUDAGraph()
        # enqueueV3 explicitly uses slot.stream. The graph context must capture
        # that same stream, rather than torch.cuda.graph's default side stream.
        with torch.cuda.graph(graph, stream=slot.stream):
            self._execute(slot)
        torch.cuda.synchronize(self.device)

        slot.graph_input = static_input
        slot.graph_output_name = output_name
        slot.graph = graph

    def capture_graph(self, input_shape: tuple[int, ...]) -> None:
        for slot in self._slots:
            self._capture_graph_for_slot(input_shape, slot)

    def clear_graph(self) -> None:
        for slot in self._slots:
            slot.graph = None
            slot.graph_input = None
            slot.graph_output_name = None

    def run_graph(self, tensor: torch.Tensor) -> torch.Tensor:
        output, _slot_index = self.run_graph_with_slot(tensor)
        return output

    def run_graph_with_slot(self, tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
        slot = self._acquire_slot()
        try:
            if slot.graph is None or slot.graph_input is None or slot.graph_output_name is None:
                self._capture_graph_for_slot(tuple(tensor.shape), slot)
            assert slot.graph is not None
            assert slot.graph_input is not None
            assert slot.graph_output_name is not None
            tensor = tensor.contiguous().to(device=self.device, dtype=self.dtype)
            if tuple(tensor.shape) != tuple(slot.graph_input.shape):
                raise RuntimeError(f"CUDA graph input shape mismatch: expected {tuple(slot.graph_input.shape)}, got {tuple(tensor.shape)}")
            stream = slot.stream or torch.cuda.current_stream(self.device)
            caller_stream = torch.cuda.current_stream(self.device)
            input_ready = torch.cuda.Event(blocking=False)
            input_ready.record(caller_stream)
            with torch.cuda.stream(stream):
                stream.wait_event(input_ready)
                slot.graph_input.copy_(tensor)
                slot.graph.replay()
                completion = torch.cuda.Event(blocking=False)
                completion.record(stream)
            caller_stream.wait_event(completion)
            return slot.output_buffers[slot.graph_output_name], slot.index
        except Exception:
            self.mark_slot_available_after_current_stream(slot.index)
            raise

    def mark_slot_available_after_current_stream(self, slot_index: int | None) -> None:
        if slot_index is None:
            return
        slot = self._slots[int(slot_index)]
        # Slot acquisition may wait on this event. Let CUDA sleep the host
        # thread during that wait instead of spinning a CPU core each frame.
        event = torch.cuda.Event(blocking=True)
        event.record(torch.cuda.current_stream(self.device))
        slot_lock = getattr(self, "_slot_lock", None)
        if slot_lock is None:
            slot_lock = threading.Lock()
            self._slot_lock = slot_lock
        with slot_lock:
            slot.available_event = event
            slot.reserved = False

    def close(self) -> None:
        for slot in self._slots:
            synchronize = getattr(slot.available_event, "synchronize", None)
            if callable(synchronize):
                try:
                    synchronize()
                except Exception:
                    pass
        self.clear_graph()
        for slot in self._slots:
            slot.input_tensor = None
            slot.output_buffers.clear()
            slot.available_event = None
            slot.stream = None
            slot.context = None
            slot.engine = None
            slot.runtime = None
        self._slots.clear()
        for attr in ("context", "engine", "runtime", "_serialized_engine", "_logger"):
            try:
                setattr(self, attr, None)
            except Exception:
                pass


def _dtype_label(dtype: torch.dtype) -> str:
    return "fp16" if dtype == torch.float16 else "fp32" if dtype == torch.float32 else str(dtype).replace("torch.", "")


def _record_cuda_event(events: dict[str, object], name: str, tensor: torch.Tensor | None) -> None:
    if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
        return
    try:
        event = torch.cuda.Event(blocking=False, enable_timing=True)
        event.record(torch.cuda.current_stream(tensor.device))
        events[name] = event
    except Exception:
        return


class _TensorRtProgressState:
    def __init__(self, desc: str) -> None:
        self.desc = desc
        self.started_at = time.perf_counter()
        self.lock = threading.Lock()
        self.phase_totals: dict[str, int] = {}
        self.last_emit_at = self.started_at
        self.emitted = False

    def phase_start(self, phase_name: str, parent_phase: str | None, num_steps: int) -> None:
        del parent_phase
        phase = str(phase_name or "TensorRT build")
        total = max(1, int(num_steps or 1))
        with self.lock:
            self.phase_totals[phase] = total

    def step_complete(self, phase_name: str, step: int) -> bool:
        phase = str(phase_name or "TensorRT build")
        with self.lock:
            total = self.phase_totals.get(phase)
            if not total:
                return True
            completed = min(total, max(0, int(step) + 1))
            self._emit_locked(phase, completed, total)
        return True

    def phase_finish(self, phase_name: str) -> None:
        phase = str(phase_name or "TensorRT build")
        with self.lock:
            total = self.phase_totals.pop(phase, 1)
            self._emit_locked(phase, total, total, force=not self.emitted)

    def _emit_locked(self, phase: str, completed: int, total: int, *, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self.last_emit_at < 1.0:
            return
        self.last_emit_at = now
        self.emitted = True
        try:
            emit_progress_event(f"{self.desc}: {phase}", completed, total, started_at=self.started_at, unit="steps")
        except Exception:
            return


def _attach_tensorrt_progress_monitor(trt, config, desc: str):
    if not hasattr(trt, "IProgressMonitor") or not hasattr(config, "progress_monitor"):
        return None

    state = _TensorRtProgressState(desc)

    class _ProgressMonitor(trt.IProgressMonitor):
        def __init__(self) -> None:
            super().__init__()

        def phase_start(self, phase_name: str, parent_phase: str | None, num_steps: int) -> None:
            state.phase_start(phase_name, parent_phase, num_steps)

        def step_complete(self, phase_name: str, step: int) -> bool:
            return state.step_complete(phase_name, step)

        def phase_finish(self, phase_name: str) -> None:
            state.phase_finish(phase_name)

    try:
        monitor = _ProgressMonitor()
        config.progress_monitor = monitor
    except Exception:
        return None
    return monitor


def build_native_tensorrt_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    fp16: bool = True,
    workspace_gb: int = 4,
    force: bool = False,
) -> Path:
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    if engine_path.exists() and not force:
        return engine_path
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    print(
        "[TensorRT] building native engine:"
        f" onnx={_compact_artifact_path(onnx_path)}"
        f" engine={_compact_artifact_path(engine_path)}"
        f" dtype={'fp16' if fp16 else 'fp32'}"
        f" workspace_gb={workspace_gb}"
        f" force={force}",
        flush=True,
    )

    ensure_tensorrt_dll_path()
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with activity_progress(f"Parsing ONNX for TensorRT: {onnx_path.name}"):
        parsed = parser.parse(onnx_path.read_bytes())
    if not parsed:
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed: " + "; ".join(errors))

    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.TF32)
    if hasattr(trt.BuilderFlag, "SPARSE_WEIGHTS"):
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb) << 30)

    input_tensor = network.get_input(0)
    input_shape = tuple(int(dim) for dim in input_tensor.shape)
    if any(dim < 1 for dim in input_shape):
        raise RuntimeError(f"native TensorRT build requires fixed input shape, got {input_shape}")

    profile = builder.create_optimization_profile()
    profile.set_shape(input_tensor.name, input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)

    build_desc = f"Building TensorRT engine: {engine_path.name}"
    monitor = _attach_tensorrt_progress_monitor(trt, config, build_desc)
    print(f"[Main] {build_desc}...", flush=True)
    status_write(
        f"正在编译 TensorRT 引擎：{engine_path.name}。这个步骤耗时较长，请耐心等待，进度请看上方进度条。"
    )
    if monitor is None:
        with activity_progress(build_desc):
            serialized = builder.build_serialized_network(network, config)
    else:
        serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build_serialized_network returned None")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_with_progress(engine_path, serialized, f"Saving TensorRT engine: {engine_path.name}")
    print(
        "[TensorRT] native engine ready: "
        f"engine={_compact_artifact_path(engine_path)}",
        flush=True,
    )
    status_write(f"TensorRT 引擎编译完成：{engine_path.name}。")
    return engine_path


class DistillAnyDepthBaseNativeTensorRt:
    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        cache_dir: str | Path | None = None,
        onnx_path: str | Path | None = None,
        onnx_dtype: str = "auto",
        engine_path: str | Path | None = None,
        model_id: str = DISTILL_ANY_DEPTH_BASE_MODEL_ID,
        model_name: str = DISTILL_ANY_DEPTH_BASE_NAME,
        depth_resolution: int = DISTILL_ANY_DEPTH_BASE_RESOLUTION,
        local_files_only: bool = False,
        force_download: bool = False,
        build_engine: bool = False,
        force_rebuild: bool = False,
        use_cuda_graph: bool = False,
        profile_sync: bool = False,
        execution_slot_count: int = 1,
        depth_upsample: DepthUpsampleMode = "bilinear",
        depth_upsample_edge_strength: float = 0.35,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("Native TensorRT depth provider requires CUDA")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_lab_cache_dir()
        self._explicit_onnx_path = Path(onnx_path) if onnx_path is not None else None
        self.onnx_dtype = str(onnx_dtype)
        self._explicit_engine_path = Path(engine_path) if engine_path is not None else None
        self.onnx_path = self._explicit_onnx_path or default_onnx_path(self.cache_dir)
        self.engine_path = self._explicit_engine_path or default_native_tensorrt_engine_path(self.cache_dir)
        self.model_id, self.model_name = _infer_model_metadata_from_paths(
            self.onnx_path,
            self.engine_path,
            model_id=model_id,
            model_name=model_name,
        )
        self.depth_resolution = max(1, int(depth_resolution))
        self.local_files_only = bool(local_files_only)
        self.force_download = bool(force_download)
        self.build_engine = bool(build_engine)
        self.force_rebuild = bool(force_rebuild)
        self.use_cuda_graph = bool(use_cuda_graph)
        self.profile_sync = bool(profile_sync)
        requested_slots = min(3, max(1, int(execution_slot_count)))
        self.execution_slot_count = requested_slots
        self.depth_upsample = depth_upsample
        self.depth_upsample_edge_strength = float(depth_upsample_edge_strength)
        self.dtype = _dtype_from_onnx_name(self.onnx_path, torch.float16)
        self.info = DepthProviderInfo(
            provider="tensorrt.Runtime",
            model_name=self.model_name,
            model_id=self.model_id,
            depth_resolution=self.depth_resolution,
            cache_dir=str(self.cache_dir),
            load_mode="local_onnx_native_tensorrt",
            depth_backend="tensorrt_native_graph" if self.use_cuda_graph else "tensorrt_native",
            runtime=("tensorrt-native-cudagraph" if self.use_cuda_graph else "tensorrt-native") + ("-profile-sync" if self.profile_sync else "-async"),
            onnx_path=str(self.onnx_path),
            io_binding=False,
            output_device="cuda",
        )
        self._engine: NativeTensorRtEngine | None = None
        self._artifact_input_size: tuple[int, int] | None = None
        self._cuda_graph_disabled_reason: str | None = None
        configured_input_size = None
        if self._explicit_onnx_path is None:
            configured_input_size = _model_input_size_for_model(
                294,
                518,
                self.depth_resolution,
                ZIPDEPTH_PATCH_SIZE if _is_zipdepth_model(self.model_id) else DISTILL_ANY_DEPTH_PATCH_SIZE,
                self.model_id,
            )
        self._preprocessor = ModelOnnxPreprocessor(
            model_id=self.model_id,
            device=self.device,
            dtype=self.dtype,
            fixed_input_size=_input_size_from_artifact_name(self.onnx_path) if self._explicit_onnx_path else configured_input_size,
        )

    @property
    def pipeline_slot_count(self) -> int:
        return max(1, int(getattr(self._engine, "pipeline_slot_count", 1)))

    def _set_artifact_paths(self, onnx_path: Path, engine_path: Path, input_size: tuple[int, int]) -> None:
        dtype = _dtype_from_onnx_name(onnx_path, self.dtype)
        if onnx_path != self.onnx_path or engine_path != self.engine_path or dtype != self.dtype:
            if self._engine is not None:
                self._engine.close()
            self._engine = None
        self.onnx_path = onnx_path
        self.engine_path = engine_path
        self.dtype = dtype
        self._artifact_input_size = input_size
        self._preprocessor = ModelOnnxPreprocessor(
            model_id=self.model_id,
            device=self.device,
            dtype=self.dtype,
            fixed_input_size=input_size,
        )
        self.info = replace(self.info, onnx_path=str(self.onnx_path))

    def _ensure_artifacts_for_input(self, height: int, width: int) -> None:
        input_size = self._preprocessor.input_size(height, width)
        if self._explicit_onnx_path is not None or self._explicit_engine_path is not None:
            fixed_size = _input_size_from_artifact_name(self.onnx_path) or _input_size_from_artifact_name(self.engine_path) or input_size
            self._set_artifact_paths(self.onnx_path, self.engine_path, fixed_size)
            return
        if self._artifact_input_size == input_size and self.engine_path.exists():
            return
        cfg = DepthProviderConfig(
            model_id=self.model_id,
            model_name=self.model_name,
            device=self.device,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            force_download=self.force_download,
            onnx_dtype=self.onnx_dtype,
            build_engine=self.build_engine,
            force_rebuild=self.force_rebuild,
        )
        artifacts = _prepare_accelerated_artifacts(cfg, build_trt=self.build_engine or self.force_rebuild, input_size=input_size)
        if artifacts.selected_onnx_path is None and not artifacts.trt_ready:
            raise FileNotFoundError(f"ONNX artifact not found for {self.model_id} at input size {input_size}")
        source_path = Path(artifacts.selected_onnx_path) if artifacts.selected_onnx_path is not None else artifacts.paths.trt_path_for_dtype(self.onnx_dtype if self.onnx_dtype in ("fp16", "fp32") else "fp16")
        dtype_name = "fp32" if "fp32" in source_path.name.lower() else "fp16"
        self._set_artifact_paths(artifacts.paths.onnx_path_for_dtype(dtype_name), artifacts.paths.trt_path_for_dtype(dtype_name), input_size)

    def load(self) -> NativeTensorRtEngine | None:
        if self._engine is not None:
            self._preprocessor.fixed_input_size = self._engine.input_image_size
            return self._engine
        if self._explicit_onnx_path is None and self._explicit_engine_path is None and self._artifact_input_size is None:
            return None
        if self.build_engine or self.force_rebuild or not self.engine_path.exists():
            build_native_tensorrt_engine(self.onnx_path, self.engine_path, fp16=self.dtype == torch.float16, force=self.force_rebuild)
        trt_lib_dirs = ensure_tensorrt_dll_path()
        self._engine = NativeTensorRtEngine(
            self.engine_path,
            device=self.device,
            dtype=self.dtype,
            execution_slot_count=self.execution_slot_count,
        )
        self._preprocessor.fixed_input_size = self._engine.input_image_size
        compact_engine_path = Path(self.engine_path.parent.name) / self.engine_path.name
        print(
            "[TensorRT] native provider loaded:"
            f" engine={compact_engine_path}",
            flush=True,
        )
        self.info = replace(
            self.info,
            execution_provider="TensorRT native",
            trt_lib_dirs=trt_lib_dirs,
        )
        return self._engine

    def predict(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.predict_profile(rgb).depth

    def predict_profile(self, rgb: torch.Tensor) -> DepthProfileResult:
        def sync() -> None:
            if self.profile_sync and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)

        cuda_events: dict[str, object] = {}
        rgb = ensure_bchw(rgb, name="rgb")
        _, _, height, width = rgb.shape
        self._ensure_artifacts_for_input(height, width)
        engine = self.load()
        sync()
        _record_cuda_event(cuda_events, "depth_pre_start", rgb)
        start = time.perf_counter()
        tensor = self._preprocessor(rgb)
        sync()
        _record_cuda_event(cuda_events, "depth_pre_end", tensor)
        preprocess_ms = (time.perf_counter() - start) * 1000.0

        sync()
        _record_cuda_event(cuda_events, "depth_model_start", tensor)
        start = time.perf_counter()
        execution_slot: int | None = None
        try_cuda_graph = self.use_cuda_graph and self._cuda_graph_disabled_reason is None
        if try_cuda_graph:
            try:
                run_graph_with_slot = getattr(engine, "run_graph_with_slot", None)
                if callable(run_graph_with_slot):
                    predicted, execution_slot = run_graph_with_slot(tensor)
                else:
                    predicted = engine.run_graph(tensor)
            except RuntimeError as exc:
                self._cuda_graph_disabled_reason = f"{type(exc).__name__}: {exc}"
                clear_graph = getattr(engine, "clear_graph", None)
                if callable(clear_graph):
                    clear_graph()
                try:
                    torch.cuda.synchronize(self.device)
                except Exception:
                    pass
                print(
                    "\033[31m[TensorRT] CUDA graph capture failed; disabled for this process. "
                    f"Using native TensorRT enqueue until restart: {self._cuda_graph_disabled_reason}\033[0m",
                    flush=True,
                )
                try:
                    run_with_slot = getattr(engine, "run_with_slot", None)
                    if callable(run_with_slot):
                        predicted, execution_slot = run_with_slot(
                            tensor,
                            synchronize=self.profile_sync,
                        )
                    else:
                        predicted = engine(tensor, synchronize=self.profile_sync)
                except TypeError:
                    predicted = engine(tensor)
        else:
            try:
                run_with_slot = getattr(engine, "run_with_slot", None)
                if callable(run_with_slot):
                    predicted, execution_slot = run_with_slot(
                        tensor,
                        synchronize=self.profile_sync,
                    )
                else:
                    predicted = engine(tensor, synchronize=self.profile_sync)
            except TypeError:
                predicted = engine(tensor)
        sync()
        _record_cuda_event(cuda_events, "depth_model_end", predicted)
        model_elapsed_ms = (time.perf_counter() - start) * 1000.0
        slot_wait_ms = 0.0
        slot_wait = getattr(engine, "slot_acquire_wait_ms", None)
        if callable(slot_wait):
            slot_wait_ms = max(0.0, float(slot_wait(execution_slot)))
        model_ms = max(0.0, model_elapsed_ms - slot_wait_ms)

        try:
            start = time.perf_counter()
            _record_cuda_event(cuda_events, "depth_post_start", predicted)
            _record_cuda_event(cuda_events, "depth_norm_start", predicted)
            depth = ensure_b1hw(predicted.float())
            if _is_zipdepth_model(self.model_id):
                depth = depth.clamp_min(5e-3).reciprocal()
            depth = _normalize_depth(depth)
            _record_cuda_event(cuda_events, "depth_norm_end", depth)
            _record_cuda_event(cuda_events, "depth_upsample_start", depth)
            depth = upsample_depth(
                depth,
                height,
                width,
                rgb=rgb,
                mode=self.depth_upsample,
                edge_strength=self.depth_upsample_edge_strength,
            )
            sync()
            _record_cuda_event(cuda_events, "depth_upsample_end", depth)
            _record_cuda_event(cuda_events, "depth_post_end", depth)
            postprocess_ms = (time.perf_counter() - start) * 1000.0
            return DepthProfileResult(
                depth,
                preprocess_ms,
                model_ms,
                postprocess_ms,
                cuda_events,
                execution_slot=execution_slot,
                execution_slot_count=max(1, int(getattr(engine, "pipeline_slot_count", 1))),
                slot_wait_ms=slot_wait_ms,
            )
        finally:
            mark_slot_available = getattr(
                engine,
                "mark_slot_available_after_current_stream",
                None,
            )
            if callable(mark_slot_available):
                mark_slot_available(execution_slot)


NativeTensorRtDepthProvider = DistillAnyDepthBaseNativeTensorRt
