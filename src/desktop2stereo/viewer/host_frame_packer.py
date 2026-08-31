"""Dedicated host-frame packer thread (macOS Local Viewer fast path).

Sits between the stereo pipeline and the viewer queue. The pipeline keeps
dispatching frames without ever blocking on an MPS->host copy; this thread
absorbs the sync wait per frame (the GPU stays fed by concurrent dispatch),
then hands the viewer a ready HWC RGBA8 numpy frame.

Without it the runtime loop alternates dispatch -> wait -> idle-GPU, capping
throughput at dispatch+wait instead of max(dispatch, wait).
"""
from __future__ import annotations

import dataclasses
import numpy as np
import os
import queue
import threading
import time
from typing import Any, Callable


class HostFramePacker:
    """Forward runtime results, packing device SBS -> host RGBA8 off-loop."""

    def __init__(
        self,
        in_q: Any,
        out_q: Any,
        *,
        on_stat: Callable[[str, float], None] | None = None,
        max_pending: int = 4,
    ) -> None:
        self.in_q = in_q
        self.out_q = out_q
        self.on_stat = on_stat or (lambda name, value: None)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._shape_logged = False

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="HostFramePacker", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # -- worker -------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self.in_q.get(timeout=0.05)
            except queue.Empty:
                continue
            # Pipeline publishes (runtime_result, capture_start_time).
            if isinstance(item, tuple) and len(item) == 2:
                result, stamp = item
            else:
                result, stamp = item, None
            if (
                not self._shape_logged
                and os.environ.get("D2S_PREP_TRACE", "0") in ("1", "true")
            ):
                print(
                    f"[packer-gate] got result sbs={getattr(result, 'sbs', None) is not None} "
                    f"vfn={getattr(result, 'viewer_frame_np', None) is not None}",
                    flush=True,
                )
            started = time.perf_counter()
            packed = None
            sbs = getattr(result, "sbs", None)
            deferred = (
                getattr(result, "viewer_rgb", None) is not None
                or getattr(result, "viewer_bgra", None) is not None
                or getattr(result, "viewer_frame_np", None) is not None
            )
            # Fused warp-pack for DEFERRED frames (darwin+vulkan): one Metal
            # kernel turns the shipped rgb+depth tensors into final Half-SBS
            # bytes here, hiding its cost behind next-frame preprocess.
            if deferred and os.environ.get(
                "D2S_VK_FUSED_WARP", "1"
            ) not in {"0", "false", "off"}:
                vr = getattr(result, "viewer_rgb", None)
                vd = getattr(result, "viewer_depth", None)
                if os.environ.get("D2S_FUSED_DEBUG"):
                    from stereo_runtime._fused_warp_mps import fused_enabled as _fe

                    print(
                        f"[fused-pack] deferred frame: vr={vr is not None} "
                        f"vd={vd is not None} enabled={_fe()} "
                        f"env={os.environ.get('D2S_MAC_VIEWER')!r}",
                        flush=True,
                    )
                if vr is not None and vd is not None:
                    try:
                        from stereo_runtime._fused_warp_mps import (
                            fused_enabled,
                            fused_sbs_pack,
                        )

                        if fused_enabled():
                            _host_out = None
                            try:
                                from utils.iosurface_stage import (
                                    acquire_stage,
                                )
                                from stereo_runtime._fused_warp_mps import (
                                    pack_target,
                                )

                                # Same contract as the NVIDIA local path:
                                # SBS geometry follows the runtime input
                                # resolution and the selected output format.
                                _fmt = str(
                                    getattr(result, "output_format", "")
                                    or "half_sbs"
                                )
                                _tw, _th = pack_target(
                                    int(vr.shape[-1]), int(vr.shape[-2]), _fmt
                                )
                                # Direct GPU-write staging first: warp
                                # straight into the viewer's mapped pages,
                                # removing the per-frame CPU memcpy from
                                # the present loop entirely.
                                _host_out = None
                                try:
                                    from viewer.direct_sink import (
                                        DIRECT_SINK,
                                    )

                                    _src, _dv = DIRECT_SINK.acquire(_tw, _th)
                                    if _src is not None and _dv is not None:
                                        _nbytes = _tw * _th * 4
                                        if len(_dv) >= _nbytes:
                                            _host_out = _dv[:_nbytes]
                                            object.__setattr__(
                                                result,
                                                "viewer_frame_direct",
                                                _src,
                                            )
                                            if not getattr(
                                                self,
                                                "_direct_logged",
                                                False,
                                            ):
                                                self._direct_logged = True
                                                print(
                                                    "[HostFramePacker] direct GPU-write into Vulkan staging active",
                                                    flush=True,
                                                )
                                except Exception:
                                    _host_out = None
                                if _host_out is None:
                                    object.__setattr__(
                                        result, "viewer_frame_direct", None
                                    )
                                    _st = acquire_stage(_tw, _th)
                                    _host_out = (
                                        _st[1].writable_view()
                                        if _st is not None
                                        else None
                                    )
                            except Exception:
                                _host_out = None
                            _t0 = time.perf_counter()
                            packed = fused_sbs_pack(
                                vr, vd, host_out=_host_out,
                                out_size=(_tw, _th),
                                output_format=_fmt,
                            )
                            if packed is None:
                                # Kernel failed after we claimed a direct
                                # slot: release it so the path stays alive.
                                _ds = getattr(result, "viewer_frame_direct", None)
                                if _ds is not None:
                                    object.__setattr__(
                                        result, "viewer_frame_direct", None
                                    )
                                    try:
                                        _ds._release_direct()
                                    except Exception:
                                        pass
                            if packed is not None:
                                # StereoRuntimeResult is frozen; the
                                # sanctioned late-binding escape hatch.
                                object.__setattr__(
                                    result, "viewer_frame_np", packed
                                )
                                # Diagnostics: D2S_DUMP_SBS=path dumps one
                                # packed Half-SBS frame as PPM. Read back via
                                # a fresh kernel->cpu pass: the stage IOSurface
                                # may be re-tiled once imported as a GPU
                                # texture, making CPU byte reads unreliable.
                                _dump = os.environ.get("D2S_DUMP_SBS")
                                if _dump and not os.path.exists(_dump):
                                    try:
                                        _v2 = fused_sbs_pack(
                                            vr, vd, out_size=(_tw, _th)
                                        )
                                        if _v2 is not None:
                                            _a2 = np.asarray(_v2[0])
                                            with open(_dump, "wb") as fh:
                                                fh.write(
                                                    f"P6\n{_tw} {_th}\n255\n".encode()
                                                )
                                                fh.write(
                                                    np.ascontiguousarray(
                                                        _a2[:, :, :3]
                                                    ).tobytes()
                                                )
                                            print(
                                                f"[sbs-dump] wrote {_dump}",
                                                flush=True,
                                            )
                                    except Exception:
                                        pass
                                # Sequential dumps for temporal-edge analysis.
                                _dump_dir = os.environ.get("D2S_DUMP_SBS_DIR")
                                if _dump_dir and not os.path.isdir(_dump_dir):
                                    try:
                                        os.makedirs(_dump_dir, exist_ok=True)
                                    except Exception:
                                        _dump_dir = None
                                if _dump_dir:
                                    _seq = getattr(self, "_d2s_dump_seq", 0) + 1
                                    self._d2s_dump_seq = _seq
                                    if _seq <= int(
                                        os.environ.get("D2S_DUMP_SBS_FRAMES", "60")
                                    ):
                                        try:
                                            _vs = fused_sbs_pack(
                                                vr, vd, out_size=(_tw, _th)
                                            )
                                            if _vs is not None:
                                                _as = np.asarray(_vs[0])
                                                with open(
                                                    os.path.join(
                                                        _dump_dir,
                                                        f"frame_{_seq:04d}.ppm",
                                                    ),
                                                    "wb",
                                                ) as fh:
                                                    fh.write(
                                                        f"P6\n{_tw} {_th}\n255\n".encode()
                                                    )
                                                    fh.write(
                                                        np.ascontiguousarray(
                                                            _as[:, :, :3]
                                                        ).tobytes()
                                                    )
                                        except Exception:
                                            pass
                                if isinstance(getattr(result, "timing", None), dict):
                                    result.timing["rt_fused_pack_ms"] = (
                                        time.perf_counter() - _t0
                                    ) * 1000.0
                    except Exception as exc:
                        if os.environ.get("D2S_FUSED_DEBUG"):
                            print(f"[fused-pack] failed: {exc!r}", flush=True)

            if sbs is not None and not deferred:
                try:
                    # Late-bound per frame so runtime reloads and patches are
                    # always picked up (binding at thread start races them).
                    from stereo_runtime.runtime import _pack_sbs_host_frame

                    packed = _pack_sbs_host_frame(sbs)
                    if (
                        packed is None
                        and os.environ.get("D2S_PREP_TRACE", "0") in ("1", "true")
                        and not self._shape_logged
                    ):
                        print(
                            f"[packer-gate] pack returned None "
                            f"sbs_shape={tuple(sbs.shape)} dtype={sbs.dtype} "
                            f"device={sbs.device}",
                            flush=True,
                        )
                except Exception as exc:
                    if os.environ.get("D2S_PREP_TRACE", "0") in ("1", "true"):
                        print(f"[packer-gate] pack exc: {exc}", flush=True)
                    packed = None
            if packed is not None:
                try:
                    timing = getattr(result, "timing", None)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    if isinstance(timing, dict):
                        timing["packer_ms"] = elapsed_ms
                    self.on_stat("packer_ms", elapsed_ms)
                    if not self._shape_logged:
                        self._shape_logged = True
                        arr = packed[0] if isinstance(packed, tuple) else packed
                        h, w = arr.shape[0], arr.shape[1]
                        print(
                            f"[HostFramePacker] packing {w}x{h} RGBA8 host frames",
                            flush=True,
                        )
                    result = dataclasses.replace(result, viewer_frame_np=packed)
                except Exception:
                    pass
            self._forward((result, stamp) if stamp is not None else result)

    def _forward(self, result: Any) -> None:
        out_put = getattr(self.out_q, "put", None)
        # Mirror the pipeline's latest-wins semantics on the consumer queue.
        put_latest = getattr(self.out_q, "put_latest", None)
        try:
            if callable(put_latest):
                put_latest(result)
                return
            if out_put is None:
                return
            try:
                out_put(result, timeout=0.05)
            except queue.Full:
                try:
                    self.out_q.get_nowait()  # drop oldest, keep newest
                    self.on_stat("packer_drop", 1)
                except queue.Empty:
                    pass
                out_put(result, timeout=0.05)
        except Exception:
            self.on_stat("packer_error", 1)


def install_host_frame_packer(
    pipeline_q: Any,
    viewer_q: Any,
    *,
    on_stat: Callable[[str, float], None] | None = None,
) -> HostFramePacker:
    """Create, start, and return a packer between pipeline and viewer."""
    packer = HostFramePacker(pipeline_q, viewer_q, on_stat=on_stat)
    packer.start()
    return packer


def maybe_install_local_viewer_packer(
    *,
    pipeline_q: Any,
    local_viewer_kwargs: dict,
    on_stat: Callable[[str, float], None] | None = None,
    os_name: str,
    host_frame_env: str | None = None,
    packer_env: str | None = None,
) -> bool:
    """Entry splice: gate + install, retargeting the viewer queue in place.

    Returns True when the packer was installed (viewer_kwargs now points at
    the packer's output queue); False when gated off.
    """
    if os_name != "Darwin":
        return False
    host_frame_env = (
        os.environ.get("D2S_VIEWER_HOST_FRAME", "0")
        if host_frame_env is None
        else host_frame_env
    )
    packer_env = (
        os.environ.get("D2S_PACKER_THREAD", "1")
        if packer_env is None
        else packer_env
    )
    if host_frame_env.strip() not in {"1", "true", "on"}:
        return False
    if packer_env.strip() in {"0", "false", "off"}:
        return False
    # Hand packing to the packer thread: runtime ships device SBS only.
    os.environ["D2S_RUNTIME_INLINE_HOST_PACK"] = "0"
    # Dedicated packer thread: the runtime loop keeps dispatching while this
    # thread absorbs the MPS->host sync wait, keeping the GPU fed instead of
    # alternating dispatch -> idle-GPU.
    if os.environ.get("D2S_PREP_TRACE", "0") in ("1", "true"):
        print(
            f"[packer-gate] installing host frame packer in_q_id={id(pipeline_q):x}",
            flush=True,
        )
    packed_q: "queue.Queue[Any]" = queue.Queue(maxsize=4)
    install_host_frame_packer(
        pipeline_q, packed_q, on_stat=on_stat
    )
    local_viewer_kwargs["runtime_q"] = packed_q
    return True
