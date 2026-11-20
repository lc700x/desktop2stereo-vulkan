"""Measure real media capture and successful presentation without changing saved settings."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src" / "desktop2stereo"
sys.path[:0] = [str(ROOT / "src"), str(APP)]
os.chdir(APP)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cuda", "rocm"], required=True)
    parser.add_argument("--mode", choices=["local", "openxr"], required=True)
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--media", type=Path, help="Video to test without changing saved settings")
    parser.add_argument("--start-position", type=float, default=15.0)
    parser.add_argument("--next-media", action="append", type=Path, default=[],
                        help="Open another video in the same session after --switch-after seconds")
    parser.add_argument("--next-position", action="append", type=float, default=[],
                        help="Source start position for each --next-media; defaults to --start-position")
    parser.add_argument("--switch-after", type=float, default=60.0)
    parser.add_argument("--pause-at", type=float, default=None, help="Seconds after the first presented frame")
    parser.add_argument("--pause-seconds", type=float, default=15)
    parser.add_argument("--output", required=True)
    parser.add_argument("--xr-runtime")
    parser.add_argument("--wait-before-release", action="store_true")
    parser.add_argument("--sampled-projection", action="store_true")
    parser.add_argument("--minimal-projection", action="store_true")
    parser.add_argument(
        "--capture-stages", action="store_true", help="Read back each eye once for black-screen diagnosis"
    )
    parser.add_argument("--isolate-tools", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    import torch

    expected_hip = args.backend == "rocm"
    if not torch.cuda.is_available() or bool(torch.version.hip) != expected_hip:
        raise SystemExit(f"Requested {args.backend} GPU is unavailable in this environment; refusing CPU fallback")
    for item in args.set:
        key, value = item.split("=", 1)
        os.environ[key] = value
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        D2S_CUDA_STAGE_EVENTS="1",
        D2S_FPS_BREAKDOWN="1",
        D2S_FPS_BREAKDOWN_INTERVAL="5",
        D2S_FPS_BREAKDOWN_LIMIT="0",
        D2S_OPENXR_SCREEN_INTERVAL_TRACE="1",
        D2S_MEDIA_RESUME_POSITION=str(max(0.0, args.start_position)),
        D2S_MEDIA_STATE_PATH=str(output.with_suffix(".state.json")),
        D2S_MEDIA_SESSION_ID=output.stem,
        D2S_MEDIA_CONTROL_PATH=str(output.with_suffix(".control.json")),
        D2S_MEDIA_HISTORY_PATH=str(output.with_suffix(".recent.json")),
    )
    if args.isolate_tools:
        os.environ.update(D2S_CONTROLLER_MODEL="none", D2S_OPENXR_DISABLE_TOOL_QUADS="1")
    if args.xr_runtime:
        os.environ["XR_RUNTIME_JSON"] = args.xr_runtime
    elif args.mode == "openxr":
        os.environ.pop("XR_RUNTIME_JSON", None)
    if args.backend == "cuda" and args.mode == "local":
        icds = list(Path("C:/Windows/System32/DriverStore/FileRepository").glob("nv_dispsi*/nv-vk64.json"))
        if not icds:
            raise RuntimeError("NVIDIA Vulkan ICD missing")
        icd = max(icds, key=lambda path: path.stat().st_mtime_ns)
        os.environ["VK_DRIVER_FILES"] = str(icd)
        os.environ["VK_ICD_FILENAMES"] = str(icd)
    elif args.mode == "openxr":
        # The runtime chooses its actual submission adapter; do not hide it
        # from xrGetVulkanGraphicsDevice2KHR when CUDA inference uses another GPU.
        os.environ.pop("VK_DRIVER_FILES", None)
        os.environ.pop("VK_ICD_FILENAMES", None)
    from windows_dpi import set_per_monitor_dpi_v2

    set_per_monitor_dpi_v2()
    import utils

    settings = dict(utils._get_settings())
    settings.update(
        {
            "Run Mode": "Local Viewer" if args.mode == "local" else "OpenXR Link",
            "Capture Mode": "File",
            "TensorRT": args.backend == "cuda",
            "MIGraphX": args.backend == "rocm",
            "Computing Device": 0,
            "XR Headset Model": "Meta Quest 2",
            "Parallel Inference": False,
            "Parallel Inference Workers": 1,
        }
    )
    if args.media is not None:
        media = args.media.resolve(strict=True)
        if not media.is_file():
            parser.error("--media must name a regular file")
        settings["Media Path"] = str(media)
    next_media = [path.resolve(strict=True) for path in args.next_media]
    if any(not path.is_file() for path in next_media):
        parser.error("--next-media must name regular files")
    if args.next_position and len(args.next_position) != len(next_media):
        parser.error("provide one --next-position per --next-media")
    next_positions = args.next_position or [args.start_position] * len(next_media)
    if any(not math.isfinite(position) or position < 0.0 for position in next_positions):
        parser.error("--next-position must be finite and nonnegative")
    utils._settings = settings
    utils._runtime_exports = None
    from stereo_runtime.hot_reload import StereoHotReloader

    StereoHotReloader.poll_settings_snapshot_if_needed = lambda *_a, **_k: None
    from app_runtime.runtime_callbacks import RuntimeCallbacks

    captures, presents, observations, scenario_events = [], [], [], []
    media_sessions = {}
    scenario_errors = []
    first_presented = threading.Event()
    scenario_stop = threading.Event()
    started = time.perf_counter()

    def record(target, metadata):
        metadata = metadata or {}
        target.append(
            {
                "t": time.perf_counter() - started,
                **{
                    key: metadata.get(key)
                    for key in (
                        "media_generation",
                        "source_frame_index",
                        "source_pts",
                        "media_playback_fps",
                        "media_terminal_partial",
                        "media_presentation_deadline",
                        "media_source_fps",
                        "capture_poll_returned_at",
                        "media_display_fps",
                        "media_feedback_clock",
                        "rt_depth_backend",
                        "media_depth_prefetch",
                        "depth_frame_id",
                        "depth_age_frames",
                        "runtime_ready_at",
                        "capture_to_runtime_ready_ms",
                        "runtime_queue_latency_ms",
                        "capture_to_runtime_start_ms",
                        "viewer_ready_at",
                        "runtime_completion_ms",
                        "render_size",
                        "runtime_depth_backend",
                        "runtime_output_eye_size",
                        "runtime_output_display_size",
                        "media_eye_output_size",
                        "media_packed_output_size",
                        "media_output_format",
                    )
                },
            }
        )

    raw_saved = False

    def save_tensor(name, tensor):
        import torch
        from PIL import Image

        if not isinstance(tensor, torch.Tensor):
            return
        data = tensor.detach().cpu()
        while data.ndim > 3:
            data = data[0]
        if data.ndim == 3 and data.shape[0] in (3, 4):
            data = data.permute(1, 2, 0)
        print(
            f"[Acceptance] tensor {name} shape={tuple(data.shape)} range={data.min().item()}/{data.max().item()}",
            flush=True,
        )
        if data.ndim == 3:
            if data.is_floating_point():
                data = (data.clamp(0, 1) * 255).to(torch.uint8)
            elif data.dtype != torch.uint8:
                return
            directory = output.parent / (output.stem + "-stages")
            directory.mkdir(parents=True, exist_ok=True)
            Image.fromarray(data.numpy()).save(directory / (name + ".png"))

    original_capture = RuntimeCallbacks.put_raw_latest

    def capture(self, item):
        nonlocal raw_saved
        metadata = getattr(item, "metadata", None) or {}
        if args.capture_stages and not raw_saved and int(metadata.get("source_frame_index") or 0) >= 240:
            save_tensor("00_raw_capture", item.frame)
            raw_saved = True
        record(captures, metadata)
        overwritten = original_capture(self, item)
        captures[-1]["overwritten"] = bool(overwritten)
        return overwritten

    RuntimeCallbacks.put_raw_latest = capture
    callback_owner = [None]
    original_init = RuntimeCallbacks.__init__

    def init_callback(self, *pos, **kwargs):
        original_init(self, *pos, **kwargs)
        callback_owner[0] = self

    RuntimeCallbacks.__init__ = init_callback
    original_session_update = RuntimeCallbacks.capture_session_update
    def session_update(self, session, control):
        previous = self.capture_session
        if session is None and previous is not None:
            generation = previous.media_generation
            media_sessions.setdefault(generation, {}).update(
                final_decode_backend=previous._decode_backend,
                final_decode_fallback_reason=previous.decode_fallback_reason,
                audio_error=previous.audio_error,
            )
        if session is not None and getattr(session, "_browser", None) is not None:
            # Test only requested files, not an implicit folder playlist after EOF.
            session._browser.playlist_clear()
            session._browser.playlist_add(session.path)
            session._browser.playlist_select(session.path)
        return original_session_update(self, session, control)
    RuntimeCallbacks.capture_session_update = session_update
    def sample_local_audio():
        while not scenario_stop.wait(0.5):
            owner = callback_owner[0]
            source = getattr(owner, "capture_session", None)
            if source is None:
                continue
            observations.append(
                {
                    "t": time.perf_counter() - started,
                    "playing": getattr(source, "playing", None),
                    "source_identity": getattr(source, "_presented_media_identity", None),
                    "captured": len(captures),
                    "decode_backend": source._decode_backend,
                    "decode_fallback_reason": source.decode_fallback_reason,
                    "audio_error": source.audio_error,
                    "audio_sync": source.audio_sync_metrics(),
                }
            )

    if args.mode == "local":
        threading.Thread(target=sample_local_audio, name="LocalAudioAcceptance", daemon=True).start()
    local_refresh = [0.0]
    xr_frame_info = {}
    original_feedback = RuntimeCallbacks.on_media_presented

    def on_feedback(self, metadata, presented_at):
        if int(metadata.get("source_frame_index") or 0) % 120 == 0 and not getattr(
            self, "_acceptance_clock_logged", False
        ):
            self._acceptance_clock_logged = True
            print(f"[Acceptance] video_display_clock={presented_at:.6f} now={time.perf_counter():.6f}", flush=True)
        result = original_feedback(self, metadata, presented_at)
        generation = metadata.get("media_generation")
        if generation is not None and generation not in media_sessions and self.capture_session is not None:
            source = self.capture_session
            media_sessions[generation] = {
                "path": source.path, "decode_backend": source._decode_backend,
                "decode_fallback_reason": source.decode_fallback_reason, "audio_enabled": source._has_audio,
                "width": source.width, "height": source.height, "hdr": source.hdr,
            }
        if args.mode in {"local", "openxr"}:
            identity = (metadata.get("media_generation"), metadata.get("source_frame_index"))
            previous = (presents[-1]["media_generation"], presents[-1]["source_frame_index"]) if presents else None
            if identity != previous:
                record(presents, metadata)
                presents[-1]["predicted_present_time_s"] = float(presented_at)
                presents[-1]["presentation_at_relative_s"] = float(presented_at) - started
                presents[-1]["refresh_hz"] = local_refresh[0]
                ready_at = metadata.get("runtime_ready_at")
                if ready_at is not None:
                    presents[-1]["runtime_to_present_ms"] = (float(presented_at) - float(ready_at)) * 1000.0
                if args.mode == "openxr":
                    presents[-1].update(xr_frame_info)
                first_presented.set()
        return result

    RuntimeCallbacks.on_media_presented = on_feedback
    from capture.media import MediaCaptureSource

    original_control = MediaCaptureSource._read_control
    def read_control(self):
        before = dict(self._last_control_revisions)
        generation = self.media_generation
        result = original_control(self)
        for path, revision in self._last_control_revisions.items():
            if before.get(path) == revision:
                continue
            try:
                request = json.loads(Path(path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if request.get("revision") == revision:
                clock = self.media_presentation_clock()
                phase_at = clock.get("paused_at") if request.get("action") == "pause" else (
                    clock.get("resumed_at") if request.get("action") == "play" else None
                )
                scenario_events.append({"t": time.perf_counter() - started,
                                        "clock_t": phase_at - started if phase_at is not None else None,
                                        "action": "control_applied",
                                        "media_generation": generation, "new_generation": self.media_generation,
                                        "command": request.get("action"), "value": request.get("value"),
                                        "control_channel": Path(path).suffix})
        return result
    MediaCaptureSource._read_control = read_control
    original_open_media = MediaCaptureSource._open_control_media
    def open_media(self, value):
        result = original_open_media(self, value)
        if result and getattr(self, "_browser", None) is not None:
            # Multi-video tests switch explicitly after measuring each EOF;
            # do not auto-step an older managed queue into an unrequested file.
            self._browser.playlist_clear()
            self._browser.playlist_add(self.path)
            self._browser.playlist_select(self.path)
        return result
    MediaCaptureSource._open_control_media = open_media
    original_finish = MediaCaptureSource._finish_video_stream
    def finish_video(self):
        generation, path, duration = self.media_generation, self.path, self.duration
        event_at = time.perf_counter() - started
        result = original_finish(self)
        scenario_events.append({"t": event_at, "action": "eof",
                                "media_generation": generation, "path": path, "duration": duration})
        return result
    MediaCaptureSource._finish_video_stream = finish_video
    if args.mode == "openxr":
        from xr_viewer.core_openxr_vulkan import OpenXrVulkanPresenter

        if args.minimal_projection:
            from xr_viewer.core_openxr_vulkan import OpenXrCompositionBuilder

            def projection(self, views, output_frame=None):
                targets = []
                for eye in self.swapchains:
                    index = self.xr.acquire_swapchain_image(eye.handle)
                    self.xr.wait_swapchain_image(
                        eye.handle, self.xr.SwapchainImageWaitInfo(timeout=self.xr.INFINITE_DURATION)
                    )
                    targets.append((eye, index))
                timelines = [
                    self.vulkan.clear_color_image(eye.resources[index].image, color)
                    for (eye, index), color in zip(targets, ((1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0)))
                ]
                self.vulkan.wait_for_timeline(max(timelines))
                for eye, index in targets:
                    self.xr.release_swapchain_image(eye.handle)
                return OpenXrCompositionBuilder(self.xr, self.reference_space).projection_layer(views, self.swapchains)

            OpenXrVulkanPresenter._render_projection_layer = projection
            OpenXrVulkanPresenter._render_quad_layers = lambda *_a: []
        if args.sampled_projection:
            original_create = OpenXrVulkanPresenter._create_projection_swapchain

            def create_projection(self, width, height, **kwargs):
                kwargs["sampled"] = True
                return original_create(self, width, height, **kwargs)

            OpenXrVulkanPresenter._create_projection_swapchain = create_projection
        if args.wait_before_release:
            original_initialize = OpenXrVulkanPresenter.initialize

            def initialize(self):
                result = original_initialize(self)
                original_release = self.xr.release_swapchain_image

                def release(handle, *pos, **kwargs):
                    timeline = self.vulkan.last_submitted_timeline_value
                    if timeline:
                        self.vulkan.wait_for_timeline(timeline)
                    return original_release(handle, *pos, **kwargs)

                self.xr.release_swapchain_image = release
                return result

            OpenXrVulkanPresenter.initialize = initialize
        if args.capture_stages:
            original_convert = OpenXrVulkanPresenter._submit_runtime_result_on_presenter

            def convert(self, runtime_result, timestamp):
                debug_info = getattr(runtime_result, "debug_info", None)
                if (
                    isinstance(debug_info, dict)
                    and len(self._visual_regression_capture_eyes) < 2
                    and int(debug_info.get("source_frame_index") or 0) >= 240
                ):
                    if not getattr(self, "_acceptance_output_saved", False):
                        save_tensor("02_runtime_left_eye", getattr(runtime_result, "left_eye", None))
                        self._acceptance_output_saved = True
                    debug_info["visual_regression_dir"] = str(output.parent / (output.stem + "-stages"))
                return original_convert(self, runtime_result, timestamp)

            OpenXrVulkanPresenter._submit_runtime_result_on_presenter = convert
        if args.capture_stages:
            original_overlay = OpenXrVulkanPresenter._render_filament_controller_overlay

            def overlay(self, acquired_images, record_time, render_views=None):
                timeline = original_overlay(self, acquired_images, record_time, render_views)
                frame = self._rendering_output or self._displayed_output
                index = int((getattr(frame, "metadata", None) or {}).get("source_frame_index") or 0)
                if not getattr(self, "_acceptance_final_saved", False) and index >= 240:
                    from viewer.vulkan_context import ImageState
                    from viewer.vulkan_resources import VulkanHostImage

                    vk = self.vulkan.vk
                    folder = output.parent / (output.stem + "-stages")
                    folder.mkdir(parents=True, exist_ok=True)
                    for eye_index, (eye, image_index) in enumerate(acquired_images):
                        resource = eye.resources[image_index]
                        host = VulkanHostImage(
                            self.vulkan, eye.width, eye.height, format=int(resource.format), readback=True
                        )
                        try:
                            self.vulkan.register_image_state(
                                resource.image,
                                ImageState(
                                    layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    access_mask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                    stage_mask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                    queue_family_index=self.vulkan.queue_family_index,
                                ),
                            )
                            readback = self.vulkan.copy_image(resource, host.resource, destination_host_readable=True)
                            self.vulkan.wait_for_timeline(readback)
                            self._save_visual_regression_host_image(host, folder / f"07_final_eye_{eye_index}.png")
                        finally:
                            host.close()
                    self._acceptance_final_saved = True
                return timeline

            OpenXrVulkanPresenter._render_filament_controller_overlay = overlay
        original_end = OpenXrVulkanPresenter._end_openxr_frame

        def end_frame(self, display_time, layer_pointers, **kwargs):
            local_refresh[0] = 1.0 / max(1e-6, self._display_period_s or self._last_frame_dt)
            xr_frame_info.update(
                predicted_xr_time_s=self._media_present_time_s,
                session_state=int(self.session_state or 0), layers=len(layer_pointers),
                xr_swapchain_eye_sizes=[[eye.width, eye.height] for eye in self.swapchains],
            )
            displayed = self._displayed_output
            if displayed is not None:
                xr_frame_info["eye_image_sizes"] = [
                    [int(getattr(eye, "width", 0)), int(getattr(eye, "height", 0))]
                    for eye in (displayed.left_eye, displayed.right_eye)
                ]
            if getattr(self, "_acceptance_end_count", 0) % 60 == 0:
                displayed = self._displayed_output
                observations.append(
                    {
                        "t": time.perf_counter() - started,
                        "layers": len(layer_pointers),
                        "displayed_frame_id": getattr(displayed, "frame_id", None),
                        "source_identity": self._output_source_identity(displayed) if displayed is not None else None,
                        "session_state": int(self.session_state or 0),
                        "captured": len(captures),
                        "playing": self._media_overlay_state.get("playing"),
                        "audio_sync": (
                            callback_owner[0].capture_session.audio_sync_metrics()
                            if callback_owner[0] is not None
                            and callable(getattr(callback_owner[0].capture_session, "audio_sync_metrics", None))
                            else {}
                        ),
                    }
                )
            if getattr(self, "_acceptance_end_count", 0) % 120 == 0:
                print(
                    f"[Acceptance] xrEndFrame layers={len(layer_pointers)} "
                    f"state={getattr(self, 'session_state', None)}",
                    flush=True,
                )
            self._acceptance_end_count = getattr(self, "_acceptance_end_count", 0) + 1
            return original_end(self, display_time, layer_pointers, **kwargs)

        OpenXrVulkanPresenter._end_openxr_frame = end_frame
        original_commit = OpenXrVulkanPresenter._commit_output_frame

        def commit(self, frame):
            result = original_commit(self, frame)
            if len(presents) == 0 or len(presents) % 120 == 0:
                print(
                    "[Acceptance] placement=" + json.dumps({
                        "t": time.perf_counter() - started,
                        "source_index": (frame.metadata or {}).get("source_frame_index"),
                        "screen_pose": self._filament_screen,
                        "head_position": self._head_position_w.tolist() if self._head_position_w is not None else None,
                    }), flush=True,
                )
            return result

        OpenXrVulkanPresenter._commit_output_frame = commit
    else:
        from viewer.vulkan_local_viewer import VulkanLocalViewer

        original_observe = VulkanLocalViewer.observe_media_presentation

        def observe(self, result):
            original_observe(self, result)
            local_refresh[0] = self._output_refresh_hz
            metadata = getattr(result, "debug_info", None)
            if isinstance(metadata, dict):
                metadata["media_eye_output_size"] = getattr(result, "output_eye_size", None)
                metadata["media_packed_output_size"] = getattr(result, "output_display_size", None)
                metadata["media_output_format"] = getattr(result, "output_format", None)

        VulkanLocalViewer.observe_media_presentation = observe
    print(
        "[Acceptance] config="
        + json.dumps(
            {
                k: settings.get(k)
                for k in (
                    "Run Mode",
                    "Depth Model",
                    "Depth Resolution",
                    "Stereo Quality",
                    "Render Scale",
                    "TensorRT",
                    "MIGraphX",
                    "Media Path",
                )
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    from app_runtime.runtime_entry import run_processing_runtime

    def playback_scenario():
        while not first_presented.wait(0.1):
            if scenario_stop.is_set():
                return
        revision = 0
        control_path = Path(os.environ["D2S_MEDIA_CONTROL_PATH"])

        def send(action, value=None):
            nonlocal revision
            revision += 1
            request = {"session": output.stem, "revision": revision, "action": action, "value": value}
            temporary = control_path.with_suffix(control_path.suffix + ".tmp")
            temporary.write_text(json.dumps(request), encoding="utf-8")
            for attempt in range(10):
                try:
                    os.replace(temporary, control_path)
                    break
                except PermissionError:
                    if attempt == 9:
                        raise
                    time.sleep(0.01 * (attempt + 1))
            event = {"t": time.perf_counter() - started, "action": action,
                     "media_generation": presents[-1]["media_generation"], "value": value}
            scenario_events.append(event)
            print("[Acceptance] scenario=" + json.dumps(event), flush=True)

        for index in range(len(next_media) + 1):
            video_started = time.perf_counter()
            if args.pause_at is not None:
                if scenario_stop.wait(max(0.0, args.pause_at)):
                    return
                send("pause")
                if scenario_stop.wait(max(0.0, args.pause_seconds)):
                    return
                send("play")
            if index == len(next_media):
                return
            if scenario_stop.wait(max(0.0, args.switch_after - (time.perf_counter() - video_started))):
                return
            generation = presents[-1]["media_generation"]
            send("open", {"path": str(next_media[index]), "mode": "File", "position": next_positions[index]})
            while presents[-1]["media_generation"] == generation:
                if scenario_stop.wait(0.1):
                    return

    def execute_scenario():
        try:
            playback_scenario()
        except Exception as exc:  # noqa: BLE001 - export every daemon-thread failure instead of a false pass
            error = f"{type(exc).__name__}: {exc}"
            scenario_errors.append(error)
            print("[Acceptance] scenario_failed=" + error, flush=True)
    if args.pause_at is not None or next_media:
        threading.Thread(target=execute_scenario, name="MediaPlaybackAcceptance", daemon=True).start()
    rc = None
    try:
        rc = run_processing_runtime(max_seconds=args.seconds)
    finally:
        scenario_stop.set()
        if not presents and rc == 0:
            rc = 3
        observed_paths = {info.get("path") for info in media_sessions.values()}
        missing_paths = [str(path) for path in next_media if str(path) not in observed_paths]
        if rc == 0 and (scenario_errors or missing_paths):
            rc = 4
        output.write_text(
            json.dumps(
                {
                    "media_sessions": media_sessions,
                    "verification": {
                        "presentation_clock": "successful_submission_perf_counter" if args.mode == "local"
                        else "xr_predicted_display_time_qpc_or_explicit_fallback",
                        "readback_enabled": bool(args.capture_stages),
                        "stage_events_enabled": os.environ.get("D2S_CUDA_STAGE_EVENTS"),
                    },
                    "requested_media_paths": [settings["Media Path"], *(str(path) for path in next_media)],
                    "backend": args.backend,
                    "mode": args.mode,
                    "rc": rc,
                    "gpu": torch.cuda.get_device_name(0),
                    "media": {
                        "decode_backend": getattr(callback_owner[0].capture_session, "_decode_backend", None),
                        "decode_fallback_reason": getattr(callback_owner[0].capture_session, "decode_fallback_reason", None),
                        "audio_enabled": getattr(callback_owner[0].capture_session, "_has_audio", None),
                        "audio_error": getattr(callback_owner[0].capture_session, "audio_error", None),
                        "width": getattr(callback_owner[0].capture_session, "width", None),
                        "height": getattr(callback_owner[0].capture_session, "height", None),
                    } if callback_owner[0] is not None else {},
                    "settings": {
                        key: settings.get(key)
                        for key in (
                            "Depth Model", "Depth Resolution", "Stereo Quality", "Render Scale", "Media Path"
                        )
                    },
                    "elapsed": time.perf_counter() - started,
                    "perf_counter_start": started,
                    "captures": captures,
                    "presents": presents,
                    "observations": observations,
                    "scenario_events": scenario_events,
                    "scenario_errors": scenario_errors,
                    "missing_media_paths": missing_paths,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"[Acceptance] captures={len(captures)} presents={len(presents)} rc={rc}", flush=True)
    raise SystemExit(int(rc or 0))


if __name__ == "__main__":
    main()
