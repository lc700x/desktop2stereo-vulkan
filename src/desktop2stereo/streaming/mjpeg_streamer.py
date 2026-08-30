import threading
import time
import os
from typing import Any, Optional, Tuple
import numpy as np
import cv2
from socketserver import ThreadingMixIn
from wsgiref.simple_server import make_server, WSGIServer
from utils.network import get_local_ip
from streaming.encoder_profile import EncoderProfile

# Path to favicon file
ICON_PATH = "icon2.ico"

# Custom WSGI server class that supports threading
# Modified: Added connection limiting to reduce server pressure from too many concurrent clients
class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    allow_reuse_address = True
    block_on_close = False
    max_connections = 10  # Limit concurrent clients to prevent overload

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_connections = 0
        self.connection_lock = threading.Lock()

    def process_request(self, request, client_address):
        with self.connection_lock:
            if self.active_connections >= self.max_connections:
                request.close()
                return  # Gracefully reject excess connections
            self.active_connections += 1
        try:
            super().process_request(request, client_address)
        finally:
            with self.connection_lock:
                self.active_connections -= 1


class MJPEGStreamer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 1303,
        fps: int = 60,
        quality: int = 90,
        profile: EncoderProfile | None = None,
        display_mode: str = "Half-SBS",
        fit_mode: str = "contain",
        input_size: Tuple[int, int] | None = None,
    ):
        """
        Initialize the MJPEG streamer with configuration parameters.
        Legacy fps/quality arguments remain supported; EncoderProfile is the
        canonical transport contract.
        """
        self.profile = profile or EncoderProfile(codec="mjpeg", quality=quality, target_fps=fps)
        if self.profile.codec != "mjpeg":
            raise ValueError(f"MJPEGStreamer requires an mjpeg profile, got {self.profile.codec}")
        # MJPEG stream boundary marker
        self.boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        self.quality = int(self.profile.quality)
        self.requested_fps = int(self.profile.target_fps)
        self.fps = self.requested_fps  # actual fps (updated after probe)
        self.delay = 1.0 / max(1, self.fps)

        self.raw_frame: Optional[np.ndarray] = None       # Latest frame (numpy RGB)
        self.cuda_frame: Any = None                       # Latest frame (CUDA tensor)
        self.cuda_event: Any = None                       # CUDA event for sync
        self.encoded_frame: Optional[bytes] = None        # Latest JPEG
        self.lock = threading.Lock()

        self.shutdown = threading.Event()
        self.new_raw_event = threading.Event()
        self.new_encoded_event = threading.Event()
        self._started = False

        # Stream dimensions
        self.sbs_width: Optional[int] = None
        self.sbs_height: Optional[int] = None
        self.index_bytes: Optional[bytes] = None

        # Aspect ratio config (from local viewer)
        self.display_mode = str(display_mode or "Half-SBS").strip()
        self.fit_mode = str(fit_mode or "contain").strip()
        self.input_size = input_size

        # HTML template with auto reconnect and fullscreen
        self.template = """<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="icon" type="image/x-icon" href="./favicon.ico">
        <title>Desktop2Stereo Streamer</title>
        <script>
            const FPS = {fps};
            const WIDTH = {width};
            const HEIGHT = {height};
            const STREAM_URI = "/stream.mjpg";

            window.onload = () => {{
                const video = document.getElementById("player-canvas");
                let canvas = document.createElement("canvas");
                let ctx = canvas.getContext("2d");
                let canvasStream = null;

                const img = new Image();
                img.crossOrigin = "anonymous";
                img.src = STREAM_URI;

                function ensureCanvasSize(w, h) {{
                    if (!w || !h) return;
                    if (canvas.width !== w || canvas.height !== h) {{
                        canvas.width = w;
                        canvas.height = h;
                        if (canvasStream) {{
                            try {{ canvasStream.getTracks().forEach(t => t.stop()); }} catch(e) {{}}
                        }}
                        try {{
                            canvasStream = canvas.captureStream(FPS || 30);
                            video.srcObject = canvasStream;
                        }} catch(e) {{}}
                    }}
                }}

                let last_timestamp = 0;
                img.onload = () => {{
                    const w = img.naturalWidth || img.width || canvas.width;
                    const h = img.naturalHeight || img.height || canvas.height;
                    ensureCanvasSize(w, h);

                    function render(timestamp) {{
                        if (timestamp - last_timestamp < 1000.0 / (FPS || 30)) {{
                            requestAnimationFrame(render);
                            return;
                        }}
                        last_timestamp = timestamp;
                        try {{ ctx.drawImage(img, 0, 0, canvas.width, canvas.height); }} catch(e) {{ console.error("Failed to draw frame:", e); }}
                        requestAnimationFrame(render);
                    }}

                    requestAnimationFrame(render);
                }};

                img.onerror = () => {{
                    setTimeout(() => {{
                        img.src = STREAM_URI + "?t=" + new Date().getTime();
                    }}, 1000);
                }};

                canvas.style.display = 'none';
                document.body.appendChild(canvas);
            }};
        </script>
        <style type="text/css">
            body {{ margin: 0; background-color: rgb(45,48,53); }}
            .video-container {{ position: fixed; left: 0; top: 0; width: 100vw; height: 100vh; display:flex; align-items:center; justify-content:center; }}
            .video {{ max-height:100%; max-width:100%; width:auto; height:auto; background: black; }}
        </style>
    </head>
    <body>
        <div class="video-container">
            <video id="player-canvas" class="video" controls controlsList="nodownload"
                autoplay loop muted poster="" disablepictureinpicture ></video>
        </div>
    </body>
</html>
"""

        # WSGI application handler
        def app(environ, start_response):
            path = environ.get("PATH_INFO", "/")

            if path == "/":
                if self.index_bytes is None:
                    start_response("503 Service Unavailable", [("Content-Type", "text/plain")])
                    return [b"Stream not ready yet"]
                start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
                return [self.index_bytes]

            if path == "/stream.mjpg":
                start_response("200 OK", [("Content-Type", "multipart/x-mixed-replace; boundary=frame")])
                return self._generate()

            if path == "/favicon.ico" and os.path.exists(ICON_PATH):
                with open(ICON_PATH, "rb") as f:
                    data = f.read()
                start_response("200 OK", [("Content-Type", "image/x-icon")])
                return [data]

            start_response("404 Not Found", [("Content-Type", "text/plain")])
            return [b"Not Found"]

        # Create WSGI server instance
        self.server = make_server(host, port, app, ThreadingWSGIServer)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.encoder_thread = threading.Thread(target=self._encoder_loop, daemon=True)

    def start(self):
        print(f"[MJPEGStreamer] serving on http://{get_local_ip()}:{self.server.server_address[1]}/")
        self.server_thread.start()
        self.encoder_thread.start()
        self._started = True

    def stop(self):
        self.shutdown.set()
        try:
            if self._started:
                self.server.shutdown()
            self.server.server_close()
        except Exception:
            pass
        self.new_raw_event.set()

    def _transport_frame_size(self, frame_np: np.ndarray) -> Tuple[int, int]:
        h, w = frame_np.shape[:2]
        if self.profile.resize_size is not None:
            return self.profile.resize_size
        # Mirror local viewer / legacy fill_16_9: contain pads each eye into a
        # 16:9 canvas, cover/stretch keep the original input aspect.
        from streaming.aspect import transport_canvas_size

        return transport_canvas_size(
            (w, h),
            self.fit_mode,
            input_size=self.input_size,
            display_mode=self.display_mode,
        )

    def set_frame(self, frame_np: np.ndarray):
        """
        Set the current CPU frame to be streamed.
        """
        with self.lock:
            w, h = self._transport_frame_size(frame_np)
            if (self.sbs_width, self.sbs_height) != (w, h):
                self.sbs_width = w
                self.sbs_height = h
                try:
                    self.index_bytes = self.template.format(
                        fps=self.fps,
                        width=self.sbs_width,
                        height=self.sbs_height
                    ).encode("utf-8")
                except Exception:
                    self.index_bytes = b"<html><body>Desktop2Stereo Streamer</body></html>"

            self.raw_frame = frame_np
            self.cuda_frame = None
            self.cuda_event = None
            self.new_raw_event.set()

    @staticmethod
    def _hw_of(tensor: Any) -> Tuple[int, int]:
        """Return (h, w) for a CUDA/CPU tensor in CHW or HWC layout."""
        if tensor.ndim == 4:
            if int(tensor.shape[-1]) in (1, 3, 4):
                return int(tensor.shape[-3]), int(tensor.shape[-2])  # B,H,W,C
            return int(tensor.shape[-2]), int(tensor.shape[-1])  # B,C,H,W
        if tensor.ndim == 3:
            if int(tensor.shape[0]) in (1, 3, 4) and int(tensor.shape[-1]) not in (1, 3, 4):
                return int(tensor.shape[-2]), int(tensor.shape[-1])  # C,H,W
            return int(tensor.shape[0]), int(tensor.shape[1])  # H,W,C
        raise ValueError(f"unexpected tensor shape {tuple(tensor.shape)}")

    def set_cuda_frame(self, cuda_tensor: Any, cuda_event: Any = None):
        """
        Set the current CUDA frame to be streamed (zerocopy path).
        cuda_tensor: torch.Tensor on CUDA, shape [H,W,3] or [1,3,H,W] or [3,H,W] (uint8 or float32 0..1)
        cuda_event: optional torch.cuda.Event to wait for before processing
        """
        with self.lock:
            # Determine source size from tensor
            h, w = self._hw_of(cuda_tensor)

            if self.profile.resize_size is not None:
                w, h = self.profile.resize_size
            else:
                from streaming.aspect import transport_canvas_size

                w, h = transport_canvas_size(
                    (w, h),
                    self.fit_mode,
                    input_size=self.input_size,
                    display_mode=self.display_mode,
                )

            if (self.sbs_width, self.sbs_height) != (w, h):
                self.sbs_width = w
                self.sbs_height = h
                try:
                    self.index_bytes = self.template.format(
                        fps=self.fps,
                        width=self.sbs_width,
                        height=self.sbs_height
                    ).encode("utf-8")
                except Exception:
                    self.index_bytes = b"<html><body>Desktop2Stereo Streamer</body></html>"

            self.raw_frame = None
            self.cuda_frame = cuda_tensor
            self.cuda_event = cuda_event
            self.new_raw_event.set()

    def _encoder_loop(self):
        """
        Encoder loop: process CUDA frames (zerocopy aspect+resize) or CPU frames.
        JPEG encoding stays on CPU (cv2.imencode).
        """
        import torch

        # Pre-allocate pinned host buffer for CUDA->CPU download
        host_buffer: Optional[np.ndarray] = None

        while not self.shutdown.is_set():
            if not self.new_raw_event.wait(timeout=0.02):
                continue
            self.new_raw_event.clear()

            cuda_frame = None
            cuda_event = None
            raw_frame = None

            with self.lock:
                if self.cuda_frame is not None:
                    cuda_frame = self.cuda_frame
                    cuda_event = self.cuda_event
                    self.cuda_frame = None
                    self.cuda_event = None
                elif self.raw_frame is not None:
                    raw_frame = self.raw_frame
                    self.raw_frame = None
                else:
                    continue

            try:
                # Without a resize profile the canvas equals the source size, so
                # cover/stretch must pass the frame through (same as local viewer
                # on a matching canvas) instead of cropping against itself.
                from streaming.aspect import normalize_display_fit_mode
                effective_fit = self.fit_mode
                if (
                    self.profile.resize_size is None
                    and normalize_display_fit_mode(self.fit_mode) != "contain"
                ):
                    effective_fit = "stretch"

                if cuda_frame is not None:
                    # A CUDA tensor implies CUDA is available; GPU letterboxing
                    # is mandatory here (never route to the CPU aspect path).
                    # Wait for producer event if provided
                    if cuda_event is not None:
                        cuda_event.synchronize()

                    # Apply aspect/ratio on GPU, result is [H_t, W_t, 3] uint8 CUDA
                    from streaming.aspect import apply_aspect_on_gpu
                    src_h, src_w = self._hw_of(cuda_frame)
                    processed = apply_aspect_on_gpu(
                        cuda_frame,
                        source_size=(src_w, src_h),
                        target_size=(self.sbs_width, self.sbs_height),
                        fit_mode=effective_fit,
                        display_mode=self.display_mode,
                        input_size=self.input_size,
                    )

                    # Download to pinned host buffer (single copy, no CPU resize)
                    if host_buffer is None or host_buffer.shape != (processed.shape[0], processed.shape[1], 3):
                        host_buffer = np.empty(
                            (processed.shape[0], processed.shape[1], 3),
                            dtype=np.uint8,
                        )
                    torch.cuda.current_stream(processed.device).synchronize()
                    np.copyto(host_buffer, processed.cpu().numpy(), casting="unsafe")

                    # JPEG encode on CPU (contiguous, correct size)
                    # OpenCV expects BGR, but host_buffer is RGB from GPU path
                    host_buffer_bgr = host_buffer[..., ::-1]
                    success, buf = cv2.imencode(".jpg", host_buffer_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                    if success:
                        with self.lock:
                            self.encoded_frame = buf.tobytes()
                            self.new_encoded_event.set()

                elif raw_frame is not None:
                    # CPU fallback path
                    from streaming.aspect import apply_aspect_on_cpu
                    processed = apply_aspect_on_cpu(
                        raw_frame,
                        source_size=(raw_frame.shape[1], raw_frame.shape[0]),
                        target_size=(self.sbs_width, self.sbs_height),
                        fit_mode=effective_fit,
                        display_mode=self.display_mode,
                        input_size=self.input_size,
                    )
                    # OpenCV expects BGR
                    processed_bgr = processed[..., ::-1]
                    success, buf = cv2.imencode(".jpg", processed_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                    if success:
                        with self.lock:
                            self.encoded_frame = buf.tobytes()
                            self.new_encoded_event.set()

            except Exception as e:
                print(f"[MJPEGStreamer] encoder error: {type(e).__name__}: {e}", flush=True)

    def _generate(self):
        next_frame_time = time.perf_counter()
        while not self.shutdown.is_set():
            if not self.new_encoded_event.wait(timeout=0.02):
                continue
            self.new_encoded_event.clear()

            with self.lock:
                f = self.encoded_frame
                if not f:
                    continue

            yield self.boundary + f + b"\r\n"

            # Enforce consistent pacing
            next_frame_time += self.delay
            sleep_time = next_frame_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_frame_time = time.perf_counter()
        yield b""

    def _prepare_frame_for_jpeg(self, arr: np.ndarray) -> np.ndarray:
        """Legacy CPU-only path (kept for compatibility)."""
        if arr is None:
            return arr
        frame = arr
        if self.profile.resize_size is not None:
            frame = cv2.resize(frame, self.profile.resize_size, interpolation=cv2.INTER_AREA)
        if self.profile.pixel_format == "rgb":
            frame = frame[..., ::-1]
        elif self.profile.pixel_format == "bgra":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif self.profile.pixel_format == "bgr":
            pass
        else:
            raise ValueError("MJPEGStreamer only accepts rgb, bgr, or bgra packed frames")
        return np.ascontiguousarray(frame)

    def encode_jpeg(self, arr: np.ndarray) -> bytes:
        if arr is None:
            return b""
        frame_for_jpeg = self._prepare_frame_for_jpeg(arr)
        success, buf = cv2.imencode(".jpg", frame_for_jpeg, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return buf.tobytes() if success else b""