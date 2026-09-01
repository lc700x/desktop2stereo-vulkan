# 21 — RTMP/WebRTC Browser Black-Frame Fix: Full Debugging Experience (AMD ROCm)

**Status**: Fixed and verified. **Date**: 2026-09-01. **Scope**: AMD ROCm (`src\python3`, RX 9060 XT),
RTMP Streamer mode, WebRTC protocol, MediaMTX + FFmpeg + h264_amf. NVIDIA CUDA and macOS paths untouched.

## Symptom
- RTMP Streamer published fine: MediaMTX logged `[path live] stream is available and online, 1 track (H264)`,
  WebRTC sessions connected, tens of Mbps flowed (0 lost, 0 discarded).
- The browser player (`http://<host>:1122/live/`) stayed **black / "loading" forever**: no frame ever rendered.

## Root cause (3 layers, each masked the next)
1. **Encoder option bug** — `h264_amf` has no FFmpeg `preset` option; the shared hardware branch added
   `-preset fast` → AMF died at option-apply time. Fixed by stripping `-preset` in the AMF-only branch.
2. **`force_key_frames` expression** — the Windows block used `expr:eq(n%fps\,0)`; this FFmpeg build's
   evaluator rejects the `%` operator for every encoder ("Missing ')' or too many args") and the `\,` escaping
   is wrong for argv-list spawning. Fixed to `expr:eq(mod(n,fps),0)` (no backslash).
3. **AMF `ultralowlatency` emits only ONE true IDR** — THE browser killer. Wire-level NAL capture
   (interleaved RTSP reader) showed AMF `-usage ultralowlatency` produces **only 1 NAL type 5 (IDR) at stream
   start**; after that `-g` and `-force_key_frames` turn into **non-IDR I-slices** (open-GOP). A browser
   WebRTC H.264 depacketizer joining mid-stream never sees a keyframe → `framesReceived` stays 0 forever
   even though all RTP flows. CDP getStats confirmed: `framesReceived: 0`, `codec: null`, 50k+ packets.

### Encoder verification matrix (180 frames, NAL type-5 count)
| `-usage`         | True IDR (NAL 5) | Browser decode |
|------------------|------------------|----------------|
| ultralowlatency  | 1                | broken (black) |
| lowlatency       | 1                | broken         |
| **webcam**       | **6**            | **works**      |
| transcoding      | 6                | works          |

## The fix (`src\desktop2stereo\streaming\direct_sbs.py`, AMF branch)
- `WEBRTC + H.264` AMF now uses **`-usage webcam`** (low latency, honors GOP with true IDRs).
- `SRT/RTSP` headset paths keep `ultralowlatency` (their players tolerate it).
- `HEVC` AMF keeps defaults; NVIDIA (`_nvenc`/`_vulkan`) and macOS (`videotoolbox`) never enter `_amf`.

Also kept from layer 2/3: `-profile:v constrained_baseline` + `-level:v <computed>` for WebRTC H.264
(`_required_h264_level`/`_format_h264_level`: 5.2 for 4K@40, 4.0 for 1080p@30, 4.2 for 1080p@60) so the SPS
matches the browser-negotiated profile and is valid for the frame rate.

## Verification
- Real 4K AMF stream in headless Chrome (CDP): `framesDecoded: 200/200`, `videoReadyState: 4`,
  `videoWidth: 3840`, playing.
- libx264 control streams decode in the browser at any packet size; AMF `ultralowlatency` never did.

## Diagnostic tooling (kept under `src\.tmp` for reuse)
- `cdp_probe*.mjs` — headless Chrome + DevTools Protocol: video element state, `getStats()` (codec report,
  framesReceived/Decoded, decoderImplementation), real WHEP answer SDP capture via fetch hook.
- `rtsp_analyze*.py` — interleaved RTSP RTP capture: payload-type histogram, NAL type histogram,
  IDR/SPS/PPS presence per stream.

## Key lessons
- "Stream online + RTP flowing" does NOT mean the browser can decode. Always check `getStats()`:
  `framesReceived` (depacketizer) vs `framesDecoded` (decoder) vs `packetsReceived`.
- A `codec: null` / missing `codecId` in inbound-rtp means the browser never matched the RTP payload to a
  negotiated codec (PT mismatch or no keyframe).
- ffprobe's `pict_type=I` counts **non-IDR I-slices too** — it can report I=8 while the bitstream has only
  1 true IDR (NAL type 5). Verify with a NAL-type scan, not ffprobe pict_type.
- AMF usage modes are not interchangeable for live streaming: `ultralowlatency`/`lowlatency` sacrifice
  periodic IDR; `webcam`/`transcoding` keep them.
- `-c:v copy -f h264` producing 0 bytes ("nothing was encoded") is a fast signal the live source has no
  in-band keyframe to start from.

## Commits
- `f51c1da` fix(rtmp): AMF WebRTC uses 'webcam' usage so browsers get real IDR keyframes
- `4f0ce92` docs: AMF ultralowlatency only emits 1 IDR; webcam usage fixes browser decode
- Earlier in the chain: audio-input fallback (`f802c5b`), AMF preset + force_key_frames fix (`bea9e18`),
  CB profile + valid level (`a84d8f8`).
