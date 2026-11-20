# FFmpeg D3D11VA GPU decoder bridge

This Windows-only DLL decodes a video through FFmpeg D3D11VA, converts each
decoded frame to SDR/BT.709 with the D3D11 Video Processor, and exposes a small
ring of shared BGRA8 textures to Python.

The bridge is used by `capture.media_d3d11`:

1. FFmpeg decodes into a D3D11 hardware surface.
2. The D3D11 Video Processor performs GPU-resident HDR/PQ or HLG/BT.2020 to
   SDR/BT.709 conversion. SDR full/studio range is selected from the AVFrame.
3. The bridge writes BGRA8 into shared textures and signals a D3D11 fence.
4. HIP imports the texture and fence, copies directly into a preallocated ROCm
   tensor, and signals the consumed fence. No CPU pixel readback is used.
5. Python reuses one tensor per decoder slot. Queued media keeps an owned GPU
   copy completed on the import stream before publishing it, so decode-ahead
   fences do not block unrelated inference on the default stream. No CPU pixel
   readback is used.

Hardware textures may be macroblock-aligned, such as 1920x1088 for a 1920x1080
H.264 picture. The bridge validates the AVFrame's visible dimensions and uses
its crop rectangle for the Video Processor, rather than treating surface
padding as a resolution change. A real visible-resolution change is still
reported explicitly; padded rows must never enter the displayed image.

Unsupported codecs or hardware are reported to the caller. The Python media
source then falls back to AMF or FFmpeg software decoding.

## Build

Use the bundled FFmpeg package and the Visual Studio 2022 Build Tools:

```powershell
$cmake = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$ff = (Resolve-Path 'src\desktop2stereo\streaming\rtmp\ffmpeg').Path

& $cmake -S native\ffmpeg_d3d11_decoder `
  -B build\ffmpeg_d3d11_decoder `
  -G "Visual Studio 17 2022" `
  -A x64 `
  "-DD2S_FFMPEG_ROOT:PATH=$ff"

& $cmake --build build\ffmpeg_d3d11_decoder --config Release --parallel 4

Copy-Item `
  build\ffmpeg_d3d11_decoder\Release\d2s_ffmpeg_d3d11_decoder.dll `
  src\desktop2stereo\capture\native\windows\ `
  -Force
```

The DLL links the MSVC runtime statically. It still requires the bundled
FFmpeg DLLs (`avcodec-63.dll`, `avformat-63.dll`, and `avutil-61.dll`) at
runtime; `capture.media_d3d11` adds that directory with `os.add_dll_directory`.
