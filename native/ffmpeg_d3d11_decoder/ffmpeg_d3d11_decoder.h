#pragma once
#ifdef _WIN32
#include <stdint.h>
#ifdef D2S_FFMPEG_D3D11_DECODER_BUILD
#define D2S_FFMPEG_D3D11_API __declspec(dllexport)
#else
#define D2S_FFMPEG_D3D11_API __declspec(dllimport)
#endif
#ifdef __cplusplus
extern "C" {
#endif

typedef struct D2SFFmpegD3D11Frame {
    uint32_t struct_size;
    uint32_t slot;
    uint32_t width;
    uint32_t height;
    uint32_t dxgi_format;
    uint32_t adapter_luid_low;
    int32_t adapter_luid_high;
    uint64_t texture_shared_handle;
    uint64_t ready_fence_handle;
    uint64_t consumed_fence_handle;
    uint64_t ready_fence_value;
    int64_t pts_ns;
    int64_t duration_ns;
    int64_t frame_index;
    uint32_t color_primaries;
    uint32_t color_transfer;
    uint32_t color_matrix;
    uint32_t full_range;
} D2SFFmpegD3D11Frame;

D2S_FFMPEG_D3D11_API int d2s_ffmpeg_d3d11_probe(char* error, int error_capacity);
D2S_FFMPEG_D3D11_API void* d2s_ffmpeg_d3d11_create(
    const wchar_t* path,
    double start_seconds,
    int ring_size,
    char* error,
    int error_capacity);
D2S_FFMPEG_D3D11_API int d2s_ffmpeg_d3d11_next(
    void* decoder,
    double target_seconds,
    D2SFFmpegD3D11Frame* frame,
    int timeout_ms,
    char* error,
    int error_capacity);
D2S_FFMPEG_D3D11_API int d2s_ffmpeg_d3d11_seek(
    void* decoder,
    double seconds,
    char* error,
    int error_capacity);
D2S_FFMPEG_D3D11_API double d2s_ffmpeg_d3d11_position(void* decoder);
D2S_FFMPEG_D3D11_API void d2s_ffmpeg_d3d11_destroy(void* decoder);

#ifdef __cplusplus
}
#endif
#endif
