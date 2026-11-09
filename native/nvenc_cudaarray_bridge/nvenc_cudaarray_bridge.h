#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(D2S_NVENC_CUDAARRAY_BUILD)
#    define D2S_NVENC_API __declspec(dllexport)
#  else
#    define D2S_NVENC_API __declspec(dllimport)
#  endif
#else
#  define D2S_NVENC_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* d2s_nvenc_cudaarray_handle;

typedef struct d2s_nvenc_cudaarray_packet_t {
    size_t packet_size;
    int64_t pts;
    int64_t dts;
    int64_t duration;
} d2s_nvenc_cudaarray_packet_t;

D2S_NVENC_API uint32_t d2s_nvenc_cudaarray_abi_version(void);
D2S_NVENC_API int32_t d2s_nvenc_cudaarray_probe(void);

D2S_NVENC_API d2s_nvenc_cudaarray_handle d2s_nvenc_cudaarray_create(
    uint32_t width,
    uint32_t height,
    uint32_t fps,
    uint32_t bitrate,
    int32_t hevc,
    uint64_t cuda_array
);

/* Additional NV12 device-pointer entry points; existing ABI 3 array calls are unchanged. */
D2S_NVENC_API d2s_nvenc_cudaarray_handle d2s_nvenc_nv12_create(
    uint32_t width,
    uint32_t height,
    uint32_t fps,
    uint32_t bitrate,
    int32_t hevc,
    uint64_t device_pointer,
    uint32_t pitch
);

/* The caller must finish GPU writes before submission and keep the buffer registered until destroy. */
D2S_NVENC_API int32_t d2s_nvenc_nv12_submit(
    d2s_nvenc_cudaarray_handle handle,
    int64_t timestamp,
    int32_t force_idr
);

D2S_NVENC_API int32_t d2s_nvenc_cudaarray_submit(
    d2s_nvenc_cudaarray_handle handle,
    int64_t timestamp
);

/*
 * Launch a CUDA kernel on cuda_stream that reads the final SBS tensor and
 * writes RGBA pixels directly into the NVENC-registered CUDA array.
 * scalar_type: 0=uint8, 1=float32 normalized, 2=float16 normalized.
 */
D2S_NVENC_API int32_t d2s_nvenc_cudaarray_submit_tensor(
    d2s_nvenc_cudaarray_handle handle,
    uint64_t device_pointer,
    uint32_t channels,
    size_t stride_y,
    size_t stride_x,
    size_t stride_c,
    int32_t scalar_type,
    uint64_t cuda_stream,
    int64_t timestamp
);

D2S_NVENC_API int32_t d2s_nvenc_cudaarray_read_packet(
    d2s_nvenc_cudaarray_handle handle,
    uint8_t* destination,
    size_t capacity,
    size_t* packet_size
);

D2S_NVENC_API int32_t d2s_nvenc_cudaarray_read_packet_timed(
    d2s_nvenc_cudaarray_handle handle,
    uint8_t* destination,
    size_t capacity,
    d2s_nvenc_cudaarray_packet_t* packet
);

D2S_NVENC_API int32_t d2s_nvenc_cudaarray_flush(
    d2s_nvenc_cudaarray_handle handle
);

D2S_NVENC_API void d2s_nvenc_cudaarray_destroy(
    d2s_nvenc_cudaarray_handle handle
);

D2S_NVENC_API const char* d2s_nvenc_cudaarray_last_error(void);

#ifdef __cplusplus
}
#endif
