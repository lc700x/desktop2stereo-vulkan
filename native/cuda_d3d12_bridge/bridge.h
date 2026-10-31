#pragma once

#include <stdint.h>

#if defined(_WIN32)
#if defined(D2S_CUDA_D3D12_BRIDGE_BUILD)
#define D2S_CUDA_D3D12_API __declspec(dllexport)
#else
#define D2S_CUDA_D3D12_API __declspec(dllimport)
#endif
#else
#define D2S_CUDA_D3D12_API
#endif

extern "C" {
D2S_CUDA_D3D12_API int d2s_d3d12_bridge_create(
    uint64_t source_luid,
    uint64_t target_luid,
    uint32_t width,
    uint32_t height,
    uint32_t ring_size,
    void** output,
    char* error,
    uint32_t error_capacity);
D2S_CUDA_D3D12_API void d2s_d3d12_bridge_destroy(void* bridge);
D2S_CUDA_D3D12_API uint64_t d2s_d3d12_bridge_row_pitch(void* bridge);
D2S_CUDA_D3D12_API uint64_t d2s_d3d12_bridge_buffer_size(void* bridge);
D2S_CUDA_D3D12_API int64_t d2s_d3d12_bridge_cuda_buffer_handle(
    void* bridge, uint32_t slot, uint32_t eye, char* error, uint32_t error_capacity);
D2S_CUDA_D3D12_API int64_t d2s_d3d12_bridge_cuda_fence_handle(
    void* bridge, char* error, uint32_t error_capacity);
D2S_CUDA_D3D12_API int64_t d2s_d3d12_bridge_cuda_consumed_fence_handle(
    void* bridge, char* error, uint32_t error_capacity);
D2S_CUDA_D3D12_API int64_t d2s_d3d12_bridge_vulkan_ready_fence_handle(
    void* bridge, char* error, uint32_t error_capacity);
D2S_CUDA_D3D12_API int64_t d2s_d3d12_bridge_vulkan_release_fence_handle(
    void* bridge, uint32_t slot, uint32_t eye, char* error, uint32_t error_capacity);
D2S_CUDA_D3D12_API int64_t d2s_d3d12_bridge_vulkan_image_handle(
    void* bridge, uint32_t slot, uint32_t eye, char* error, uint32_t error_capacity);
D2S_CUDA_D3D12_API int d2s_d3d12_bridge_check_slot_ready(
    void* bridge, uint32_t slot, char* error, uint32_t error_capacity);
D2S_CUDA_D3D12_API int d2s_d3d12_bridge_copy_to_vulkan(
    void* bridge,
    uint32_t slot,
    uint64_t source_fence_value,
    uint64_t left_release_value,
    uint64_t right_release_value,
    uint64_t* completion_value,
    char* error,
    uint32_t error_capacity);
D2S_CUDA_D3D12_API int d2s_d3d12_bridge_wait(
    void* bridge, uint64_t completion_value, uint32_t timeout_ms,
    char* error, uint32_t error_capacity);
D2S_CUDA_D3D12_API int d2s_d3d12_bridge_wait_slot(
    void* bridge, uint32_t slot, uint32_t timeout_ms,
    char* error, uint32_t error_capacity);
}
