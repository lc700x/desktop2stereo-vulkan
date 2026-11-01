#pragma once

#ifdef _WIN32
#define D2S_AMD_API __declspec(dllexport)
#else
#define D2S_AMD_API
#endif

extern "C" {

// Returns 1 when the AMD AMF runtime and a D3D11 device can be created.
D2S_AMD_API int d2s_amd_encoder_probe();

// Reports the native ABI level used for Moonlight keyframe requests and
// resizable access-unit reads.
D2S_AMD_API int d2s_amd_encoder_abi_version();

// Writes a UTF-8 diagnostic into caller-provided storage.
D2S_AMD_API int d2s_amd_encoder_last_error(char* output, int capacity);

// Create an AMF encoder that accepts D3D11 RGBA textures. Returns an opaque
// handle, or nullptr when the AMF SDK/runtime is unavailable.
D2S_AMD_API void* d2s_amd_encoder_create(int width, int height, int fps, int bitrate, int hevc);
D2S_AMD_API void* d2s_amd_encoder_create_for_hip_device(
    int width, int height, int fps, int bitrate, int hevc, int hip_device_index);
D2S_AMD_API int d2s_amd_encoder_submit_texture(void* handle, void* d3d11_texture);

// Copy a HIP/ROCm RGBA8 device tensor into the encoder's shared D3D11
// texture, then submit it to AMF. stream is the native hipStream_t value.
D2S_AMD_API int d2s_amd_encoder_submit_hip_rgba(
    void* handle, const void* device_pointer, int pitch_bytes, void* stream, int force_idr);
D2S_AMD_API int d2s_amd_encoder_read_packet(void* handle, void* output, int capacity);
D2S_AMD_API void d2s_amd_encoder_destroy(void* handle);

}
