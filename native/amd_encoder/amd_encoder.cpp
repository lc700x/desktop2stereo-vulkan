#include "amd_encoder.h"

#include <d3d11.h>
#include <dxgi1_2.h>
#include <windows.h>

#include <cstdint>
#include <cstring>
#include <string>

#ifdef D2S_AMF_SDK
#include "public/common/AMFFactory.h"
#include "public/include/components/VideoEncoderVCE.h"
#include "public/include/components/VideoEncoderHEVC.h"
#endif

namespace {
thread_local std::string g_last_error;

void set_error(const char* message) {
    g_last_error = message ? message : "unknown AMD encoder error";
}

bool amf_runtime_present() {
    // AMF is loaded dynamically: the AMD display driver owns the runtime DLL.
    // The SDK headers are only required by the future surface encoder layer.
    HMODULE module = LoadLibraryW(L"amfrt64.dll");
    if (!module) {
        set_error("amfrt64.dll was not found; install an AMD Adrenalin driver");
        return false;
    }
    FreeLibrary(module);
    return true;
}

#ifdef D2S_AMF_SDK
// Keep the HIP ABI local so the bridge can be built without installing the
// ROCm SDK. These layouts match hip/hip_runtime_api.h on Windows.
struct HipWin32Handle { void* handle; void* name; };
union HipExternalHandle { int fd; HipWin32Handle win32; };
struct HipExternalMemoryHandleDesc {
    int type;
    HipExternalHandle handle;
    std::uint64_t size;
    unsigned int flags;
};
struct HipChannelFormatDesc { int x; int y; int z; int w; int f; };
struct HipExtent { std::size_t width; std::size_t height; std::size_t depth; };
struct HipExternalMipmappedArrayDesc {
    std::uint64_t offset;
    HipChannelFormatDesc format_desc;
    HipExtent extent;
    unsigned int flags;
    unsigned int num_levels;
};
using HipImportExternalMemory = int (*)(void**, const HipExternalMemoryHandleDesc*);
using HipDeviceGetLuid = int (*)(char*, unsigned int*, int);
using HipGetMappedMipmappedArray = int (*)(void**, void*, const HipExternalMipmappedArrayDesc*);
using HipGetMipmappedArrayLevel = int (*)(void**, void*, unsigned int);
using HipMemcpy2DToArrayAsync = int (*)(void*, std::size_t, std::size_t, const void*, std::size_t, std::size_t, std::size_t, int, void*);
using HipDestroyExternalMemory = int (*)(void*);
using HipStreamSynchronize = int (*)(void*);

HMODULE load_hip_runtime() {
    for (const auto* name : {L"amdhip64_7.dll", L"amdhip64_6.dll", L"amdhip64_5.dll", L"amdhip64.dll"}) {
        if (auto module = LoadLibraryW(name)) return module;
    }
    return nullptr;
}

struct AmdEncoder {
    amf::AMFContextPtr context;
    amf::AMFComponentPtr encoder;
    int width = 0;
    int height = 0;
    int fps = 0;
    bool hevc = false;
    int hip_device_index = -1;
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* device_context = nullptr;
    ID3D11Texture2D* shared_texture = nullptr;
    HANDLE shared_handle = nullptr;
    HMODULE hip_runtime = nullptr;
    HipDeviceGetLuid hip_device_get_luid = nullptr;
    void* hip_external_memory = nullptr;
    void* hip_array = nullptr;
    HipImportExternalMemory hip_import_external_memory = nullptr;
    HipGetMappedMipmappedArray hip_get_mapped_mipmapped_array = nullptr;
    HipGetMipmappedArrayLevel hip_get_mipmapped_array_level = nullptr;
    HipMemcpy2DToArrayAsync hip_memcpy_2d_to_array_async = nullptr;
    HipDestroyExternalMemory hip_destroy_external_memory = nullptr;
    HipStreamSynchronize hip_stream_synchronize = nullptr;
};

bool create_shared_rgba_texture(AmdEncoder* state) {
    D3D_FEATURE_LEVEL feature_level{};
    IDXGIAdapter1* selected_adapter = nullptr;
    if (state->hip_device_index >= 0) {
        if (!state->hip_runtime) state->hip_runtime = load_hip_runtime();
        if (!state->hip_runtime) {
            set_error("amdhip64.dll was not found; ROCm is required for adapter matching");
            return false;
        }
        state->hip_device_get_luid = reinterpret_cast<HipDeviceGetLuid>(
            GetProcAddress(state->hip_runtime, "hipDeviceGetLuid"));
        if (!state->hip_device_get_luid) {
            set_error("ROCm cannot report a device LUID; refusing an ambiguous adapter");
            return false;
        }
        char hip_luid[8]{};
        unsigned int node_mask = 0;
        if (state->hip_device_get_luid(hip_luid, &node_mask, state->hip_device_index) != 0) {
            set_error("ROCm failed to query the selected HIP device LUID");
            return false;
        }
        IDXGIFactory1* factory = nullptr;
        const HRESULT factory_result = CreateDXGIFactory1(
            __uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory));
        if (FAILED(factory_result) || !factory) {
            set_error("DXGI factory creation failed during HIP adapter matching");
            return false;
        }
        for (UINT index = 0; ; ++index) {
            IDXGIAdapter1* adapter = nullptr;
            const HRESULT enum_result = factory->EnumAdapters1(index, &adapter);
            if (enum_result == DXGI_ERROR_NOT_FOUND) break;
            if (FAILED(enum_result) || !adapter) continue;
            DXGI_ADAPTER_DESC1 desc{};
            if (SUCCEEDED(adapter->GetDesc1(&desc)) &&
                std::memcmp(&desc.AdapterLuid, hip_luid, sizeof(desc.AdapterLuid)) == 0) {
                selected_adapter = adapter;
                break;
            }
            adapter->Release();
        }
        factory->Release();
        if (!selected_adapter) {
            set_error("No DXGI adapter matches the selected HIP device LUID");
            return false;
        }
    }
    const auto hr = D3D11CreateDevice(
        selected_adapter,
        selected_adapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        nullptr,
        0,
        D3D11_SDK_VERSION,
        &state->device,
        &feature_level,
        &state->device_context);
    if (selected_adapter) selected_adapter->Release();
    if (FAILED(hr) || !state->device || !state->device_context) {
        set_error("D3D11 device creation failed");
        return false;
    }
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = static_cast<UINT>(state->width);
    desc.Height = static_cast<UINT>(state->height);
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE |
                     D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
    const HRESULT texture_result = state->device->CreateTexture2D(&desc, nullptr, &state->shared_texture);
    if (FAILED(texture_result)) {
        set_error((std::string("shared D3D11 RGBA texture creation failed, HRESULT=") +
                   std::to_string(static_cast<unsigned long>(texture_result))).c_str());
        return false;
    }
    IDXGIResource1* resource = nullptr;
    if (FAILED(state->shared_texture->QueryInterface(
            __uuidof(IDXGIResource1), reinterpret_cast<void**>(&resource))) || !resource) {
        set_error("D3D11 texture does not expose IDXGIResource1");
        return false;
    }
    const HRESULT handle_result = resource->CreateSharedHandle(
        nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
        nullptr, &state->shared_handle);
    resource->Release();
    if (FAILED(handle_result) || !state->shared_handle) {
        set_error("D3D11 shared handle creation failed");
        return false;
    }
    return true;
}

bool import_shared_texture_into_hip(AmdEncoder* state) {
    if (!state->hip_runtime) state->hip_runtime = load_hip_runtime();
    if (!state->hip_runtime) {
        set_error("amdhip64.dll was not found; ROCm is required for AMD GPU surface import");
        return false;
    }
    state->hip_import_external_memory = reinterpret_cast<HipImportExternalMemory>(
        GetProcAddress(state->hip_runtime, "hipImportExternalMemory"));
    state->hip_get_mapped_mipmapped_array = reinterpret_cast<HipGetMappedMipmappedArray>(
        GetProcAddress(state->hip_runtime, "hipExternalMemoryGetMappedMipmappedArray"));
    state->hip_get_mipmapped_array_level = reinterpret_cast<HipGetMipmappedArrayLevel>(
        GetProcAddress(state->hip_runtime, "hipGetMipmappedArrayLevel"));
    state->hip_memcpy_2d_to_array_async = reinterpret_cast<HipMemcpy2DToArrayAsync>(
        GetProcAddress(state->hip_runtime, "hipMemcpy2DToArrayAsync"));
    state->hip_destroy_external_memory = reinterpret_cast<HipDestroyExternalMemory>(
        GetProcAddress(state->hip_runtime, "hipDestroyExternalMemory"));
    state->hip_stream_synchronize = reinterpret_cast<HipStreamSynchronize>(
        GetProcAddress(state->hip_runtime, "hipStreamSynchronize"));
    if (!state->hip_import_external_memory || !state->hip_get_mapped_mipmapped_array ||
        !state->hip_get_mipmapped_array_level || !state->hip_memcpy_2d_to_array_async ||
        !state->hip_destroy_external_memory || !state->hip_stream_synchronize) {
        set_error("ROCm HIP external-memory functions are unavailable");
        return false;
    }
    HipExternalMemoryHandleDesc handle_desc{};
    handle_desc.type = 2; // hipExternalMemoryHandleTypeOpaqueWin32
    handle_desc.handle.win32.handle = state->shared_handle;
    handle_desc.size = static_cast<std::uint64_t>(state->width) *
                       static_cast<std::uint64_t>(state->height) * 4u;
    void* external_memory = nullptr;
    if (state->hip_import_external_memory(&external_memory, &handle_desc) != 0 || !external_memory) {
        set_error("HIP could not import the shared D3D11 texture");
        return false;
    }
    HipExternalMipmappedArrayDesc mapped_desc{};
    mapped_desc.format_desc = {8, 8, 8, 8, 0};
    mapped_desc.extent = {static_cast<std::size_t>(state->width), static_cast<std::size_t>(state->height), 0};
    mapped_desc.num_levels = 1;
    void* mipmapped = nullptr;
    if (state->hip_get_mapped_mipmapped_array(&mipmapped, external_memory, &mapped_desc) != 0 || !mipmapped) {
        state->hip_destroy_external_memory(external_memory);
        set_error("HIP could not map the shared D3D11 texture");
        return false;
    }
    void* array = nullptr;
    if (state->hip_get_mipmapped_array_level(&array, mipmapped, 0) != 0 || !array) {
        state->hip_destroy_external_memory(external_memory);
        set_error("HIP could not access the mapped D3D11 texture array");
        return false;
    }
    state->hip_external_memory = external_memory;
    state->hip_array = array;
    return true;
}

void release_native_resources(AmdEncoder* state) {
    if (!state) return;
    if (state->hip_external_memory && state->hip_destroy_external_memory) {
        state->hip_destroy_external_memory(state->hip_external_memory);
    }
    if (state->hip_runtime) FreeLibrary(state->hip_runtime);
    if (state->shared_handle) CloseHandle(state->shared_handle);
    if (state->shared_texture) state->shared_texture->Release();
    if (state->device_context) state->device_context->Release();
    if (state->device) state->device->Release();
    state->hip_external_memory = nullptr;
    state->hip_runtime = nullptr;
    state->shared_handle = nullptr;
    state->shared_texture = nullptr;
    state->device_context = nullptr;
    state->device = nullptr;
}
#endif
}  // namespace

extern "C" D2S_AMD_API int d2s_amd_encoder_probe() {
    if (!amf_runtime_present()) {
        return 0;
    }

    IDXGIFactory* factory = nullptr;
    HRESULT hr = CreateDXGIFactory(__uuidof(IDXGIFactory), reinterpret_cast<void**>(&factory));
    if (FAILED(hr) || !factory) {
        set_error("DXGI factory creation failed");
        return 0;
    }

    IDXGIAdapter* adapter = nullptr;
    bool found_amd = false;
    for (UINT index = 0; factory->EnumAdapters(index, &adapter) != DXGI_ERROR_NOT_FOUND; ++index) {
        DXGI_ADAPTER_DESC desc{};
        if (SUCCEEDED(adapter->GetDesc(&desc))) {
            const std::wstring name(desc.Description);
            if (name.find(L"AMD") != std::wstring::npos ||
                name.find(L"Radeon") != std::wstring::npos) {
                found_amd = true;
                adapter->Release();
                break;
            }
        }
        adapter->Release();
        adapter = nullptr;
    }
    factory->Release();
    if (!found_amd) {
        set_error("no AMD/Radeon DXGI adapter was found");
        return 0;
    }
    g_last_error.clear();
    return 1;
}

extern "C" D2S_AMD_API int d2s_amd_encoder_abi_version() {
    return 2;
}

extern "C" D2S_AMD_API int d2s_amd_encoder_last_error(char* output, int capacity) {
    if (!output || capacity <= 0) {
        return static_cast<int>(g_last_error.size());
    }
    const int count = static_cast<int>(g_last_error.size());
    const int copied = (count < capacity - 1) ? count : capacity - 1;
    std::memcpy(output, g_last_error.data(), copied);
    output[copied] = '\0';
    return count;
}

static void* d2s_amd_encoder_create_impl(
    int width, int height, int fps, int bitrate, int hevc, int hip_device_index) {
#ifndef D2S_AMF_SDK
    (void)width; (void)height; (void)fps; (void)bitrate; (void)hevc; (void)hip_device_index;
    set_error("AMD bridge was built without the AMF SDK headers");
    return nullptr;
#else
    if (!amf_runtime_present()) return nullptr;
    auto* state = new AmdEncoder();
    state->width = width; state->height = height; state->fps = fps; state->hevc = hevc != 0;
    state->hip_device_index = hip_device_index;
    AMF_RESULT result = g_AMFFactory.Init();
    if (result != AMF_OK) {
        set_error("AMF factory initialization failed");
        release_native_resources(state);
        delete state;
        return nullptr;
    }
    result = g_AMFFactory.GetFactory()->CreateContext(&state->context);
    if (result != AMF_OK || !state->context) {
        set_error("AMF context creation failed");
        release_native_resources(state);
        g_AMFFactory.Terminate();
        delete state;
        return nullptr;
    }
    if (!create_shared_rgba_texture(state) || !import_shared_texture_into_hip(state)) {
        release_native_resources(state);
        delete state;
        return nullptr;
    }
    result = state->context->InitDX11(state->device);
    if (result != AMF_OK) {
        set_error("AMF DX11 context initialization failed");
        state->context->Terminate();
        release_native_resources(state);
        g_AMFFactory.Terminate();
        delete state;
        return nullptr;
    }
    const wchar_t* component = state->hevc ? AMFVideoEncoder_HEVC : AMFVideoEncoderVCE_AVC;
    result = g_AMFFactory.GetFactory()->CreateComponent(state->context, component, &state->encoder);
    if (result != AMF_OK || !state->encoder) {
        set_error("AMF video encoder component creation failed");
        state->context->Terminate();
        release_native_resources(state);
        g_AMFFactory.Terminate();
        delete state;
        return nullptr;
    }
    const amf_int64 rate = static_cast<amf_int64>(bitrate);
    state->encoder->SetProperty(AMF_VIDEO_ENCODER_FRAMESIZE, AMFConstructSize(width, height));
    state->encoder->SetProperty(AMF_VIDEO_ENCODER_FRAMERATE, AMFConstructRate(fps, 1));
    state->encoder->SetProperty(AMF_VIDEO_ENCODER_TARGET_BITRATE, rate);
    if (hevc) {
        state->encoder->SetProperty(AMF_VIDEO_ENCODER_HEVC_GOP_SIZE, static_cast<amf_int64>(fps));
        state->encoder->SetProperty(AMF_VIDEO_ENCODER_HEVC_NUM_GOPS_PER_IDR, static_cast<amf_int64>(1));
    } else {
        state->encoder->SetProperty(AMF_VIDEO_ENCODER_IDR_PERIOD, static_cast<amf_int64>(fps));
    }
    result = state->encoder->Init(amf::AMF_SURFACE_RGBA, width, height);
    if (result != AMF_OK) {
        set_error("AMF encoder initialization failed");
        state->encoder->Terminate();
        state->context->Terminate();
        release_native_resources(state);
        g_AMFFactory.Terminate();
        delete state;
        return nullptr;
    }
    g_last_error.clear();
    return state;
#endif
}

extern "C" D2S_AMD_API void* d2s_amd_encoder_create(
    int width, int height, int fps, int bitrate, int hevc) {
    return d2s_amd_encoder_create_impl(width, height, fps, bitrate, hevc, -1);
}

extern "C" D2S_AMD_API void* d2s_amd_encoder_create_for_hip_device(
    int width, int height, int fps, int bitrate, int hevc, int hip_device_index) {
    return d2s_amd_encoder_create_impl(width, height, fps, bitrate, hevc, hip_device_index);
}

extern "C" D2S_AMD_API int d2s_amd_encoder_submit_texture(void* handle, void* d3d11_texture) {
#ifndef D2S_AMF_SDK
    (void)handle; (void)d3d11_texture; return -1;
#else
    auto* state = static_cast<AmdEncoder*>(handle);
    if (!state || !state->context || !state->encoder || !d3d11_texture) return -1;
    amf::AMFSurfacePtr surface;
    AMF_RESULT result = state->context->CreateSurfaceFromDX11Native(d3d11_texture, &surface, nullptr);
    if (result != AMF_OK || !surface) { set_error("AMF could not wrap the D3D11 texture"); return -1; }
    result = state->encoder->SubmitInput(surface);
    return result == AMF_OK || result == AMF_NEED_MORE_INPUT ? 1 : 0;
#endif
}

extern "C" D2S_AMD_API int d2s_amd_encoder_submit_hip_rgba(
    void* handle, const void* device_pointer, int pitch_bytes, void* stream, int force_idr) {
#ifndef D2S_AMF_SDK
    (void)handle; (void)device_pointer; (void)pitch_bytes; (void)stream; (void)force_idr; return -1;
#else
    auto* state = static_cast<AmdEncoder*>(handle);
    if (!state || !device_pointer || pitch_bytes < state->width * 4 || !state->hip_array) {
        set_error("invalid HIP RGBA surface arguments");
        return -1;
    }
    if (state->hip_memcpy_2d_to_array_async(
            state->hip_array, 0, 0, device_pointer,
            static_cast<std::size_t>(pitch_bytes),
            static_cast<std::size_t>(state->width) * 4u,
            static_cast<std::size_t>(state->height), 3, stream) != 0) {
        set_error("HIP RGBA to D3D11 surface copy failed");
        return -1;
    }
    if (state->hip_stream_synchronize(stream) != 0) {
        set_error("HIP stream synchronization failed before AMF submit");
        return -1;
    }
    if (force_idr) {
        const wchar_t* property = state->hevc
            ? AMF_VIDEO_ENCODER_HEVC_FORCE_PICTURE_TYPE
            : AMF_VIDEO_ENCODER_FORCE_PICTURE_TYPE;
        const amf_int64 picture_type = state->hevc
            ? AMF_VIDEO_ENCODER_HEVC_PICTURE_TYPE_IDR
            : AMF_VIDEO_ENCODER_PICTURE_TYPE_IDR;
        if (state->encoder->SetProperty(property, picture_type) != AMF_OK) {
            set_error("AMF rejected the forced IDR request");
            return -1;
        }
    }
    const int submitted = d2s_amd_encoder_submit_texture(state, state->shared_texture);
    if (force_idr) {
        const wchar_t* property = state->hevc
            ? AMF_VIDEO_ENCODER_HEVC_FORCE_PICTURE_TYPE
            : AMF_VIDEO_ENCODER_FORCE_PICTURE_TYPE;
        state->encoder->SetProperty(property, static_cast<amf_int64>(0));
    }
    return submitted;
#endif
}

extern "C" D2S_AMD_API int d2s_amd_encoder_read_packet(void* handle, void* output, int capacity) {
#ifndef D2S_AMF_SDK
    (void)handle; (void)output; (void)capacity; return -1;
#else
    auto* state = static_cast<AmdEncoder*>(handle);
    if (!state || !output || capacity <= 0) return -1;
    amf::AMFDataPtr data;
    AMF_RESULT result = state->encoder->QueryOutput(&data);
    if (result != AMF_OK || !data) return 0;
    amf::AMFBufferPtr buffer(data);
    const auto size = static_cast<int>(buffer->GetSize());
    if (size > capacity) return -size;
    std::memcpy(output, buffer->GetNative(), static_cast<size_t>(size));
    return size;
#endif
}

extern "C" D2S_AMD_API void d2s_amd_encoder_destroy(void* handle) {
#ifdef D2S_AMF_SDK
    auto* state = static_cast<AmdEncoder*>(handle);
    if (!state) return;
    if (state->encoder) state->encoder->Terminate();
    if (state->context) state->context->Terminate();
    release_native_resources(state);
    delete state;
    g_AMFFactory.Terminate();
#else
    (void)handle;
#endif
}
