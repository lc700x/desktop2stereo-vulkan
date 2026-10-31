#include "bridge.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {
struct EyeSlot {
    ComPtr<ID3D12Heap> source_heap;
    ComPtr<ID3D12Heap> target_heap;
    ComPtr<ID3D12Resource> source_buffer;
    ComPtr<ID3D12Resource> target_buffer;
    ComPtr<ID3D12Resource> target_image;
    ComPtr<ID3D12Fence> release_fence;
    HANDLE source_resource_handle = nullptr;
};

struct Bridge {
    ComPtr<IDXGIFactory1> factory;
    ComPtr<IDXGIAdapter1> source_adapter;
    ComPtr<IDXGIAdapter1> target_adapter;
    ComPtr<ID3D12Device> source_device;
    ComPtr<ID3D12Device> target_device;
    ComPtr<ID3D12CommandQueue> target_queue;
    ComPtr<ID3D12Fence> source_fence;
    ComPtr<ID3D12Fence> source_consumed_fence;
    ComPtr<ID3D12Fence> target_source_fence;
    ComPtr<ID3D12Fence> target_source_consumed_fence;
    ComPtr<ID3D12Fence> target_fence;
    HANDLE source_fence_handle = nullptr;
    HANDLE source_consumed_fence_handle = nullptr;
    HANDLE target_event = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t ring_size = 0;
    uint32_t row_pitch = 0;
    uint64_t buffer_size = 0;
    uint64_t next_completion = 0;
    std::vector<EyeSlot> eyes;
    std::vector<uint64_t> slot_completions;
    std::vector<ComPtr<ID3D12CommandAllocator>> allocators;
    std::vector<ComPtr<ID3D12GraphicsCommandList>> command_lists;
};

void set_error(char* output, uint32_t capacity, const char* message) {
    if (output && capacity) {
        std::snprintf(output, static_cast<size_t>(capacity), "%s", message ? message : "unknown error");
    }
}

std::string hresult_text(const char* operation, HRESULT result) {
    char text[256]{};
    std::snprintf(text, sizeof(text), "%s failed with HRESULT 0x%08lx",
                  operation, static_cast<unsigned long>(result));
    return text;
}

uint64_t packed_luid(const LUID& luid) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(luid.HighPart)) << 32) |
           static_cast<uint32_t>(luid.LowPart);
}

HRESULT find_adapter(IDXGIFactory1* factory, uint64_t wanted, IDXGIAdapter1** result) {
    if (!factory || !result || !wanted) return E_INVALIDARG;
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        HRESULT hr = factory->EnumAdapters1(index, &adapter);
        if (hr == DXGI_ERROR_NOT_FOUND) return DXGI_ERROR_NOT_FOUND;
        if (FAILED(hr)) return hr;
        DXGI_ADAPTER_DESC1 description{};
        hr = adapter->GetDesc1(&description);
        if (FAILED(hr)) return hr;
        if (packed_luid(description.AdapterLuid) == wanted) {
            *result = adapter.Detach();
            return S_OK;
        }
    }
}

HRESULT create_shared_resource_handle(ID3D12Device* device, ID3D12Resource* resource, HANDLE* handle) {
    if (!device || !resource || !handle) return E_INVALIDARG;
    return device->CreateSharedHandle(resource, nullptr, GENERIC_ALL, nullptr, handle);
}

HRESULT duplicate_handle(HANDLE source, HANDLE* output) {
    if (!source || !output) return E_INVALIDARG;
    HANDLE duplicate = nullptr;
    if (!DuplicateHandle(GetCurrentProcess(), source, GetCurrentProcess(), &duplicate,
                         0, FALSE, DUPLICATE_SAME_ACCESS)) {
        return HRESULT_FROM_WIN32(GetLastError());
    }
    *output = duplicate;
    return S_OK;
}

HRESULT create_source_buffer(Bridge* bridge, EyeSlot& slot,
                             const char** failed_operation) {
    D3D12_HEAP_DESC heap{};
    heap.SizeInBytes = bridge->buffer_size;
    heap.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.Properties.CreationNodeMask = 1;
    heap.Properties.VisibleNodeMask = 1;
    heap.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    heap.Flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER;
    D3D12_RESOURCE_DESC description{};
    description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    description.Width = bridge->buffer_size;
    description.Height = 1;
    description.DepthOrArraySize = 1;
    description.MipLevels = 1;
    description.SampleDesc.Count = 1;
    description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    description.Flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
    HRESULT hr = bridge->source_device->CreateHeap(&heap, IID_PPV_ARGS(&slot.source_heap));
    if (FAILED(hr) && failed_operation) *failed_operation = "CreateHeap(shared cross-adapter source)";
    if (FAILED(hr)) return hr;
    hr = bridge->source_device->CreatePlacedResource(
        slot.source_heap.Get(), 0, &description, D3D12_RESOURCE_STATE_COMMON,
        nullptr, IID_PPV_ARGS(&slot.source_buffer));
    if (FAILED(hr) && failed_operation) *failed_operation = "CreatePlacedResource(source buffer)";
    if (FAILED(hr)) return hr;

    HANDLE shared_heap_handle = nullptr;
    hr = bridge->source_device->CreateSharedHandle(slot.source_heap.Get(), nullptr,
                                                   GENERIC_ALL, nullptr,
                                                   &shared_heap_handle);
    if (FAILED(hr) && failed_operation) *failed_operation = "CreateSharedHandle(source heap)";
    if (FAILED(hr)) return hr;
    hr = bridge->target_device->OpenSharedHandle(shared_heap_handle,
                                                  IID_PPV_ARGS(&slot.target_heap));
    CloseHandle(shared_heap_handle);
    if (FAILED(hr) && failed_operation) *failed_operation = "OpenSharedHandle(source heap on target)";
    if (FAILED(hr)) return hr;
    hr = bridge->target_device->CreatePlacedResource(
        slot.target_heap.Get(), 0, &description, D3D12_RESOURCE_STATE_COMMON,
        nullptr, IID_PPV_ARGS(&slot.target_buffer));
    if (FAILED(hr) && failed_operation) *failed_operation = "CreatePlacedResource(target buffer)";
    return hr;
}

HRESULT create_target_image(Bridge* bridge, EyeSlot& slot) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = 1;
    heap.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC description{};
    description.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    description.Width = bridge->width;
    description.Height = bridge->height;
    description.DepthOrArraySize = 1;
    description.MipLevels = 1;
    description.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    description.SampleDesc.Count = 1;
    description.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    return bridge->target_device->CreateCommittedResource(
        &heap,
        D3D12_HEAP_FLAG_SHARED,
        &description,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&slot.target_image));
}

HRESULT initialize_bridge(Bridge* bridge, uint64_t source_luid, uint64_t target_luid,
                          uint32_t width, uint32_t height, uint32_t ring_size,
                          const char** failed_operation) {
    if (!bridge || !source_luid || !target_luid || source_luid == target_luid ||
        !width || !height || ring_size < 2 || ring_size > 8) return E_INVALIDARG;
    bridge->width = width;
    bridge->height = height;
    bridge->ring_size = ring_size;
    const uint64_t row_bytes = static_cast<uint64_t>(width) * 4;
    bridge->row_pitch = static_cast<uint32_t>((row_bytes + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) &
                                               ~(static_cast<uint64_t>(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT) - 1));
    const uint64_t raw_size = static_cast<uint64_t>(bridge->row_pitch) * height;
    bridge->buffer_size = (raw_size + D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT - 1) &
                          ~(static_cast<uint64_t>(D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT) - 1);

    const char* operation = "CreateDXGIFactory1";
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&bridge->factory));
    if (FAILED(hr) && failed_operation) *failed_operation = operation;
    if (FAILED(hr)) return hr;
    operation = "find CUDA adapter by LUID";
    hr = find_adapter(bridge->factory.Get(), source_luid, &bridge->source_adapter);
    if (FAILED(hr) && failed_operation) *failed_operation = operation;
    if (FAILED(hr)) return hr;
    operation = "find Vulkan adapter by LUID";
    hr = find_adapter(bridge->factory.Get(), target_luid, &bridge->target_adapter);
    if (FAILED(hr) && failed_operation) *failed_operation = operation;
    if (FAILED(hr)) return hr;
    operation = "D3D12CreateDevice(CUDA adapter)";
    hr = D3D12CreateDevice(bridge->source_adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                           IID_PPV_ARGS(&bridge->source_device));
    if (FAILED(hr) && failed_operation) *failed_operation = operation;
    if (FAILED(hr)) return hr;
    operation = "D3D12CreateDevice(Vulkan adapter)";
    hr = D3D12CreateDevice(bridge->target_adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                           IID_PPV_ARGS(&bridge->target_device));
    if (FAILED(hr) && failed_operation) *failed_operation = operation;
    if (FAILED(hr)) return hr;

    D3D12_COMMAND_QUEUE_DESC queue_description{};
    queue_description.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    operation = "create target command queue";
    hr = bridge->target_device->CreateCommandQueue(&queue_description,
                                                   IID_PPV_ARGS(&bridge->target_queue));
    if (FAILED(hr) && failed_operation) *failed_operation = operation;
    if (FAILED(hr)) return hr;
    operation = "create shared cross-adapter source fence";
    hr = bridge->source_device->CreateFence(
        0, static_cast<D3D12_FENCE_FLAGS>(D3D12_FENCE_FLAG_SHARED |
                                          D3D12_FENCE_FLAG_SHARED_CROSS_ADAPTER),
        IID_PPV_ARGS(&bridge->source_fence));
    if (FAILED(hr) && failed_operation) *failed_operation = operation;
    if (FAILED(hr)) return hr;
    operation = "share cross-adapter source fence";
    hr = bridge->source_device->CreateSharedHandle(bridge->source_fence.Get(), nullptr,
                                                   GENERIC_ALL, nullptr,
                                                   &bridge->source_fence_handle);
    if (FAILED(hr) && failed_operation) *failed_operation = operation;
    if (FAILED(hr)) return hr;
    operation = "open shared source fence on target adapter";
    hr = bridge->target_device->OpenSharedHandle(bridge->source_fence_handle,
                                                  IID_PPV_ARGS(&bridge->target_source_fence));
    if (FAILED(hr) && failed_operation) *failed_operation = operation;
    if (FAILED(hr)) return hr;
    operation = "create shared cross-adapter source-consumed fence";
    hr = bridge->source_device->CreateFence(
        0, static_cast<D3D12_FENCE_FLAGS>(D3D12_FENCE_FLAG_SHARED |
                                          D3D12_FENCE_FLAG_SHARED_CROSS_ADAPTER),
        IID_PPV_ARGS(&bridge->source_consumed_fence));
    if (FAILED(hr) && failed_operation) *failed_operation = operation;
    if (FAILED(hr)) return hr;
    operation = "share cross-adapter source-consumed fence";
    hr = bridge->source_device->CreateSharedHandle(
        bridge->source_consumed_fence.Get(), nullptr, GENERIC_ALL, nullptr,
        &bridge->source_consumed_fence_handle);
    if (FAILED(hr) && failed_operation) *failed_operation = operation;
    if (FAILED(hr)) return hr;
    operation = "open shared source-consumed fence on target adapter";
    hr = bridge->target_device->OpenSharedHandle(
        bridge->source_consumed_fence_handle,
        IID_PPV_ARGS(&bridge->target_source_consumed_fence));
    if (FAILED(hr) && failed_operation) *failed_operation = operation;
    if (FAILED(hr)) return hr;
    operation = "create target completion fence";
    hr = bridge->target_device->CreateFence(0, D3D12_FENCE_FLAG_SHARED,
                                            IID_PPV_ARGS(&bridge->target_fence));
    if (FAILED(hr) && failed_operation) *failed_operation = operation;
    if (FAILED(hr)) return hr;
    bridge->target_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!bridge->target_event && failed_operation) *failed_operation = "CreateEventW";
    if (!bridge->target_event) return HRESULT_FROM_WIN32(GetLastError());

    bridge->eyes.resize(static_cast<size_t>(ring_size) * 2);
    bridge->slot_completions.resize(ring_size, 0);
    bridge->allocators.resize(ring_size);
    bridge->command_lists.resize(ring_size);
    for (uint32_t slot_index = 0; slot_index < ring_size; ++slot_index) {
        operation = "create target command allocator";
        hr = bridge->target_device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&bridge->allocators[slot_index]));
        if (FAILED(hr) && failed_operation) *failed_operation = operation;
        if (FAILED(hr)) return hr;
        operation = "create target command list";
        hr = bridge->target_device->CreateCommandList(
            0, D3D12_COMMAND_LIST_TYPE_DIRECT,
            bridge->allocators[slot_index].Get(), nullptr,
            IID_PPV_ARGS(&bridge->command_lists[slot_index]));
        if (FAILED(hr) && failed_operation) *failed_operation = operation;
        if (FAILED(hr)) return hr;
        operation = "close target command list";
        hr = bridge->command_lists[slot_index]->Close();
        if (FAILED(hr) && failed_operation) *failed_operation = operation;
        if (FAILED(hr)) return hr;
    }
    for (EyeSlot& eye : bridge->eyes) {
        hr = create_source_buffer(bridge, eye, failed_operation);
        if (FAILED(hr)) return hr;
        hr = create_target_image(bridge, eye);
        if (FAILED(hr) && failed_operation) *failed_operation = "create Vulkan-shared target texture";
        if (FAILED(hr)) return hr;
        hr = bridge->target_device->CreateFence(
            0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&eye.release_fence));
        if (FAILED(hr) && failed_operation) *failed_operation = "create shared Vulkan-release fence";
        if (FAILED(hr)) return hr;
    }
    return S_OK;
}

EyeSlot* get_eye(Bridge* bridge, uint32_t slot, uint32_t eye) {
    if (!bridge || slot >= bridge->ring_size || eye > 1) return nullptr;
    return &bridge->eyes[static_cast<size_t>(slot) * 2 + eye];
}
}

extern "C" {
int d2s_d3d12_bridge_create(uint64_t source_luid, uint64_t target_luid,
                            uint32_t width, uint32_t height, uint32_t ring_size,
                            void** output, char* error, uint32_t error_capacity) {
    if (!output) {
        set_error(error, error_capacity, "output bridge pointer is null");
        return static_cast<int>(E_INVALIDARG);
    }
    *output = nullptr;
    auto bridge = std::make_unique<Bridge>();
    const char* failed_operation = "validate D3D12 bridge arguments";
    const HRESULT hr = initialize_bridge(bridge.get(), source_luid, target_luid,
                                         width, height, ring_size,
                                         &failed_operation);
    if (FAILED(hr)) {
        const std::string message = hresult_text(failed_operation, hr);
        set_error(error, error_capacity, message.c_str());
        if (bridge->target_event) CloseHandle(bridge->target_event);
        if (bridge->source_fence_handle) CloseHandle(bridge->source_fence_handle);
        if (bridge->source_consumed_fence_handle) CloseHandle(bridge->source_consumed_fence_handle);
        for (EyeSlot& eye : bridge->eyes) {
            if (eye.source_resource_handle) CloseHandle(eye.source_resource_handle);
        }
        return static_cast<int>(hr);
    }
    *output = bridge.release();
    return 0;
}

void d2s_d3d12_bridge_destroy(void* pointer) {
    auto* bridge = static_cast<Bridge*>(pointer);
    if (!bridge) return;
    if (bridge->target_event) CloseHandle(bridge->target_event);
    if (bridge->source_fence_handle) CloseHandle(bridge->source_fence_handle);
    if (bridge->source_consumed_fence_handle) CloseHandle(bridge->source_consumed_fence_handle);
    for (EyeSlot& eye : bridge->eyes) {
        if (eye.source_resource_handle) CloseHandle(eye.source_resource_handle);
    }
    delete bridge;
}

uint64_t d2s_d3d12_bridge_row_pitch(void* pointer) {
    const auto* bridge = static_cast<Bridge*>(pointer);
    return bridge ? bridge->row_pitch : 0;
}

uint64_t d2s_d3d12_bridge_buffer_size(void* pointer) {
    const auto* bridge = static_cast<Bridge*>(pointer);
    return bridge ? bridge->buffer_size : 0;
}

int64_t d2s_d3d12_bridge_cuda_buffer_handle(void* pointer, uint32_t slot_index,
                                             uint32_t eye_index, char* error,
                                             uint32_t error_capacity) {
    auto* bridge = static_cast<Bridge*>(pointer);
    EyeSlot* eye = get_eye(bridge, slot_index, eye_index);
    if (!eye) {
        set_error(error, error_capacity, "invalid D3D12 source buffer slot");
        return 0;
    }
    HANDLE handle = nullptr;
    HRESULT hr = bridge->source_device->CreateSharedHandle(
        eye->source_heap.Get(), nullptr, GENERIC_ALL, nullptr, &handle);
    if (FAILED(hr)) {
        const std::string message = hresult_text("CreateSharedHandle(source heap)", hr);
        set_error(error, error_capacity, message.c_str());
        return 0;
    }
    return static_cast<int64_t>(reinterpret_cast<intptr_t>(handle));
}

int64_t d2s_d3d12_bridge_cuda_fence_handle(void* pointer, char* error,
                                             uint32_t error_capacity) {
    auto* bridge = static_cast<Bridge*>(pointer);
    if (!bridge || !bridge->source_fence_handle) {
        set_error(error, error_capacity, "D3D12 source fence is unavailable");
        return 0;
    }
    HANDLE duplicate = nullptr;
    HRESULT hr = duplicate_handle(bridge->source_fence_handle, &duplicate);
    if (FAILED(hr)) {
        const std::string message = hresult_text("DuplicateHandle(source fence)", hr);
        set_error(error, error_capacity, message.c_str());
        return 0;
    }
    return static_cast<int64_t>(reinterpret_cast<intptr_t>(duplicate));
}

int64_t d2s_d3d12_bridge_cuda_consumed_fence_handle(void* pointer, char* error,
                                                     uint32_t error_capacity) {
    auto* bridge = static_cast<Bridge*>(pointer);
    if (!bridge || !bridge->source_consumed_fence_handle) {
        set_error(error, error_capacity, "D3D12 source-consumed fence is unavailable");
        return 0;
    }
    HANDLE duplicate = nullptr;
    HRESULT hr = duplicate_handle(bridge->source_consumed_fence_handle, &duplicate);
    if (FAILED(hr)) {
        const std::string message = hresult_text("DuplicateHandle(source-consumed fence)", hr);
        set_error(error, error_capacity, message.c_str());
        return 0;
    }
    return static_cast<int64_t>(reinterpret_cast<intptr_t>(duplicate));
}

int64_t d2s_d3d12_bridge_vulkan_ready_fence_handle(void* pointer, char* error,
                                                   uint32_t error_capacity) {
    auto* bridge = static_cast<Bridge*>(pointer);
    if (!bridge || !bridge->target_fence) {
        set_error(error, error_capacity, "D3D12 Vulkan-ready fence is unavailable");
        return 0;
    }
    HANDLE handle = nullptr;
    HRESULT hr = bridge->target_device->CreateSharedHandle(
        bridge->target_fence.Get(), nullptr, GENERIC_ALL, nullptr, &handle);
    if (FAILED(hr)) {
        const std::string message = hresult_text("CreateSharedHandle(Vulkan-ready fence)", hr);
        set_error(error, error_capacity, message.c_str());
        return 0;
    }
    return static_cast<int64_t>(reinterpret_cast<intptr_t>(handle));
}

int64_t d2s_d3d12_bridge_vulkan_release_fence_handle(
    void* pointer, uint32_t slot_index, uint32_t eye_index, char* error,
    uint32_t error_capacity) {
    auto* bridge = static_cast<Bridge*>(pointer);
    EyeSlot* eye = get_eye(bridge, slot_index, eye_index);
    if (!eye || !eye->release_fence) {
        set_error(error, error_capacity, "invalid D3D12 Vulkan-release fence slot");
        return 0;
    }
    HANDLE handle = nullptr;
    HRESULT hr = bridge->target_device->CreateSharedHandle(
        eye->release_fence.Get(), nullptr, GENERIC_ALL, nullptr, &handle);
    if (FAILED(hr)) {
        const std::string message = hresult_text("CreateSharedHandle(Vulkan-release fence)", hr);
        set_error(error, error_capacity, message.c_str());
        return 0;
    }
    return static_cast<int64_t>(reinterpret_cast<intptr_t>(handle));
}

int64_t d2s_d3d12_bridge_vulkan_image_handle(void* pointer, uint32_t slot_index,
                                              uint32_t eye_index, char* error,
                                              uint32_t error_capacity) {
    auto* bridge = static_cast<Bridge*>(pointer);
    EyeSlot* eye = get_eye(bridge, slot_index, eye_index);
    if (!eye) {
        set_error(error, error_capacity, "invalid D3D12 target image slot");
        return 0;
    }
    HANDLE handle = nullptr;
    HRESULT hr = create_shared_resource_handle(bridge->target_device.Get(),
                                                eye->target_image.Get(), &handle);
    if (FAILED(hr)) {
        const std::string message = hresult_text("CreateSharedHandle(target image)", hr);
        set_error(error, error_capacity, message.c_str());
        return 0;
    }
    return static_cast<int64_t>(reinterpret_cast<intptr_t>(handle));
}

int d2s_d3d12_bridge_check_slot_ready(void* pointer, uint32_t slot_index,
                                      char* error, uint32_t error_capacity) {
    auto* bridge = static_cast<Bridge*>(pointer);
    if (!bridge || slot_index >= bridge->ring_size) {
        set_error(error, error_capacity, "invalid D3D12 bridge output slot");
        return static_cast<int>(E_INVALIDARG);
    }
    const uint64_t completed_value = bridge->target_fence->GetCompletedValue();
    if (completed_value == UINT64_MAX) {
        set_error(error, error_capacity, "D3D12 target device was removed");
        return static_cast<int>(DXGI_ERROR_DEVICE_REMOVED);
    }
    if (bridge->slot_completions[slot_index] &&
        completed_value < bridge->slot_completions[slot_index]) {
        set_error(error, error_capacity,
                  "D3D12 slot must finish its previous copy before it is reused");
        return static_cast<int>(E_PENDING);
    }
    return 0;
}

int d2s_d3d12_bridge_copy_to_vulkan(void* pointer, uint32_t slot_index,
                                    uint64_t source_fence_value,
                                    uint64_t left_release_value,
                                    uint64_t right_release_value,
                                    uint64_t* completion_value, char* error,
                                    uint32_t error_capacity) {
    auto* bridge = static_cast<Bridge*>(pointer);
    if (!bridge || slot_index >= bridge->ring_size || source_fence_value == 0 || !completion_value) {
        set_error(error, error_capacity, "invalid D3D12 bridge copy arguments");
        return static_cast<int>(E_INVALIDARG);
    }
    HRESULT ready = static_cast<HRESULT>(d2s_d3d12_bridge_check_slot_ready(
        pointer, slot_index, error, error_capacity));
    if (FAILED(ready)) {
        return static_cast<int>(ready);
    }
    EyeSlot* left_eye = get_eye(bridge, slot_index, 0);
    EyeSlot* right_eye = get_eye(bridge, slot_index, 1);
    HRESULT hr = S_OK;
    auto& allocator = bridge->allocators[slot_index];
    auto& command = bridge->command_lists[slot_index];
    hr = allocator->Reset();
    if (FAILED(hr)) {
        const std::string message = hresult_text("reset target command allocator", hr);
        set_error(error, error_capacity, message.c_str());
        return static_cast<int>(hr);
    }
    hr = command->Reset(allocator.Get(), nullptr);
    if (FAILED(hr)) {
        const std::string message = hresult_text("reset target command list", hr);
        set_error(error, error_capacity, message.c_str());
        return static_cast<int>(hr);
    }
    for (uint32_t eye_index = 0; eye_index < 2; ++eye_index) {
        EyeSlot* eye = get_eye(bridge, slot_index, eye_index);
        D3D12_RESOURCE_BARRIER source_barrier{};
        source_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        source_barrier.Transition.pResource = eye->target_buffer.Get();
        source_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        source_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        source_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        command->ResourceBarrier(1, &source_barrier);

        D3D12_RESOURCE_BARRIER image_barrier{};
        image_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        image_barrier.Transition.pResource = eye->target_image.Get();
        image_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        image_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        image_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        command->ResourceBarrier(1, &image_barrier);

        D3D12_TEXTURE_COPY_LOCATION source{};
        source.pResource = eye->target_buffer.Get();
        source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        source.PlacedFootprint.Offset = 0;
        source.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        source.PlacedFootprint.Footprint.Width = bridge->width;
        source.PlacedFootprint.Footprint.Height = bridge->height;
        source.PlacedFootprint.Footprint.Depth = 1;
        source.PlacedFootprint.Footprint.RowPitch = bridge->row_pitch;
        D3D12_TEXTURE_COPY_LOCATION destination{};
        destination.pResource = eye->target_image.Get();
        destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destination.SubresourceIndex = 0;
        command->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);

        source_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        source_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
        command->ResourceBarrier(1, &source_barrier);
        image_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        image_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
        command->ResourceBarrier(1, &image_barrier);
    }
    hr = command->Close();
    if (FAILED(hr)) {
        const std::string message = hresult_text("close target command list", hr);
        set_error(error, error_capacity, message.c_str());
        return static_cast<int>(hr);
    }
    hr = bridge->target_queue->Wait(bridge->target_source_fence.Get(), source_fence_value);
    if (FAILED(hr)) {
        const std::string message = hresult_text("D3D12 target queue wait(source fence)", hr);
        set_error(error, error_capacity, message.c_str());
        return static_cast<int>(hr);
    }
    if (left_release_value) {
        hr = bridge->target_queue->Wait(left_eye->release_fence.Get(), left_release_value);
        if (FAILED(hr)) {
            const std::string message = hresult_text("D3D12 target queue wait(left Vulkan release)", hr);
            set_error(error, error_capacity, message.c_str());
            return static_cast<int>(hr);
        }
    }
    if (right_release_value) {
        hr = bridge->target_queue->Wait(right_eye->release_fence.Get(), right_release_value);
        if (FAILED(hr)) {
            const std::string message = hresult_text("D3D12 target queue wait(right Vulkan release)", hr);
            set_error(error, error_capacity, message.c_str());
            return static_cast<int>(hr);
        }
    }
    ID3D12CommandList* lists[] = {command.Get()};
    bridge->target_queue->ExecuteCommandLists(1, lists);
    const uint64_t value = ++bridge->next_completion;
    hr = bridge->target_queue->Signal(bridge->target_fence.Get(), value);
    if (FAILED(hr)) {
        const std::string message = hresult_text("signal target completion fence", hr);
        set_error(error, error_capacity, message.c_str());
        return static_cast<int>(hr);
    }
    hr = bridge->target_queue->Signal(bridge->target_source_consumed_fence.Get(), value);
    if (FAILED(hr)) {
        const std::string message = hresult_text("signal source-consumed fence", hr);
        set_error(error, error_capacity, message.c_str());
        return static_cast<int>(hr);
    }
    bridge->slot_completions[slot_index] = value;
    *completion_value = value;
    return 0;
}

int d2s_d3d12_bridge_wait(void* pointer, uint64_t completion_value,
                         uint32_t timeout_ms, char* error,
                         uint32_t error_capacity) {
    auto* bridge = static_cast<Bridge*>(pointer);
    if (!bridge || !completion_value) {
        set_error(error, error_capacity, "invalid D3D12 bridge wait arguments");
        return static_cast<int>(E_INVALIDARG);
    }
    if (bridge->target_fence->GetCompletedValue() >= completion_value) return 0;
    HRESULT hr = bridge->target_fence->SetEventOnCompletion(completion_value,
                                                             bridge->target_event);
    if (FAILED(hr)) {
        const std::string message = hresult_text("set D3D12 completion event", hr);
        set_error(error, error_capacity, message.c_str());
        return static_cast<int>(hr);
    }
    const DWORD result = WaitForSingleObject(bridge->target_event, timeout_ms);
    if (result == WAIT_OBJECT_0) return 0;
    if (result == WAIT_TIMEOUT) {
        set_error(error, error_capacity, "timed out waiting for D3D12 GPU copy");
        return static_cast<int>(HRESULT_FROM_WIN32(WAIT_TIMEOUT));
    }
    const HRESULT wait_error = result == WAIT_FAILED
        ? HRESULT_FROM_WIN32(GetLastError()) : E_FAIL;
    const std::string message = hresult_text("wait for D3D12 completion event", wait_error);
    set_error(error, error_capacity, message.c_str());
    return static_cast<int>(wait_error);
}

int d2s_d3d12_bridge_wait_slot(void* pointer, uint32_t slot,
                               uint32_t timeout_ms, char* error,
                               uint32_t error_capacity) {
    auto* bridge = static_cast<Bridge*>(pointer);
    if (!bridge || slot >= bridge->ring_size) {
        set_error(error, error_capacity, "invalid D3D12 bridge slot");
        return static_cast<int>(E_INVALIDARG);
    }
    const uint64_t completion = bridge->slot_completions[slot];
    if (!completion) return 0;
    return d2s_d3d12_bridge_wait(pointer, completion, timeout_ms, error, error_capacity);
}
}
