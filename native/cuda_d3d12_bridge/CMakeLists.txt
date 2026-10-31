cmake_minimum_required(VERSION 3.21)
project(d2s_cuda_d3d12_bridge LANGUAGES CXX)

add_library(d2s_cuda_d3d12_bridge SHARED bridge.cpp)
target_compile_features(d2s_cuda_d3d12_bridge PRIVATE cxx_std_20)
target_compile_definitions(d2s_cuda_d3d12_bridge PRIVATE D2S_CUDA_D3D12_BRIDGE_BUILD=1)
target_link_libraries(d2s_cuda_d3d12_bridge PRIVATE d3d12 dxgi)

if(MSVC)
    set_property(TARGET d2s_cuda_d3d12_bridge PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    target_compile_options(d2s_cuda_d3d12_bridge PRIVATE /W4 /EHsc /permissive-)
endif()

set_target_properties(d2s_cuda_d3d12_bridge PROPERTIES
    OUTPUT_NAME d2s_cuda_d3d12_bridge
    PREFIX "")

install(TARGETS d2s_cuda_d3d12_bridge
    RUNTIME DESTINATION viewer/native/windows
    LIBRARY DESTINATION viewer/native/windows)
