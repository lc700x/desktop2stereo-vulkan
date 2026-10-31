cmake_minimum_required(VERSION 3.21)
project(filament_bridge LANGUAGES CXX)

option(D2S_FILAMENT_VULKAN_EXTERNAL_IMAGE
    "Build against the patched Filament Vulkan external VkImage API" OFF)

if(MSVC)
    # Match Filament's source build, which uses the static MSVC runtime.
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if(WIN32)
    set(_filament_default_sdk "${CMAKE_CURRENT_SOURCE_DIR}/../sdk/windows/v1.75.0")
elseif(APPLE)
    set(_filament_default_sdk "${CMAKE_CURRENT_SOURCE_DIR}/../sdk/macos/v1.75.0/filament")
elseif(UNIX)
    set(_filament_default_sdk "${CMAKE_CURRENT_SOURCE_DIR}/../sdk/linux/v1.75.0/filament")
endif()
set(FILAMENT_SDK_ROOT "${_filament_default_sdk}" CACHE PATH "Root of the platform-specific Filament SDK")
if(NOT FILAMENT_SDK_ROOT)
    message(FATAL_ERROR "Set FILAMENT_SDK_ROOT to the extracted Filament SDK for this platform")
endif()
if(NOT EXISTS "${FILAMENT_SDK_ROOT}/include/filament/Engine.h")
    message(FATAL_ERROR "FILAMENT_SDK_ROOT does not contain include/filament/Engine.h")
endif()

# Filament's public VulkanPlatform header includes BlueVK, but release SDK
# archives do not ship that generated header. Keep the source revision pinned
# and materialize it only in the build tree.
set(_bluevk_include_dir "${CMAKE_CURRENT_BINARY_DIR}/bluevk/include")
set(_bluevk_header "${_bluevk_include_dir}/bluevk/BlueVK.h")
if(NOT EXISTS "${_bluevk_header}")
    file(MAKE_DIRECTORY "${_bluevk_include_dir}/bluevk")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/google/filament/v1.75.0/libs/bluevk/include/bluevk/BlueVK.h"
        "${_bluevk_header}"
        EXPECTED_HASH SHA256=a015af17b3cac987df35fb0896f4bd92497a5937e515db817203baac034ca424
        STATUS _bluevk_download_status
        TLS_VERIFY ON)
    list(GET _bluevk_download_status 0 _bluevk_download_code)
    if(NOT _bluevk_download_code EQUAL 0)
        file(REMOVE "${_bluevk_header}")
        message(FATAL_ERROR "Unable to download pinned Filament BlueVK header: ${_bluevk_download_status}")
    endif()
endif()

file(READ "${_bluevk_header}" _bluevk_contents)
string(REPLACE
    "#if defined(VK_EXT_depth_clamp_control) || defined(VK_EXT_shader_object)"
    "#if defined(VK_EXT_depth_clamp_control)"
    _bluevk_contents
    "${_bluevk_contents}")
file(WRITE "${_bluevk_header}" "${_bluevk_contents}")

find_path(D2S_VULKAN_INCLUDE_DIR vulkan/vulkan.h)
if(NOT D2S_VULKAN_INCLUDE_DIR)
    set(_vulkan_headers_root "${CMAKE_CURRENT_BINARY_DIR}/vulkan-headers")
    set(_vulkan_headers_archive "${CMAKE_CURRENT_BINARY_DIR}/vulkan-headers.tar.gz")
    if(NOT EXISTS "${_vulkan_headers_root}/include/vulkan/vulkan.h")
        file(DOWNLOAD
            "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v1.3.275.tar.gz"
            "${_vulkan_headers_archive}"
            SHOW_PROGRESS
            STATUS _vulkan_headers_download_status
            TLS_VERIFY ON)
        list(GET _vulkan_headers_download_status 0 _vulkan_headers_download_code)
        if(NOT _vulkan_headers_download_code EQUAL 0)
            message(FATAL_ERROR "Unable to download pinned Vulkan headers: ${_vulkan_headers_download_status}")
        endif()
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E tar xzf "${_vulkan_headers_archive}"
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            RESULT_VARIABLE _vulkan_headers_extract_code)
        if(NOT _vulkan_headers_extract_code EQUAL 0)
            message(FATAL_ERROR "Unable to extract pinned Vulkan headers")
        endif()
        file(GLOB _vulkan_headers_extracted LIST_DIRECTORIES true
            "${CMAKE_CURRENT_BINARY_DIR}/Vulkan-Headers-*")
        list(LENGTH _vulkan_headers_extracted _vulkan_headers_extracted_count)
        if(NOT _vulkan_headers_extracted_count EQUAL 1)
            message(FATAL_ERROR "Unexpected Vulkan headers archive layout")
        endif()
        file(RENAME "${_vulkan_headers_extracted}" "${_vulkan_headers_root}")
    endif()
    set(D2S_VULKAN_INCLUDE_DIR "${_vulkan_headers_root}/include")
endif()

add_library(filament_bridge SHARED
    filament_bridge.cpp
    bridge_context.cpp
    bridge_eye.cpp
    bridge_scene.cpp
    bridge_controller.cpp
    bridge_controller_guide.cpp
    bridge_text_overlay.cpp
    bridge_laser.cpp
    bridge_material.cpp
    bridge_material_provider.cpp
    preview_bridge.cpp)
target_compile_features(filament_bridge PRIVATE cxx_std_20)
target_compile_definitions(filament_bridge PRIVATE FILAMENT_BRIDGE_BUILD=1)
if(D2S_FILAMENT_VULKAN_EXTERNAL_IMAGE)
    target_compile_definitions(filament_bridge PRIVATE D2S_FILAMENT_VULKAN_EXTERNAL_IMAGE=1)
endif()
if(MSVC)
    target_compile_options(filament_bridge PRIVATE /GR-)
else()
    target_compile_options(filament_bridge PRIVATE -fno-rtti)
endif()
target_include_directories(filament_bridge PRIVATE
    "${FILAMENT_SDK_ROOT}/include"
    "${_bluevk_include_dir}"
    "${D2S_VULKAN_INCLUDE_DIR}")

set(_filament_arch x86_64)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$" OR CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    set(_filament_arch arm64)
endif()

if(WIN32)
    set(_filament_lib_dir "${FILAMENT_SDK_ROOT}/lib/${_filament_arch}/mt")
    # Filament's Windows static backend contains PlatformWGL symbols even
    # when the selected engine backend is Vulkan.
    target_link_libraries(filament_bridge PRIVATE user32 gdi32 shlwapi opengl32)
elseif(APPLE)
    set(_filament_lib_dir "${FILAMENT_SDK_ROOT}/lib/${_filament_arch}")
    find_library(COCOA_FRAMEWORK Cocoa REQUIRED)
    find_library(COREVIDEO_FRAMEWORK CoreVideo REQUIRED)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
    target_link_libraries(filament_bridge PRIVATE
        ${COCOA_FRAMEWORK}
        ${COREVIDEO_FRAMEWORK}
        ${METAL_FRAMEWORK}
        ${QUARTZCORE_FRAMEWORK})
elseif(UNIX)
    set(_filament_lib_dir "${FILAMENT_SDK_ROOT}/lib/${_filament_arch}")
    target_link_libraries(filament_bridge PRIVATE dl pthread)
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

if(WIN32)
    # Link only the SDK libraries required by this bridge. The SDK directory
    # also contains app/viewer/resource archives that are not interchangeable
    # with Filament's runtime libraries.
    set(_filament_library_names
        abseil backend basis_transcoder bluegl bluevk dracodec filabridge
        filaflat filamat filament filament-generatePrefilterMipmap geometry
        gltfio gltfio_core ibl ktxreader meshoptimizer shaders smol-v stb
        uberzlib utils zstd)
    set(_filament_libraries)
    foreach(_filament_library IN LISTS _filament_library_names)
        set(_filament_library_path "${_filament_lib_dir}/${_filament_library}.lib")
        if(NOT EXISTS "${_filament_library_path}")
            message(FATAL_ERROR "Required Filament library not found: ${_filament_library_path}")
        endif()
        list(APPEND _filament_libraries "${_filament_library_path}")
    endforeach()
else()
    file(GLOB _filament_libraries CONFIGURE_DEPENDS
        "${_filament_lib_dir}/*.lib"
        "${_filament_lib_dir}/*.a")
endif()
if(NOT _filament_libraries)
    message(FATAL_ERROR "No Filament libraries found in ${_filament_lib_dir}")
endif()
target_link_libraries(filament_bridge PRIVATE ${_filament_libraries})

if(WIN32)
    set(_filament_output_platform windows)
elseif(APPLE)
    set(_filament_output_platform macos)
elseif(UNIX)
    set(_filament_output_platform linux)
endif()
set(_filament_output_dir
    "${CMAKE_SOURCE_DIR}/../../../src/desktop2stereo/xr_viewer/native/${_filament_output_platform}")
file(MAKE_DIRECTORY "${_filament_output_dir}")
set_target_properties(filament_bridge PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    OUTPUT_NAME filament_bridge
    RUNTIME_OUTPUT_DIRECTORY "${_filament_output_dir}"
    LIBRARY_OUTPUT_DIRECTORY "${_filament_output_dir}")

foreach(_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(filament_bridge PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_${_config} "${_filament_output_dir}"
        LIBRARY_OUTPUT_DIRECTORY_${_config} "${_filament_output_dir}")
endforeach()
