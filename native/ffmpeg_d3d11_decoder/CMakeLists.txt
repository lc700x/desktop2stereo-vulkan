cmake_minimum_required(VERSION 3.21)
project(d2s_ffmpeg_d3d11_decoder LANGUAGES CXX)

if(NOT WIN32)
    message(FATAL_ERROR "d2s_ffmpeg_d3d11_decoder is Windows-only")
endif()

set(D2S_FFMPEG_ROOT "" CACHE PATH "FFmpeg package containing include, lib and bin")
if(NOT EXISTS "${D2S_FFMPEG_ROOT}/include/libavcodec/avcodec.h")
    message(FATAL_ERROR "D2S_FFMPEG_ROOT does not contain libavcodec headers")
endif()

add_library(d2s_ffmpeg_d3d11_decoder SHARED ffmpeg_d3d11_decoder.cpp)
target_compile_features(d2s_ffmpeg_d3d11_decoder PRIVATE cxx_std_20)
target_compile_definitions(d2s_ffmpeg_d3d11_decoder PRIVATE
    D2S_FFMPEG_D3D11_DECODER_BUILD=1 WIN32_LEAN_AND_MEAN NOMINMAX)
target_include_directories(d2s_ffmpeg_d3d11_decoder PRIVATE
    "${D2S_FFMPEG_ROOT}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}")

find_library(D2S_AVFORMAT_LIBRARY NAMES avformat
    PATHS "${D2S_FFMPEG_ROOT}/bin" "${D2S_FFMPEG_ROOT}/lib"
    NO_DEFAULT_PATH REQUIRED)
find_library(D2S_AVCODEC_LIBRARY NAMES avcodec
    PATHS "${D2S_FFMPEG_ROOT}/bin" "${D2S_FFMPEG_ROOT}/lib"
    NO_DEFAULT_PATH REQUIRED)
find_library(D2S_AVUTIL_LIBRARY NAMES avutil
    PATHS "${D2S_FFMPEG_ROOT}/bin" "${D2S_FFMPEG_ROOT}/lib"
    NO_DEFAULT_PATH REQUIRED)

target_link_libraries(d2s_ffmpeg_d3d11_decoder PRIVATE
    "${D2S_AVFORMAT_LIBRARY}"
    "${D2S_AVCODEC_LIBRARY}"
    "${D2S_AVUTIL_LIBRARY}"
    d3d11 dxgi)

if(MSVC)
    target_compile_options(d2s_ffmpeg_d3d11_decoder PRIVATE /utf-8 /W4 /EHsc)
    # Avoid an extra Visual C++ runtime deployment dependency. The bundled
    # FFmpeg DLLs are independent MinGW builds imported by this small bridge.
    set_property(TARGET d2s_ffmpeg_d3d11_decoder PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

set_target_properties(d2s_ffmpeg_d3d11_decoder PROPERTIES
    OUTPUT_NAME d2s_ffmpeg_d3d11_decoder
    PREFIX "")
