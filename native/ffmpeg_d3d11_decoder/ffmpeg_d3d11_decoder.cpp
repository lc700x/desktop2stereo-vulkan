#include "ffmpeg_d3d11_decoder.h"

#ifdef _WIN32
#include <windows.h>
#include <d3d11.h>
#include <d3d11_1.h>
#include <d3d11_3.h>
#include <d3d11_4.h>
#include <dxgi.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_d3d11va.h>
#include <libavutil/pixdesc.h>
}

using Microsoft::WRL::ComPtr;

namespace {

constexpr int kDefaultRingSize = 8;
constexpr int kMaxRingSize = 16;

void set_error(std::string* target, const std::string& text) {
    if (target) {
        *target = text;
    }
}

void copy_error(const std::string& text, char* output, int capacity) {
    if (!output || capacity <= 0) {
        return;
    }
    const size_t count = std::min<size_t>(
        text.size(), static_cast<size_t>(capacity - 1));
    std::memcpy(output, text.data(), count);
    output[count] = '\0';
}

const char* ffmpeg_error(int error) {
    static thread_local char text[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(error, text, sizeof(text));
    return text;
}

std::string utf8_from_wide(const wchar_t* path) {
    if (!path) {
        return {};
    }
    const int length = WideCharToMultiByte(
        CP_UTF8, WC_ERR_INVALID_CHARS, path, -1, nullptr, 0, nullptr, nullptr);
    if (length <= 0) {
        return {};
    }
    std::string result(static_cast<size_t>(length - 1), '\0');
    if (WideCharToMultiByte(
            CP_UTF8, WC_ERR_INVALID_CHARS, path, -1,
            result.data(), length, nullptr, nullptr) <= 0) {
        return {};
    }
    return result;
}

struct OutputSlot {
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11VideoProcessorOutputView> output_view;
    uint64_t shared_handle = 0;
    uint64_t last_ready_value = 0;
};

class Decoder {
public:
    Decoder(std::wstring path, double start_seconds, int ring_size)
        : path_(std::move(path)),
          start_seconds_(std::max(0.0, start_seconds)),
          ring_size_(std::clamp(ring_size, 2, kMaxRingSize)) {}

    ~Decoder() {
        close();
    }

    bool open(std::string* error) {
        const std::string utf8_path = utf8_from_wide(path_.c_str());
        if (utf8_path.empty()) {
            set_error(error, "invalid UTF-8 media path");
            return false;
        }

        int result = avformat_open_input(
            &format_, utf8_path.c_str(), nullptr, nullptr);
        if (result < 0) {
            set_error(error, std::string("avformat_open_input: ") +
                                ffmpeg_error(result));
            return false;
        }
        result = avformat_find_stream_info(format_, nullptr);
        if (result < 0) {
            set_error(error, std::string("avformat_find_stream_info: ") +
                                ffmpeg_error(result));
            return false;
        }
        stream_index_ = av_find_best_stream(
            format_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (stream_index_ < 0) {
            set_error(error, "video stream not found");
            return false;
        }
        AVStream* stream = format_->streams[stream_index_];
        const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
        if (!codec) {
            set_error(error, "video decoder not found");
            return false;
        }

        result = av_hwdevice_ctx_create(
            &hw_device_, AV_HWDEVICE_TYPE_D3D11VA, nullptr, nullptr, 0);
        if (result < 0) {
            set_error(error, std::string("D3D11VA device create: ") +
                                ffmpeg_error(result));
            return false;
        }
        auto* hw_device_context =
            reinterpret_cast<AVHWDeviceContext*>(hw_device_->data);
        d3d11_device_context_ = reinterpret_cast<AVD3D11VADeviceContext*>(
            hw_device_context->hwctx);
        if (!d3d11_device_context_ || !d3d11_device_context_->device ||
            !d3d11_device_context_->device_context ||
            !d3d11_device_context_->video_device ||
            !d3d11_device_context_->video_context) {
            set_error(error, "D3D11VA device did not expose video interfaces");
            return false;
        }
        ComPtr<IDXGIDevice> dxgi_device;
        ComPtr<IDXGIAdapter> dxgi_adapter;
        if (SUCCEEDED(d3d11_device_context_->device->QueryInterface(
                IID_PPV_ARGS(&dxgi_device))) &&
            dxgi_device &&
            SUCCEEDED(dxgi_device->GetAdapter(&dxgi_adapter)) &&
            dxgi_adapter) {
            DXGI_ADAPTER_DESC adapter_desc{};
            if (SUCCEEDED(dxgi_adapter->GetDesc(&adapter_desc))) {
                adapter_luid_low_ = adapter_desc.AdapterLuid.LowPart;
                adapter_luid_high_ = adapter_desc.AdapterLuid.HighPart;
            }
        }
        d3d11_device_context_->MiscFlags |= D3D11_RESOURCE_MISC_SHARED;

        codec_ = avcodec_alloc_context3(codec);
        if (!codec_) {
            set_error(error, "avcodec_alloc_context3 failed");
            return false;
        }
        result = avcodec_parameters_to_context(codec_, stream->codecpar);
        if (result < 0) {
            set_error(error, std::string("avcodec_parameters_to_context: ") +
                                ffmpeg_error(result));
            return false;
        }
        codec_->hw_device_ctx = av_buffer_ref(hw_device_);
        codec_->pkt_timebase = stream->time_base;
        codec_->thread_count = 1;
        result = avcodec_open2(codec_, codec, nullptr);
        if (result < 0) {
            set_error(error, std::string("avcodec_open2: ") +
                                ffmpeg_error(result));
            return false;
        }
        packet_ = av_packet_alloc();
        decoded_frame_ = av_frame_alloc();
        if (!packet_ || !decoded_frame_) {
            set_error(error, "could not allocate FFmpeg packet/frame");
            return false;
        }

        if (!create_shared_fences(error) || !create_video_processor(error)) {
            return false;
        }
        if (!seek_to(start_seconds_, error)) {
            return false;
        }
        return true;
    }

    int next(double target_seconds, D2SFFmpegD3D11Frame* output, std::string* error) {
        if (!output) {
            set_error(error, "frame output pointer is null");
            return -1;
        }
        int64_t target_ns = static_cast<int64_t>(
            std::llround(std::max(0.0, target_seconds) * 1'000'000'000.0));
        if (target_ns <= 0) {
            target_ns = static_cast<int64_t>(
                std::llround(start_seconds_ * 1'000'000'000.0));
        }

        while (true) {
            int result = avcodec_receive_frame(codec_, decoded_frame_);
            if (result == 0) {
                const int64_t pts_ns = frame_timestamp_ns(decoded_frame_);
                const int64_t duration_ns = frame_duration_ns(decoded_frame_);
                if (target_ns > 0 && pts_ns >= 0 &&
                    pts_ns + duration_ns < target_ns) {
                    av_frame_unref(decoded_frame_);
                    continue;
                }
                const bool ready = process_frame(
                    decoded_frame_, pts_ns, duration_ns, output, error);
                av_frame_unref(decoded_frame_);
                return ready ? 1 : -1;
            }
            if (result == AVERROR_EOF) {
                return 0;
            }
            if (result != AVERROR(EAGAIN)) {
                set_error(error, std::string("avcodec_receive_frame: ") +
                                    ffmpeg_error(result));
                return -1;
            }
            if (drain_sent_) {
                return 0;
            }
            result = av_read_frame(format_, packet_);
            if (result == AVERROR_EOF) {
                result = avcodec_send_packet(codec_, nullptr);
                if (result < 0 && result != AVERROR_EOF) {
                    set_error(error, std::string("avcodec_send_packet flush: ") +
                                        ffmpeg_error(result));
                    return -1;
                }
                drain_sent_ = true;
                continue;
            }
            if (result < 0) {
                set_error(error, std::string("av_read_frame: ") +
                                    ffmpeg_error(result));
                return -1;
            }
            if (packet_->stream_index == stream_index_) {
                result = avcodec_send_packet(codec_, packet_);
                av_packet_unref(packet_);
                if (result < 0 && result != AVERROR(EAGAIN)) {
                    set_error(error, std::string("avcodec_send_packet: ") +
                                        ffmpeg_error(result));
                    return -1;
                }
            } else {
                av_packet_unref(packet_);
            }
        }
    }

    bool seek_to(double seconds, std::string* error) {
        if (!format_ || !codec_) {
            set_error(error, "decoder is not open");
            return false;
        }
        AVStream* stream = format_->streams[stream_index_];
        const double time_base = av_q2d(stream->time_base);
        if (!(time_base > 0.0)) {
            set_error(error, "invalid stream time base");
            return false;
        }
        const int64_t timestamp = static_cast<int64_t>(
            std::max(0.0, seconds) / time_base);
        const int result = av_seek_frame(
            format_, stream_index_, timestamp, AVSEEK_FLAG_BACKWARD);
        if (result < 0) {
            set_error(error, std::string("av_seek_frame: ") +
                                ffmpeg_error(result));
            return false;
        }
        avcodec_flush_buffers(codec_);
        av_packet_unref(packet_);
        av_frame_unref(decoded_frame_);
        drain_sent_ = false;
        last_pts_ns_ = -1;
        start_seconds_ = std::max(0.0, seconds);
        return true;
    }

    double position() const {
        return last_pts_ns_ >= 0
            ? static_cast<double>(last_pts_ns_) / 1'000'000'000.0
            : start_seconds_;
    }

    void close() {
        destroy_video_processor();
        slots_.clear();
        if (consumed_fence_) {
            consumed_fence_.Reset();
        }
        if (ready_fence_) {
            ready_fence_.Reset();
        }
        if (consumed_fence_handle_) {
            CloseHandle(consumed_fence_handle_);
            consumed_fence_handle_ = nullptr;
        }
        if (ready_fence_handle_) {
            CloseHandle(ready_fence_handle_);
            ready_fence_handle_ = nullptr;
        }
        if (packet_) {
            av_packet_free(&packet_);
        }
        if (decoded_frame_) {
            av_frame_free(&decoded_frame_);
        }
        if (codec_) {
            avcodec_free_context(&codec_);
        }
        if (hw_device_) {
            av_buffer_unref(&hw_device_);
        }
        if (format_) {
            avformat_close_input(&format_);
        }
        context4_.Reset();
        video_context1_.Reset();
        video_device_.Reset();
        video_context_.Reset();
        d3d11_device_context_ = nullptr;
        stream_index_ = -1;
        drain_sent_ = false;
    }

private:
    bool create_shared_fences(std::string* error) {
        HRESULT result = d3d11_device_context_->device->QueryInterface(
            IID_PPV_ARGS(&device5_));
        if (FAILED(result) || !device5_) {
            set_error(error, "ID3D11Device5 is unavailable");
            return false;
        }
        result = d3d11_device_context_->device_context->QueryInterface(
            IID_PPV_ARGS(&context4_));
        if (FAILED(result) || !context4_) {
            set_error(error, "ID3D11DeviceContext4 is unavailable");
            return false;
        }
        result = d3d11_device_context_->video_context->QueryInterface(
            IID_PPV_ARGS(&video_context1_));
        if (FAILED(result) || !video_context1_) {
            set_error(error, "ID3D11VideoContext1 is unavailable");
            return false;
        }
        result = device5_->CreateFence(
            0, D3D11_FENCE_FLAG_SHARED,
            IID_PPV_ARGS(ready_fence_.GetAddressOf()));
        if (FAILED(result)) {
            set_error(error, "CreateFence(ready) failed");
            return false;
        }
        result = ready_fence_->CreateSharedHandle(
            nullptr, GENERIC_ALL, nullptr, &ready_fence_handle_);
        if (FAILED(result)) {
            set_error(error, "CreateSharedHandle(ready) failed");
            return false;
        }
        result = device5_->CreateFence(
            0, D3D11_FENCE_FLAG_SHARED,
            IID_PPV_ARGS(consumed_fence_.GetAddressOf()));
        if (FAILED(result)) {
            set_error(error, "CreateFence(consumed) failed");
            return false;
        }
        result = consumed_fence_->CreateSharedHandle(
            nullptr, GENERIC_ALL, nullptr, &consumed_fence_handle_);
        if (FAILED(result)) {
            set_error(error, "CreateSharedHandle(consumed) failed");
            return false;
        }
        return true;
    }

    bool create_video_processor(std::string* error) {
        video_device_ = d3d11_device_context_->video_device;
        video_context_ = d3d11_device_context_->video_context;
        if (!video_device_ || !video_context_) {
            set_error(error, "D3D11 video processor interfaces are unavailable");
            return false;
        }
        AVStream* stream = format_->streams[stream_index_];
        const UINT width = static_cast<UINT>(stream->codecpar->width);
        const UINT height = static_cast<UINT>(stream->codecpar->height);
        if (width == 0 || height == 0) {
            set_error(error, "video dimensions are unavailable");
            return false;
        }
        D3D11_VIDEO_PROCESSOR_CONTENT_DESC content{};
        content.InputFrameFormat = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;
        content.InputWidth = width;
        content.InputHeight = height;
        content.OutputWidth = width;
        content.OutputHeight = height;
        content.Usage = D3D11_VIDEO_USAGE_PLAYBACK_NORMAL;
        HRESULT result = video_device_->CreateVideoProcessorEnumerator(
            &content, &enumerator_);
        if (FAILED(result)) {
            set_error(error, "CreateVideoProcessorEnumerator failed");
            return false;
        }
        result = video_device_->CreateVideoProcessor(
            enumerator_.Get(), 0, &processor_);
        if (FAILED(result)) {
            set_error(error, "CreateVideoProcessor failed");
            return false;
        }
        output_width_ = width;
        output_height_ = height;
        return create_output_slots(error);
    }

    bool create_output_slots(std::string* error) {
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = output_width_;
        desc.Height = output_height_;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_RENDER_TARGET |
                         D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
        slots_.resize(static_cast<size_t>(ring_size_));
        for (auto& slot : slots_) {
            HRESULT result = d3d11_device_context_->device->CreateTexture2D(
                &desc, nullptr, &slot.texture);
            if (FAILED(result)) {
                set_error(error, "CreateTexture2D(output) failed");
                return false;
            }
            D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC view_desc{};
            view_desc.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D;
            view_desc.Texture2D.MipSlice = 0;
            result = video_device_->CreateVideoProcessorOutputView(
                slot.texture.Get(), enumerator_.Get(), &view_desc,
                &slot.output_view);
            if (FAILED(result)) {
                set_error(error, "CreateVideoProcessorOutputView failed");
                return false;
            }
            ComPtr<IDXGIResource> resource;
            result = slot.texture->QueryInterface(IID_PPV_ARGS(&resource));
            if (FAILED(result) || !resource) {
                set_error(error, "output texture is not an IDXGIResource");
                return false;
            }
            HANDLE handle = nullptr;
            result = resource->GetSharedHandle(&handle);
            if (FAILED(result) || !handle) {
                set_error(error, "output texture GetSharedHandle failed");
                return false;
            }
            slot.shared_handle = reinterpret_cast<uint64_t>(handle);
        }
        return true;
    }

    int64_t frame_timestamp_ns(const AVFrame* frame) const {
        int64_t timestamp = frame->best_effort_timestamp;
        if (timestamp == AV_NOPTS_VALUE) {
            timestamp = frame->pts;
        }
        if (timestamp == AV_NOPTS_VALUE) {
            return -1;
        }
        const double seconds =
            static_cast<double>(timestamp) *
            av_q2d(format_->streams[stream_index_]->time_base);
        return static_cast<int64_t>(std::llround(seconds * 1'000'000'000.0));
    }

    int64_t frame_duration_ns(const AVFrame* frame) const {
        const AVStream* stream = format_->streams[stream_index_];
        AVRational rate = stream->avg_frame_rate;
        if (rate.num <= 0 || rate.den <= 0) {
            rate = stream->r_frame_rate;
        }
        if (rate.num > 0 && rate.den > 0) {
            return static_cast<int64_t>(
                std::llround(1'000'000'000.0 * rate.den / rate.num));
        }
        if (frame && frame->duration > 0) {
            return static_cast<int64_t>(
                std::llround(
                    static_cast<double>(frame->duration) *
                    av_q2d(stream->time_base) * 1'000'000'000.0));
        }
        return 0;
    }

    bool process_frame(
        AVFrame* frame,
        int64_t pts_ns,
        int64_t duration_ns,
        D2SFFmpegD3D11Frame* output,
        std::string* error) {
        if (frame->format != AV_PIX_FMT_D3D11 || !frame->data[0]) {
            set_error(error, "decoder did not return AV_PIX_FMT_D3D11");
            return false;
        }
        auto* input_texture =
            reinterpret_cast<ID3D11Texture2D*>(frame->data[0]);
        const UINT array_slice = static_cast<UINT>(
            reinterpret_cast<intptr_t>(frame->data[1]));
        D3D11_TEXTURE2D_DESC input_desc{};
        input_texture->GetDesc(&input_desc);
        if (input_desc.Width != output_width_ ||
            input_desc.Height != output_height_) {
            set_error(error, "decoded frame dimensions changed");
            return false;
        }

        if (slots_.empty()) {
            set_error(error, "output slots are not initialized");
            return false;
        }
        const size_t slot_index = static_cast<size_t>(
            next_slot_++ % static_cast<uint32_t>(slots_.size()));
        OutputSlot& slot = slots_[slot_index];
        if (slot.last_ready_value > 0) {
            context4_->Wait(
                consumed_fence_.Get(), slot.last_ready_value);
        }

        D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC input_view_desc{};
        input_view_desc.FourCC = 0;
        input_view_desc.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
        input_view_desc.Texture2D.MipSlice = 0;
        input_view_desc.Texture2D.ArraySlice = array_slice;
        ComPtr<ID3D11VideoProcessorInputView> input_view;
        HRESULT result = video_device_->CreateVideoProcessorInputView(
            input_texture, enumerator_.Get(), &input_view_desc, &input_view);
        if (FAILED(result)) {
            set_error(error, "CreateVideoProcessorInputView failed");
            return false;
        }

        const bool full_range = frame->color_range == AVCOL_RANGE_JPEG;
        DXGI_COLOR_SPACE_TYPE input_color = full_range
            ? DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P709
            : DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709;
        if (frame->color_trc == AVCOL_TRC_SMPTE2084) {
            // D3D11 Video Processor performs the PQ/BT.2020 -> SDR/BT.709
            // tone-map on the GPU. Per-frame mastering metadata is not needed
            // by the generic desktop tone-map used here.
            input_color = DXGI_COLOR_SPACE_YCBCR_STUDIO_G2084_TOPLEFT_P2020;
        } else if (frame->color_trc == AVCOL_TRC_ARIB_STD_B67) {
            input_color = full_range
                ? DXGI_COLOR_SPACE_YCBCR_FULL_GHLG_TOPLEFT_P2020
                : DXGI_COLOR_SPACE_YCBCR_STUDIO_GHLG_TOPLEFT_P2020;
        } else if (frame->color_primaries == AVCOL_PRI_BT2020) {
            input_color = full_range
                ? DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P2020
                : DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P2020;
        }
        const DXGI_COLOR_SPACE_TYPE output_color =
            DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
        video_context_->VideoProcessorSetStreamFrameFormat(
            processor_.Get(), 0, D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE);
        video_context1_->VideoProcessorSetStreamColorSpace1(
            processor_.Get(), 0, input_color);
        video_context1_->VideoProcessorSetOutputColorSpace1(
            processor_.Get(), output_color);

        D3D11_VIDEO_PROCESSOR_STREAM stream_desc{};
        stream_desc.Enable = TRUE;
        stream_desc.OutputIndex = 0;
        stream_desc.InputFrameOrField = 0;
        stream_desc.pInputSurface = input_view.Get();
        result = video_context_->VideoProcessorBlt(
            processor_.Get(), slot.output_view.Get(), 0, 1, &stream_desc);
        if (FAILED(result)) {
            set_error(error, "VideoProcessorBlt failed");
            return false;
        }

        const uint64_t ready_value = ++ready_value_;
        result = context4_->Signal(ready_fence_.Get(), ready_value);
        if (FAILED(result)) {
            set_error(error, "D3D11 Signal(ready) failed");
            return false;
        }
        d3d11_device_context_->device_context->Flush();
        slot.last_ready_value = ready_value;

        D3D11_TEXTURE2D_DESC output_desc{};
        slot.texture->GetDesc(&output_desc);
        output->struct_size = sizeof(D2SFFmpegD3D11Frame);
        output->slot = static_cast<uint32_t>(slot_index);
        output->width = output_desc.Width;
        output->height = output_desc.Height;
        output->dxgi_format = static_cast<uint32_t>(output_desc.Format);
        output->adapter_luid_low = adapter_luid_low_;
        output->adapter_luid_high = adapter_luid_high_;
        output->texture_shared_handle = slot.shared_handle;
        output->ready_fence_handle =
            reinterpret_cast<uint64_t>(ready_fence_handle_);
        output->consumed_fence_handle =
            reinterpret_cast<uint64_t>(consumed_fence_handle_);
        output->ready_fence_value = ready_value;
        output->pts_ns = pts_ns;
        output->duration_ns = duration_ns;
        output->frame_index = frame_index_++;
        output->color_primaries = static_cast<uint32_t>(AVCOL_PRI_BT709);
        output->color_transfer = static_cast<uint32_t>(AVCOL_TRC_BT709);
        output->color_matrix = static_cast<uint32_t>(AVCOL_SPC_BT709);
        output->full_range = 1;
        last_pts_ns_ = pts_ns;
        return true;
    }

    void destroy_video_processor() {
        processor_.Reset();
        enumerator_.Reset();
        output_width_ = 0;
        output_height_ = 0;
        ready_value_ = 0;
        next_slot_ = 0;
    }

private:
    std::wstring path_;
    double start_seconds_ = 0.0;
    int ring_size_ = kDefaultRingSize;
    AVFormatContext* format_ = nullptr;
    AVCodecContext* codec_ = nullptr;
    AVBufferRef* hw_device_ = nullptr;
    AVD3D11VADeviceContext* d3d11_device_context_ = nullptr;
    int stream_index_ = -1;
    AVPacket* packet_ = nullptr;
    AVFrame* decoded_frame_ = nullptr;
    bool drain_sent_ = false;
    int64_t last_pts_ns_ = -1;

    ComPtr<ID3D11Device5> device5_;
    ComPtr<ID3D11DeviceContext4> context4_;
    ComPtr<ID3D11VideoDevice> video_device_;
    ComPtr<ID3D11VideoContext> video_context_;
    ComPtr<ID3D11VideoContext1> video_context1_;
    ComPtr<ID3D11VideoProcessorEnumerator> enumerator_;
    ComPtr<ID3D11VideoProcessor> processor_;
    ComPtr<ID3D11Fence> ready_fence_;
    ComPtr<ID3D11Fence> consumed_fence_;
    HANDLE ready_fence_handle_ = nullptr;
    HANDLE consumed_fence_handle_ = nullptr;
    uint64_t ready_value_ = 0;
    uint32_t next_slot_ = 0;
    uint32_t output_width_ = 0;
    uint32_t output_height_ = 0;
    uint32_t adapter_luid_low_ = 0;
    int32_t adapter_luid_high_ = 0;
    int64_t frame_index_ = 0;
    std::vector<OutputSlot> slots_;
};

}  // namespace

extern "C" D2S_FFMPEG_D3D11_API int d2s_ffmpeg_d3d11_probe(
    char* error, int error_capacity) {
    const AVCodec* decoder = avcodec_find_decoder(AV_CODEC_ID_HEVC);
    if (!decoder) {
        copy_error("FFmpeg HEVC decoder is unavailable", error, error_capacity);
        return 0;
    }
    if (error && error_capacity > 0) {
        error[0] = '\0';
    }
    return 1;
}

extern "C" D2S_FFMPEG_D3D11_API void* d2s_ffmpeg_d3d11_create(
    const wchar_t* path,
    double start_seconds,
    int ring_size,
    char* error,
    int error_capacity) {
    if (!path || !*path) {
        copy_error("media path is empty", error, error_capacity);
        return nullptr;
    }
    auto* decoder = new Decoder(
        path, start_seconds,
        ring_size > 0 ? ring_size : kDefaultRingSize);
    std::string detail;
    if (!decoder->open(&detail)) {
        copy_error(detail, error, error_capacity);
        delete decoder;
        return nullptr;
    }
    if (error && error_capacity > 0) {
        error[0] = '\0';
    }
    return decoder;
}

extern "C" D2S_FFMPEG_D3D11_API int d2s_ffmpeg_d3d11_next(
    void* opaque,
    double target_seconds,
    D2SFFmpegD3D11Frame* frame,
    int timeout_ms,
    char* error,
    int error_capacity) {
    (void)timeout_ms;
    if (!opaque) {
        copy_error("decoder is null", error, error_capacity);
        return -1;
    }
    std::string detail;
    const int result = static_cast<Decoder*>(opaque)->next(
        target_seconds, frame, &detail);
    if (result < 0) {
        copy_error(detail, error, error_capacity);
    } else if (error && error_capacity > 0) {
        error[0] = '\0';
    }
    return result;
}

extern "C" D2S_FFMPEG_D3D11_API int d2s_ffmpeg_d3d11_seek(
    void* opaque,
    double seconds,
    char* error,
    int error_capacity) {
    if (!opaque) {
        copy_error("decoder is null", error, error_capacity);
        return -1;
    }
    std::string detail;
    if (!static_cast<Decoder*>(opaque)->seek_to(seconds, &detail)) {
        copy_error(detail, error, error_capacity);
        return -1;
    }
    return 1;
}

extern "C" D2S_FFMPEG_D3D11_API double d2s_ffmpeg_d3d11_position(void* opaque) {
    return opaque ? static_cast<Decoder*>(opaque)->position() : 0.0;
}

extern "C" D2S_FFMPEG_D3D11_API void d2s_ffmpeg_d3d11_destroy(void* opaque) {
    delete static_cast<Decoder*>(opaque);
}

#endif
