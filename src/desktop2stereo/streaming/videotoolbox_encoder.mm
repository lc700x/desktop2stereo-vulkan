#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreVideo/CVMetalTextureCache.h>
#import <Metal/Metal.h>
#import <VideoToolbox/VideoToolbox.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

struct D2SVideoToolboxEncoder {
  CFTypeRef device = nullptr;
  CFTypeRef source_buffer = nullptr;
  CFTypeRef command_queue = nullptr;
  CFTypeRef resize_pipeline = nullptr;
  CVPixelBufferPoolRef pixel_pool = nullptr;
  CVMetalTextureCacheRef texture_cache = nullptr;
  VTCompressionSessionRef session = nullptr;
  int width = 0;
  int height = 0;
  int source_width = 0;
  int source_height = 0;
  int fps = 0;
  bool hevc = false;
  std::mutex mutex;
  std::condition_variable completed;
  bool waiting = false;
  bool failed = false;
  bool ready = false;
  bool keyframe = false;
  int64_t active_frame_index = 0;
  std::vector<uint8_t> bytes;
  std::string error;
};

struct D2SResizeParams {
  uint32_t source_width;
  uint32_t source_height;
  uint32_t target_width;
  uint32_t target_height;
  uint32_t region_count;
  uint32_t source_bgra;
  uint32_t regions[16];
};

static NSString *const D2SResizeMetalSource = @"\
#include <metal_stdlib>\n\
using namespace metal;\n\
struct ResizeParams { uint sw; uint sh; uint dw; uint dh; uint count; uint source_bgra; uint regions[16]; };\n\
kernel void d2s_resize_rgba(device const uchar *source [[buffer(0)]],\n\
                            texture2d<float, access::write> target [[texture(0)]],\n\
                            constant ResizeParams& p [[buffer(1)]],\n\
                            uint2 gid [[thread_position_in_grid]]) {\n\
  if (gid.x >= p.dw || gid.y >= p.dh) return;\n\
  float4 color = float4(0.0f, 0.0f, 0.0f, 1.0f);\n\
  for (uint region = 0; region < p.count; ++region) {\n\
    uint base = region * 8;\n\
    uint sx0 = p.regions[base]; uint sy0 = p.regions[base + 1];\n\
    uint sx1 = p.regions[base + 2]; uint sy1 = p.regions[base + 3];\n\
    uint dx0 = p.regions[base + 4]; uint dy0 = p.regions[base + 5];\n\
    uint dx1 = p.regions[base + 6]; uint dy1 = p.regions[base + 7];\n\
    if (gid.x < dx0 || gid.y < dy0 || gid.x >= dx1 || gid.y >= dy1) continue;\n\
    float2 src_size = float2(sx1 - sx0, sy1 - sy0);\n\
    float2 dst_size = float2(dx1 - dx0, dy1 - dy0);\n\
    float2 pixel = float2(gid) + 0.5f;\n\
    float2 coord = float2(sx0, sy0) + ((pixel - float2(dx0, dy0)) / dst_size) * src_size - 0.5f;\n\
    coord = clamp(coord, float2(sx0, sy0), float2(sx1 - 1, sy1 - 1));\n\
    uint2 p0 = uint2(floor(coord));\n\
    uint2 p1 = min(p0 + 1u, uint2(sx1 - 1, sy1 - 1));\n\
    float2 f = coord - float2(p0);\n\
    uint i00 = (p0.y * p.sw + p0.x) * 4;\n\
    uint i10 = (p0.y * p.sw + p1.x) * 4;\n\
    uint i01 = (p1.y * p.sw + p0.x) * 4;\n\
    uint i11 = (p1.y * p.sw + p1.x) * 4;\n\
    for (uint channel = 0; channel < 3; ++channel) {\n\
      uint source_channel = p.source_bgra != 0 && channel != 1 ? 2 - channel : channel;\n\
      float a = mix(float(source[i00 + source_channel]), float(source[i10 + source_channel]), f.x);\n\
      float b = mix(float(source[i01 + source_channel]), float(source[i11 + source_channel]), f.x);\n\
      color[channel] = mix(a, b, f.y) / 255.0f;\n\
    }\n\
    color.a = 1.0f;\n\
    break;\n\
  }\n\
  target.write(color, gid);\n\
}\n\
";

struct D2SFrameContext {
  D2SVideoToolboxEncoder *owner;
  CVPixelBufferRef pixel_buffer;
  int64_t frame_index;
};

static void d2s_set_error(char *buffer, size_t capacity, const std::string &error) {
  if (!buffer || capacity == 0) return;
  std::strncpy(buffer, error.c_str(), capacity - 1);
  buffer[capacity - 1] = '\0';
}

static void d2s_append_start_code(std::vector<uint8_t> &out,
                                  const uint8_t *data, size_t size) {
  static constexpr uint8_t start_code[] = {0, 0, 0, 1};
  out.insert(out.end(), start_code, start_code + sizeof(start_code));
  out.insert(out.end(), data, data + size);
}

static bool d2s_append_parameter_sets(CMFormatDescriptionRef format,
                                      bool hevc,
                                      std::vector<uint8_t> &out,
                                      int *nal_length_size) {
  size_t count = 0;
  int header_length = 0;
  OSStatus status = noErr;
  if (hevc) {
    const uint8_t *parameter_set = nullptr;
    size_t parameter_size = 0;
    status = CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
        format, 0, &parameter_set, &parameter_size, &count, &header_length);
    if (status != noErr) return false;
    for (size_t index = 0; index < count; ++index) {
      status = CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
          format, index, &parameter_set, &parameter_size, nullptr, nullptr);
      if (status != noErr) return false;
      d2s_append_start_code(out, parameter_set, parameter_size);
    }
  } else {
    const uint8_t *parameter_set = nullptr;
    size_t parameter_size = 0;
    status = CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
        format, 0, &parameter_set, &parameter_size, &count, &header_length);
    if (status != noErr) return false;
    for (size_t index = 0; index < count; ++index) {
      status = CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
          format, index, &parameter_set, &parameter_size, nullptr, nullptr);
      if (status != noErr) return false;
      d2s_append_start_code(out, parameter_set, parameter_size);
    }
  }
  *nal_length_size = header_length;
  return header_length > 0 && header_length <= 4;
}

static bool d2s_append_annex_b(CMBlockBufferRef block,
                               CMFormatDescriptionRef format,
                               bool hevc, bool keyframe,
                               std::vector<uint8_t> &out) {
  const size_t total = CMBlockBufferGetDataLength(block);
  if (total == 0) return false;
  std::vector<uint8_t> data(total);
  if (CMBlockBufferCopyDataBytes(block, 0, total, data.data()) != noErr) return false;

  int nal_length_size = 0;
  if (keyframe && !d2s_append_parameter_sets(format, hevc, out, &nal_length_size)) {
    return false;
  }
  if (nal_length_size == 0) {
    const uint8_t *parameter_set = nullptr;
    size_t parameter_size = 0;
    size_t count = 0;
    int header_length = 0;
    if (hevc) {
      CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
          format, 0, &parameter_set, &parameter_size, &count, &header_length);
    } else {
      CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
          format, 0, &parameter_set, &parameter_size, &count, &header_length);
    }
    nal_length_size = header_length;
  }
  if (nal_length_size < 1 || nal_length_size > 4) return false;
  size_t offset = 0;
  while (offset + static_cast<size_t>(nal_length_size) <= total) {
    uint32_t length = 0;
    for (int byte = 0; byte < nal_length_size; ++byte) {
      length = (length << 8) | data[offset + byte];
    }
    offset += static_cast<size_t>(nal_length_size);
    if (length == 0 || length > total - offset) return false;
    d2s_append_start_code(out, data.data() + offset, length);
    offset += length;
  }
  return offset == total;
}

static void d2s_output_callback(void *, void *source_frame_refcon, OSStatus status,
                                VTEncodeInfoFlags, CMSampleBufferRef sample) {
  auto *frame = static_cast<D2SFrameContext *>(source_frame_refcon);
  auto *owner = frame->owner;
  const int64_t frame_index = frame->frame_index;
  std::vector<uint8_t> encoded;
  bool keyframe = false;
  std::string error;
  if (status != noErr) {
    error = "VideoToolbox encoding failed (" + std::to_string(status) + ")";
  } else if (sample && CMSampleBufferDataIsReady(sample)) {
    CFArrayRef attachments = CMSampleBufferGetSampleAttachmentsArray(sample, false);
    if (attachments && CFArrayGetCount(attachments) > 0) {
      auto *attachment = (CFDictionaryRef)CFArrayGetValueAtIndex(attachments, 0);
      keyframe = !CFDictionaryContainsKey(attachment, kCMSampleAttachmentKey_NotSync);
    }
    auto block = CMSampleBufferGetDataBuffer(sample);
    auto format = CMSampleBufferGetFormatDescription(sample);
    if (!block || !format ||
        !d2s_append_annex_b(block, format, owner->hevc, keyframe, encoded)) {
      error = "VideoToolbox returned an invalid compressed access unit";
    }
  } else {
    error = "VideoToolbox returned an empty compressed frame";
  }
  // Return the pooled pixel buffer before waking the caller. The pool is
  // intentionally capped at one live buffer, so the next encode must not race
  // this callback's release after its completion signal.
  CVPixelBufferRelease(frame->pixel_buffer);
  delete frame;
  {
    std::lock_guard<std::mutex> guard(owner->mutex);
    if (owner->waiting && owner->active_frame_index == frame_index) {
      owner->bytes = std::move(encoded);
      owner->error = std::move(error);
      owner->keyframe = keyframe;
      owner->ready = true;
    }
  }
  owner->completed.notify_one();
}

static void d2s_release_encoder_resources(D2SVideoToolboxEncoder *encoder) {
  if (encoder->session) {
    VTCompressionSessionCompleteFrames(encoder->session, kCMTimeInvalid);
    VTCompressionSessionInvalidate(encoder->session);
    CFRelease(encoder->session);
    encoder->session = nullptr;
  }
  if (encoder->pixel_pool) {
    CFRelease(encoder->pixel_pool);
    encoder->pixel_pool = nullptr;
  }
  if (encoder->texture_cache) {
    CFRelease(encoder->texture_cache);
    encoder->texture_cache = nullptr;
  }
  if (encoder->source_buffer) {
    CFRelease(encoder->source_buffer);
    encoder->source_buffer = nullptr;
  }
  if (encoder->command_queue) {
    CFRelease(encoder->command_queue);
    encoder->command_queue = nullptr;
  }
  if (encoder->resize_pipeline) {
    CFRelease(encoder->resize_pipeline);
    encoder->resize_pipeline = nullptr;
  }
  if (encoder->device) {
    CFRelease(encoder->device);
    encoder->device = nullptr;
  }
}

extern "C" {

void *d2s_videotoolbox_create(int width, int height, int fps, int bitrate,
                             int hevc, char *error, size_t error_capacity) {
  if (width < 2 || height < 2 || (width & 1) || (height & 1) ||
      width > 8192 || height > 8192 || fps < 1 || fps > 240 ||
      bitrate < 64000 || (hevc != 0 && hevc != 1)) {
    d2s_set_error(error, error_capacity, "Invalid VideoToolbox encoder settings");
    return nullptr;
  }
  auto *encoder = new D2SVideoToolboxEncoder();
  encoder->width = width;
  encoder->height = height;
  encoder->fps = fps;
  encoder->hevc = hevc != 0;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      d2s_set_error(error, error_capacity, "Metal is unavailable on this Mac");
      delete encoder;
      return nullptr;
    }
    encoder->device = CFRetain((__bridge CFTypeRef)device);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSError *metal_error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:D2SResizeMetalSource
                                                  options:nil error:&metal_error];
    id<MTLFunction> function = [library newFunctionWithName:@"d2s_resize_rgba"];
    id<MTLComputePipelineState> pipeline = function
        ? [device newComputePipelineStateWithFunction:function error:&metal_error]
        : nil;
    if (!queue || !pipeline) {
      d2s_set_error(error, error_capacity,
          "Could not initialize the Metal stereo-fit pipeline: " +
          std::string(metal_error.localizedDescription.UTF8String ?: "unknown Metal error"));
      d2s_release_encoder_resources(encoder);
      delete encoder;
      return nullptr;
    }
    encoder->command_queue = CFRetain((__bridge CFTypeRef)queue);
    encoder->resize_pipeline = CFRetain((__bridge CFTypeRef)pipeline);

    NSDictionary *pixel_attributes = @{
      (__bridge NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
      (__bridge NSString *)kCVPixelBufferWidthKey: @(width),
      (__bridge NSString *)kCVPixelBufferHeightKey: @(height),
      (__bridge NSString *)kCVPixelBufferMetalCompatibilityKey: @YES,
      (__bridge NSString *)kCVPixelBufferIOSurfacePropertiesKey: @{},
    };
    NSDictionary *pool_attributes = @{
      (__bridge NSString *)kCVPixelBufferPoolMinimumBufferCountKey: @1,
    };
    OSStatus status = CVPixelBufferPoolCreate(
        kCFAllocatorDefault, (__bridge CFDictionaryRef)pool_attributes,
        (__bridge CFDictionaryRef)pixel_attributes, &encoder->pixel_pool);
    if (status == kCVReturnSuccess) {
      status = CVMetalTextureCacheCreate(
          kCFAllocatorDefault, nullptr, device, nullptr, &encoder->texture_cache);
    }
    if (status != kCVReturnSuccess || !encoder->pixel_pool || !encoder->texture_cache) {
      d2s_set_error(error, error_capacity,
          "Could not create Metal-compatible CoreVideo encoder buffers (" +
          std::to_string(status) + ")");
      d2s_release_encoder_resources(encoder);
      delete encoder;
      return nullptr;
    }

    NSDictionary *specification = @{
      (__bridge NSString *)kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder: @YES,
    };
    OSStatus session_status = VTCompressionSessionCreate(
        kCFAllocatorDefault, width, height,
        hevc ? kCMVideoCodecType_HEVC : kCMVideoCodecType_H264,
        (__bridge CFDictionaryRef)specification,
        (__bridge CFDictionaryRef)pixel_attributes,
        nullptr, d2s_output_callback, encoder, &encoder->session);
    if (session_status != noErr || !encoder->session) {
      d2s_set_error(error, error_capacity,
          "Hardware VideoToolbox encoder is unavailable (" +
          std::to_string(session_status) + ")");
      d2s_release_encoder_resources(encoder);
      delete encoder;
      return nullptr;
    }
    VTSessionSetProperty(encoder->session, kVTCompressionPropertyKey_RealTime, kCFBooleanTrue);
    VTSessionSetProperty(encoder->session, kVTCompressionPropertyKey_AllowFrameReordering, kCFBooleanFalse);
    VTSessionSetProperty(encoder->session, kVTCompressionPropertyKey_ExpectedFrameRate,
                         (__bridge CFNumberRef)@(fps));
    VTSessionSetProperty(encoder->session, kVTCompressionPropertyKey_AverageBitRate,
                         (__bridge CFNumberRef)@(bitrate));
    VTSessionSetProperty(encoder->session, kVTCompressionPropertyKey_MaxKeyFrameInterval,
                         (__bridge CFNumberRef)@(fps * 2));
    VTSessionSetProperty(encoder->session, kVTCompressionPropertyKey_ProfileLevel,
        hevc ? kVTProfileLevel_HEVC_Main_AutoLevel : kVTProfileLevel_H264_High_AutoLevel);
    session_status = VTCompressionSessionPrepareToEncodeFrames(encoder->session);
    if (session_status != noErr) {
      d2s_set_error(error, error_capacity,
          "VideoToolbox could not prepare the hardware encoder (" +
          std::to_string(session_status) + ")");
      d2s_release_encoder_resources(encoder);
      delete encoder;
      return nullptr;
    }
  }
  return encoder;
}

void *d2s_videotoolbox_source_buffer(void *handle, int width, int height,
                                    char *error, size_t error_capacity) {
  auto *encoder = static_cast<D2SVideoToolboxEncoder *>(handle);
  if (!encoder || width < 1 || height < 1 ||
      static_cast<uint64_t>(width) * height > 268435456ULL) {
    d2s_set_error(error, error_capacity, "Invalid native Metal source frame dimensions");
    return nullptr;
  }
  if (!encoder->source_buffer || encoder->source_width != width || encoder->source_height != height) {
    if (encoder->source_buffer) CFRelease(encoder->source_buffer);
    id<MTLDevice> device = (__bridge id<MTLDevice>)encoder->device;
    id<MTLBuffer> buffer = [device newBufferWithLength:(NSUInteger)width * height * 4
                                               options:MTLResourceStorageModeShared];
    if (!buffer) {
      encoder->source_buffer = nullptr;
      d2s_set_error(error, error_capacity, "Could not allocate the native Metal source buffer");
      return nullptr;
    }
    encoder->source_buffer = CFRetain((__bridge CFTypeRef)buffer);
    encoder->source_width = width;
    encoder->source_height = height;
  }
  return (__bridge void *)(__bridge id<MTLBuffer>)encoder->source_buffer;
}

int d2s_videotoolbox_encode(void *handle, int64_t frame_index, int force_keyframe,
                            int region_count, int source_bgra, const int32_t *regions,
                            const uint8_t **output, size_t *output_size,
                            int *is_keyframe, char *error, size_t error_capacity) {
  auto *encoder = static_cast<D2SVideoToolboxEncoder *>(handle);
  if (!encoder || !output || !output_size || !is_keyframe || !regions ||
      region_count < 1 || region_count > 2 || !encoder->source_buffer) {
    d2s_set_error(error, error_capacity, "Invalid VideoToolbox encode request");
    return -1;
  }
  for (int region = 0; region < region_count; ++region) {
    const int32_t *r = regions + region * 8;
    if (r[0] < 0 || r[1] < 0 || r[2] <= r[0] || r[3] <= r[1] ||
        r[2] > encoder->source_width || r[3] > encoder->source_height ||
        r[4] < 0 || r[5] < 0 || r[6] <= r[4] || r[7] <= r[5] ||
        r[6] > encoder->width || r[7] > encoder->height) {
      d2s_set_error(error, error_capacity, "Invalid stereo fit region");
      return -1;
    }
  }
  {
    std::lock_guard<std::mutex> guard(encoder->mutex);
    if (encoder->failed) {
      d2s_set_error(error, error_capacity, "VideoToolbox encoder failed and must reconnect");
      return -1;
    }
    if (encoder->waiting) {
      d2s_set_error(error, error_capacity, "VideoToolbox encoder is already processing a frame");
      return -1;
    }
    encoder->waiting = true;
    encoder->ready = false;
    encoder->active_frame_index = frame_index;
    encoder->bytes.clear();
    encoder->error.clear();
  }

  CVPixelBufferRef pixel_buffer = nullptr;
  NSDictionary *auxiliary_attributes = @{
    (__bridge NSString *)kCVPixelBufferPoolAllocationThresholdKey: @1,
  };
  OSStatus status = CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(
      kCFAllocatorDefault, encoder->pixel_pool,
      (__bridge CFDictionaryRef)auxiliary_attributes, &pixel_buffer);
  if (status != kCVReturnSuccess || !pixel_buffer) {
    std::lock_guard<std::mutex> guard(encoder->mutex);
    encoder->waiting = false;
    d2s_set_error(error, error_capacity,
        "Could not acquire a pooled CoreVideo frame (" + std::to_string(status) + ")");
    return -1;
  }

  @autoreleasepool {
    CVMetalTextureRef wrapped_texture = nullptr;
    status = CVMetalTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault, encoder->texture_cache, pixel_buffer, nullptr,
        MTLPixelFormatBGRA8Unorm, encoder->width, encoder->height, 0,
        &wrapped_texture);
    id<MTLTexture> target = wrapped_texture
        ? CVMetalTextureGetTexture(wrapped_texture) : nil;
    id<MTLBuffer> source = (__bridge id<MTLBuffer>)encoder->source_buffer;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)encoder->command_queue;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)encoder->resize_pipeline;
    id<MTLCommandBuffer> command = [queue commandBuffer];
    if (status != kCVReturnSuccess || !target || !command) {
      if (wrapped_texture) CFRelease(wrapped_texture);
      CVPixelBufferRelease(pixel_buffer);
      std::lock_guard<std::mutex> guard(encoder->mutex);
      encoder->waiting = false;
      d2s_set_error(error, error_capacity,
          "Could not create a Metal view of the pooled CoreVideo frame (" +
          std::to_string(status) + ")");
      return -1;
    }
    D2SResizeParams params{};
    params.source_width = encoder->source_width;
    params.source_height = encoder->source_height;
    params.target_width = encoder->width;
    params.target_height = encoder->height;
    params.region_count = static_cast<uint32_t>(region_count);
    params.source_bgra = source_bgra ? 1u : 0u;
    for (int i = 0; i < region_count * 8; ++i) {
      params.regions[i] = static_cast<uint32_t>(regions[i]);
    }
    const int32_t *region = regions;
    const bool matching_dimensions =
        encoder->source_width == encoder->width &&
        encoder->source_height == encoder->height;
    const bool full_identity_region = region_count == 1 &&
        region[0] == 0 && region[1] == 0 &&
        region[2] == encoder->source_width && region[3] == encoder->source_height &&
        region[4] == 0 && region[5] == 0 &&
        region[6] == encoder->width && region[7] == encoder->height;
    const bool half_sbs_identity_regions = region_count == 2 &&
        encoder->source_width % 2 == 0 &&
        region[0] == 0 && region[1] == 0 &&
        region[2] == encoder->source_width / 2 &&
        region[3] == encoder->source_height &&
        region[4] == 0 && region[5] == 0 &&
        region[6] == encoder->width / 2 && region[7] == encoder->height &&
        region[8] == encoder->source_width / 2 && region[9] == 0 &&
        region[10] == encoder->source_width && region[11] == encoder->source_height &&
        region[12] == encoder->width / 2 && region[13] == 0 &&
        region[14] == encoder->width && region[15] == encoder->height;
    const bool identity_fit = source_bgra && matching_dimensions &&
        (full_identity_region || half_sbs_identity_regions);
    if (identity_fit) {
      id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
      if (!blit) {
        if (wrapped_texture) CFRelease(wrapped_texture);
        CVPixelBufferRelease(pixel_buffer);
        std::lock_guard<std::mutex> guard(encoder->mutex);
        encoder->waiting = false;
        d2s_set_error(error, error_capacity,
            "Could not create a Metal copy for the encoder pixel buffer");
        return -1;
      }
      [blit copyFromBuffer:source sourceOffset:0
          sourceBytesPerRow:static_cast<NSUInteger>(encoder->source_width) * 4u
          sourceBytesPerImage:static_cast<NSUInteger>(encoder->source_width) *
                              static_cast<NSUInteger>(encoder->source_height) * 4u
          sourceSize:MTLSizeMake(encoder->source_width, encoder->source_height, 1)
          toTexture:target destinationSlice:0 destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)];
      [blit endEncoding];
    } else {
      id<MTLComputeCommandEncoder> compute = [command computeCommandEncoder];
      if (!compute) {
        if (wrapped_texture) CFRelease(wrapped_texture);
        CVPixelBufferRelease(pixel_buffer);
        std::lock_guard<std::mutex> guard(encoder->mutex);
        encoder->waiting = false;
        d2s_set_error(error, error_capacity,
            "Could not create a Metal stereo-fit pass");
        return -1;
      }
      [compute setComputePipelineState:pipeline];
      [compute setBuffer:source offset:0 atIndex:0];
      [compute setTexture:target atIndex:0];
      [compute setBytes:&params length:sizeof(params) atIndex:1];
      NSUInteger tw = std::min<NSUInteger>(16, pipeline.maxTotalThreadsPerThreadgroup);
      NSUInteger th = std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup / tw);
      [compute dispatchThreads:MTLSizeMake(encoder->width, encoder->height, 1)
          threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
      [compute endEncoding];
    }
    [command commit];
    [command waitUntilCompleted];
    if (wrapped_texture) CFRelease(wrapped_texture);
    if (command.status != MTLCommandBufferStatusCompleted) {
      CVPixelBufferRelease(pixel_buffer);
      std::lock_guard<std::mutex> guard(encoder->mutex);
      encoder->waiting = false;
      d2s_set_error(error, error_capacity, "Metal could not fit the stereo frame to the negotiated size");
      return -1;
    }
  }

  auto *frame = new D2SFrameContext{encoder, pixel_buffer, frame_index};
  CFMutableDictionaryRef properties = nullptr;
  if (force_keyframe) {
    properties = CFDictionaryCreateMutable(kCFAllocatorDefault, 1,
        &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    if (properties) {
      CFDictionarySetValue(properties, kVTEncodeFrameOptionKey_ForceKeyFrame, kCFBooleanTrue);
    }
  }
  VTEncodeInfoFlags flags = 0;
  status = VTCompressionSessionEncodeFrame(
      encoder->session, pixel_buffer, CMTimeMake(frame_index, encoder->fps),
      CMTimeMake(1, encoder->fps), properties, frame, &flags);
  if (properties) CFRelease(properties);
  if (status != noErr) {
    CVPixelBufferRelease(pixel_buffer);
    delete frame;
    std::lock_guard<std::mutex> guard(encoder->mutex);
    encoder->waiting = false;
    d2s_set_error(error, error_capacity,
        "VideoToolbox rejected the frame (" + std::to_string(status) + ")");
    return -1;
  }

  std::unique_lock<std::mutex> lock(encoder->mutex);
  if (!encoder->completed.wait_for(lock, std::chrono::seconds(2),
                                   [&] { return encoder->ready; })) {
    encoder->waiting = false;
    encoder->failed = true;
    d2s_set_error(error, error_capacity, "VideoToolbox frame encoding timed out; reconnect to restart it");
    return -1;
  }
  encoder->waiting = false;
  if (!encoder->error.empty()) {
    d2s_set_error(error, error_capacity, encoder->error);
    return -1;
  }
  *output = encoder->bytes.data();
  *output_size = encoder->bytes.size();
  *is_keyframe = encoder->keyframe ? 1 : 0;
  return 0;
}

void d2s_videotoolbox_destroy(void *handle) {
  auto *encoder = static_cast<D2SVideoToolboxEncoder *>(handle);
  if (!encoder) return;
  d2s_release_encoder_resources(encoder);
  delete encoder;
}

}
