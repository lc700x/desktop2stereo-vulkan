#import "macos_media_player.h"

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreFoundation/CoreFoundation.h>
#import <Foundation/Foundation.h>
#import <VideoToolbox/VideoToolbox.h>
#import <os/lock.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#import <dispatch/dispatch.h>

@interface D2SMediaOutputDelegate : NSObject <AVPlayerItemOutputPullDelegate>
@end

@implementation D2SMediaOutputDelegate
- (void)outputMediaDataWillChange:(AVPlayerItemOutput *)sender {
    (void)sender;
}

- (void)outputSequenceWasFlushed:(AVPlayerItemOutput *)sender {
    (void)sender;
}
@end

struct D2SMediaPlayer {
    __strong AVPlayer *player;
    __strong AVPlayerItem *item;
    __strong AVAsset *asset;
    __strong AVPlayerItemVideoOutput *video_output;
    __strong D2SMediaOutputDelegate *output_delegate;
    dispatch_queue_t output_queue;
    __strong AVAssetReader *reader;
    __strong AVAssetReaderTrackOutput *reader_output;
    CMSampleBufferRef pending_sample;
    VTDecompressionSessionRef decompression_session;
    CMVideoFormatDescriptionRef format_description;
    CVPixelBufferRef decoded_frame;
    CMTime decoded_pts;
    os_unfair_lock decoder_lock;
    BOOL reader_eof;
    BOOL hdr;
    BOOL has_last_frame_pts;
    double last_frame_pts;
    char error[512];
};

static void d2s_media_set_error(D2SMediaPlayer *context, NSString *message) {
    if (context == NULL) return;
    const char *text = message.UTF8String ?: "unknown AVFoundation media error";
    snprintf(context->error, sizeof(context->error), "%s", text);
}

static void d2s_media_set_create_error(char *buffer, size_t capacity,
                                       NSString *message) {
    if (buffer == NULL || capacity == 0) return;
    const char *text = message.UTF8String ?: "unknown AVFoundation media error";
    snprintf(buffer, capacity, "%s", text);
}

static BOOL d2s_is_p010(OSType format) {
    return format == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange ||
           format == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange;
}

static void d2s_media_reset_reader(D2SMediaPlayer *context) {
    if (context == NULL) return;
    if (context->pending_sample != NULL) {
        CFRelease(context->pending_sample);
        context->pending_sample = NULL;
    }
    [context->reader cancelReading];
    context->reader_output = nil;
    context->reader = nil;
    if (context->decompression_session != NULL) {
        VTDecompressionSessionWaitForAsynchronousFrames(context->decompression_session);
        VTDecompressionSessionInvalidate(context->decompression_session);
        CFRelease(context->decompression_session);
        context->decompression_session = NULL;
    }
    if (context->format_description != NULL) {
        CFRelease(context->format_description);
        context->format_description = NULL;
    }
    os_unfair_lock_lock(&context->decoder_lock);
    if (context->decoded_frame != NULL) {
        CVPixelBufferRelease(context->decoded_frame);
        context->decoded_frame = NULL;
    }
    context->decoded_pts = kCMTimeInvalid;
    os_unfair_lock_unlock(&context->decoder_lock);
    context->reader_eof = NO;
}

static void d2s_media_decode_output(
    void *refcon,
    void *source_frame_refcon,
    OSStatus status,
    VTDecodeInfoFlags info_flags,
    CVImageBufferRef image_buffer,
    CMTime presentation_time_stamp,
    CMTime presentation_duration) {
    (void)source_frame_refcon;
    (void)info_flags;
    (void)presentation_duration;
    D2SMediaPlayer *context = (D2SMediaPlayer *)refcon;
    if (context == NULL || status != noErr || image_buffer == NULL) return;
    CVPixelBufferRef pixel_buffer = (CVPixelBufferRef)image_buffer;
    CVPixelBufferRetain(pixel_buffer);
    os_unfair_lock_lock(&context->decoder_lock);
    if (context->decoded_frame != NULL) {
        CVPixelBufferRelease(context->decoded_frame);
    }
    context->decoded_frame = pixel_buffer;
    context->decoded_pts = presentation_time_stamp;
    os_unfair_lock_unlock(&context->decoder_lock);
}

static CMVideoFormatDescriptionRef d2s_normalize_hevc_format(
    CMVideoFormatDescriptionRef source) {
    if (source == NULL || CMFormatDescriptionGetMediaSubType(source) != 'hev1') {
        return source == NULL ? NULL : (CMVideoFormatDescriptionRef)CFRetain(source);
    }
    size_t parameter_count = 0;
    int nal_length = 0;
    OSStatus status = CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
        source, 0, NULL, NULL, &parameter_count, &nal_length);
    if (status != noErr || parameter_count < 3 || parameter_count > 16) return NULL;
    const uint8_t *parameter_sets[16] = {0};
    size_t parameter_sizes[16] = {0};
    for (size_t index = 0; index < parameter_count; ++index) {
        status = CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
            source,
            index,
            &parameter_sets[index],
            &parameter_sizes[index],
            NULL,
            NULL);
        if (status != noErr) return NULL;
    }
    CMVideoFormatDescriptionRef normalized = NULL;
    status = CMVideoFormatDescriptionCreateFromHEVCParameterSets(
        kCFAllocatorDefault,
        parameter_count,
        parameter_sets,
        parameter_sizes,
        nal_length,
        CMFormatDescriptionGetExtensions(source),
        &normalized);
    return status == noErr ? normalized : NULL;
}

static BOOL d2s_media_decode_sample(D2SMediaPlayer *context, CMSampleBufferRef sample) {
    if (context == NULL || sample == NULL) return NO;
    CMVideoFormatDescriptionRef format = CMSampleBufferGetFormatDescription(sample);
    if (format == NULL) {
        return YES;
    }
    if (context->decompression_session == NULL) {
        CMVideoFormatDescriptionRef decoder_format = d2s_normalize_hevc_format(format);
        if (decoder_format == NULL) {
            d2s_media_set_error(context, @"HEVC format description could not be normalized");
            return NO;
        }
        context->format_description = decoder_format;
        NSDictionary *attributes = @{
            (id)kCVPixelBufferPixelFormatTypeKey:
                @(context->hdr ? kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
                               : kCVPixelFormatType_32BGRA),
            (id)kCVPixelBufferMetalCompatibilityKey: @YES,
            (id)kCVPixelBufferIOSurfacePropertiesKey: @{},
        };
        VTDecompressionOutputCallbackRecord callback_record = {
            d2s_media_decode_output,
            context,
        };
        NSDictionary *decoder_specification = @{
            (__bridge id)kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder:
                @YES,
        };
        OSStatus create_status = VTDecompressionSessionCreate(
            kCFAllocatorDefault,
            decoder_format,
            (__bridge CFDictionaryRef)decoder_specification,
            (__bridge CFDictionaryRef)attributes,
            &callback_record,
            &context->decompression_session);
        if (create_status != noErr || context->decompression_session == NULL) {
            FourCharCode subtype = CMFormatDescriptionGetMediaSubType(format);
            NSString *message = [NSString stringWithFormat:
                @"VideoToolbox hardware decoder creation failed (status=%d subtype=%c%c%c%c)",
                (int)create_status,
                (int)((subtype >> 24) & 0xff),
                (int)((subtype >> 16) & 0xff),
                (int)((subtype >> 8) & 0xff),
                (int)(subtype & 0xff)];
            d2s_media_set_error(context, message);
            return NO;
        }
        CFTypeRef hardware_decode = NULL;
        OSStatus hardware_status = VTSessionCopyProperty(
            (VTSessionRef)context->decompression_session,
            kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder,
            kCFAllocatorDefault,
            &hardware_decode);
        const BOOL hardware_verified = hardware_status == noErr &&
            hardware_decode != NULL &&
            CFGetTypeID(hardware_decode) == CFBooleanGetTypeID() &&
            CFBooleanGetValue((CFBooleanRef)hardware_decode);
        if (hardware_decode != NULL) CFRelease(hardware_decode);
        if (!hardware_verified) {
            VTDecompressionSessionInvalidate(context->decompression_session);
            CFRelease(context->decompression_session);
            context->decompression_session = NULL;
            d2s_media_set_error(context, @"VideoToolbox did not confirm hardware decoding");
            return NO;
        }
    }
    CMSampleBufferRef decode_sample = sample;
    if (CMFormatDescriptionGetMediaSubType(format) == 'hev1' &&
        context->format_description != format) {
        CMBlockBufferRef data_buffer = CMSampleBufferGetDataBuffer(sample);
        CMSampleTimingInfo timing = {0};
        if (data_buffer == NULL || CMSampleBufferGetSampleTimingInfo(sample, 0, &timing) != noErr ||
            CMSampleBufferCreateReady(
                kCFAllocatorDefault,
                data_buffer,
                context->format_description,
                1,
                1,
                &timing,
                0,
                NULL,
                &decode_sample) != noErr) {
            d2s_media_set_error(context, @"HEVC sample format normalization failed");
            return NO;
        }
    }
    OSStatus decode_status = VTDecompressionSessionDecodeFrame(
        context->decompression_session,
        decode_sample,
        0,
        NULL,
        NULL);
    if (decode_sample != sample) CFRelease(decode_sample);
    if (decode_status != noErr) {
        d2s_media_set_error(
            context,
            [NSString stringWithFormat:@"VideoToolbox hardware decode failed (status=%d)",
                                       (int)decode_status]);
        return NO;
    }
    return YES;
}

static BOOL d2s_media_start_reader(D2SMediaPlayer *context, CMTime start_time) {
    if (context == NULL || context->asset == nil) return NO;
    d2s_media_reset_reader(context);
    AVAssetTrack *track = [[context->asset tracksWithMediaType:AVMediaTypeVideo] firstObject];
    if (track == nil) {
        d2s_media_set_error(context, @"AVFoundation asset has no video track");
        return NO;
    }
    // Keep compressed samples unmodified, then let VideoToolbox produce the
    // destination IOSurface. No CMSampleBuffer bytes are mapped into Python.
    AVAssetReaderTrackOutput *output =
        [[AVAssetReaderTrackOutput alloc] initWithTrack:track outputSettings:nil];
    if (output == nil) {
        d2s_media_set_error(context, @"AVFoundation video reader output creation failed");
        return NO;
    }
    output.alwaysCopiesSampleData = NO;
    NSError *reader_error = nil;
    AVAssetReader *reader = [[AVAssetReader alloc] initWithAsset:context->asset
                                                           error:&reader_error];
    if (reader == nil || reader_error != nil || ![reader canAddOutput:output]) {
        d2s_media_set_error(
            context,
            reader_error.localizedDescription ?: @"AVFoundation video reader creation failed");
        return NO;
    }
    CMTime duration = context->item.duration;
    if (CMTIME_IS_NUMERIC(duration) && CMTIME_IS_NUMERIC(start_time)) {
        CMTime remaining = CMTimeSubtract(duration, start_time);
        if (CMTimeCompare(remaining, kCMTimeZero) > 0) {
            reader.timeRange = CMTimeRangeMake(start_time, remaining);
        }
    }
    [reader addOutput:output];
    if (![reader startReading]) {
        d2s_media_set_error(
            context,
            reader.error.localizedDescription ?: @"AVFoundation video reader failed to start");
        return NO;
    }
    context->reader = reader;
    context->reader_output = output;
    return YES;
}

static CMSampleBufferRef d2s_media_next_reader_sample(D2SMediaPlayer *context) {
    if (context == NULL || context->reader_output == nil || context->reader_eof) return NULL;
    CMSampleBufferRef sample = [context->reader_output copyNextSampleBuffer];
    if (sample == NULL) {
        context->reader_eof = YES;
        if (context->reader.status == AVAssetReaderStatusFailed) {
            d2s_media_set_error(
                context,
                context->reader.error.localizedDescription ?: @"AVFoundation video reader failed");
        }
    }
    return sample;
}

void *d2s_media_player_create(const char *path, int32_t hdr,
                              int32_t audio_enabled, char *error_buffer,
                              size_t error_capacity) {
    @autoreleasepool {
        if (path == NULL || path[0] == '\0') {
            d2s_media_set_create_error(error_buffer, error_capacity,
                                       @"media path is empty");
            return NULL;
        }
        NSString *path_string = [NSString stringWithUTF8String:path];
        if (path_string == nil) {
            d2s_media_set_create_error(error_buffer, error_capacity,
                                       @"media path is not valid UTF-8");
            return NULL;
        }
        NSURL *url = [NSURL fileURLWithPath:path_string];
        AVPlayerItemVideoOutput *video_output =
            [[AVPlayerItemVideoOutput alloc] initWithPixelBufferAttributes:nil];
        if (video_output == nil) {
            d2s_media_set_create_error(error_buffer, error_capacity,
                                       @"AVPlayerItemVideoOutput creation failed");
            return NULL;
        }
        video_output.suppressesPlayerRendering = YES;

        D2SMediaPlayer *context =
            (D2SMediaPlayer *)calloc(1, sizeof(D2SMediaPlayer));
        if (context == NULL) {
            d2s_media_set_create_error(error_buffer, error_capacity,
                                       @"AVPlayer allocation failed");
            return NULL;
        }
        AVURLAsset *asset = [AVURLAsset URLAssetWithURL:url options:nil];
        dispatch_semaphore_t asset_loaded = dispatch_semaphore_create(0);
        [asset loadValuesAsynchronouslyForKeys:@[@"tracks", @"duration"]
                             completionHandler:^{
            dispatch_semaphore_signal(asset_loaded);
        }];
        long load_result = dispatch_semaphore_wait(
            asset_loaded,
            dispatch_time(DISPATCH_TIME_NOW, (int64_t)(15.0 * NSEC_PER_SEC)));
        if (load_result != 0) {
            free(context);
            d2s_media_set_create_error(error_buffer, error_capacity,
                                       @"AVFoundation metadata load timed out");
            return NULL;
        }
        NSError *asset_error = nil;
        if ([asset statusOfValueForKey:@"tracks" error:&asset_error] != AVKeyValueStatusLoaded) {
            free(context);
            d2s_media_set_create_error(
                error_buffer, error_capacity,
                asset_error.localizedDescription ?: @"AVFoundation could not load video tracks");
            return NULL;
        }

        context->hdr = hdr != 0;
        context->asset = asset;
        context->video_output = video_output;
        context->output_delegate = [[D2SMediaOutputDelegate alloc] init];
        context->output_queue = dispatch_queue_create(
            "com.desktop2stereo.avplayer.video-output", DISPATCH_QUEUE_SERIAL);
        [context->video_output setDelegate:context->output_delegate
                                      queue:context->output_queue];
        context->item = [AVPlayerItem playerItemWithAsset:asset];
        for (AVPlayerItemTrack *item_track in context->item.tracks) {
            if ([item_track.assetTrack.mediaType isEqualToString:AVMediaTypeVideo]) {
                item_track.enabled = NO;
            }
        }
        [context->item addOutput:context->video_output];
        [context->video_output requestNotificationOfMediaDataChangeWithAdvanceInterval:0.03];
        context->player = [AVPlayer playerWithPlayerItem:context->item];
        context->player.automaticallyWaitsToMinimizeStalling = NO;
        context->player.allowsExternalPlayback = NO;
        context->player.muted = audio_enabled == 0;
        context->player.volume = audio_enabled == 0 ? 0.0f : 1.0f;
        if (!d2s_media_start_reader(context, kCMTimeZero)) {
            d2s_media_set_create_error(
                error_buffer,
                error_capacity,
                [NSString stringWithUTF8String:context->error] ?:
                    @"AVFoundation video reader failed to start");
            [context->player pause];
            context->player = nil;
            context->item = nil;
            context->asset = nil;
            context->video_output = nil;
            context->output_delegate = nil;
            context->output_queue = nil;
            free(context);
            return NULL;
        }
        [context->player playImmediatelyAtRate:1.0f];
        return context;
    }
}

int32_t d2s_media_player_copy_frame(void *handle, void **pixel_buffer,
                                    double *presentation_time) {
    @autoreleasepool {
        D2SMediaPlayer *context = (D2SMediaPlayer *)handle;
        if (context == NULL || pixel_buffer == NULL || presentation_time == NULL) {
            return -1;
        }
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.001, true);
        *pixel_buffer = NULL;
        *presentation_time = 0.0;
        AVPlayerItemStatus status = context->item.status;
        if (status == AVPlayerItemStatusFailed) {
            d2s_media_set_error(
                context, context->item.error.localizedDescription ?: @"AVPlayerItem failed");
            return -1;
        }
        if (status != AVPlayerItemStatusReadyToPlay) return 0;
        if (context->reader_output != nil) {
            CMTime current_time = context->player.currentTime;
            if (!CMTIME_IS_NUMERIC(current_time)) return 0;
            double current_seconds = CMTimeGetSeconds(current_time);
            for (;;) {
                os_unfair_lock_lock(&context->decoder_lock);
                CVPixelBufferRef decoded_frame = context->decoded_frame;
                CMTime decoded_pts = context->decoded_pts;
                if (decoded_frame != NULL) CVPixelBufferRetain(decoded_frame);
                os_unfair_lock_unlock(&context->decoder_lock);
                if (decoded_frame != NULL && CMTIME_IS_NUMERIC(decoded_pts)) {
                    double decoded_seconds = CMTimeGetSeconds(decoded_pts);
                    if (decoded_seconds > current_seconds + 0.05) {
                        CVPixelBufferRelease(decoded_frame);
                        return 0;
                    }
                    os_unfair_lock_lock(&context->decoder_lock);
                    if (context->decoded_frame == decoded_frame) {
                        CVPixelBufferRelease(context->decoded_frame);
                        context->decoded_frame = NULL;
                    }
                    os_unfair_lock_unlock(&context->decoder_lock);
                    if (context->has_last_frame_pts &&
                        decoded_seconds <= context->last_frame_pts + 1.0e-6) {
                        CVPixelBufferRelease(decoded_frame);
                        continue;
                    }
                    OSType pixel_format = CVPixelBufferGetPixelFormatType(decoded_frame);
                    const BOOL valid_format = context->hdr
                        ? d2s_is_p010(pixel_format)
                        : pixel_format == kCVPixelFormatType_32BGRA;
                    if (!valid_format) {
                        CVPixelBufferRelease(decoded_frame);
                        d2s_media_set_error(
                            context,
                            context->hdr
                                ? @"VideoToolbox did not provide a 10-bit P010 IOSurface for HDR media"
                                : @"VideoToolbox did not provide a BGRA IOSurface for SDR media");
                        return -2;
                    }
                    context->has_last_frame_pts = YES;
                    context->last_frame_pts = decoded_seconds;
                    *presentation_time = decoded_seconds;
                    *pixel_buffer = decoded_frame;
                    return 1;
                }
                CMSampleBufferRef sample = d2s_media_next_reader_sample(context);
                if (sample == NULL) {
                    if (context->reader.status == AVAssetReaderStatusFailed) return -1;
                    return 0;
                }
                if (!d2s_media_decode_sample(context, sample)) {
                    CFRelease(sample);
                    return -1;
                }
                CFRelease(sample);
                /* Decode synchronously; the callback has retained the IOSurface. */
                os_unfair_lock_lock(&context->decoder_lock);
                BOOL has_decoded = context->decoded_frame != NULL;
                os_unfair_lock_unlock(&context->decoder_lock);
                if (!has_decoded) continue;
            }
        }
        CMTime host_time = CMClockGetTime(CMClockGetHostTimeClock());
        double host_seconds = CMTimeGetSeconds(host_time);
        CMTime item_time = [context->video_output itemTimeForHostTime:host_seconds];
        if (!CMTIME_IS_NUMERIC(item_time)) item_time = context->player.currentTime;
        if (!CMTIME_IS_NUMERIC(item_time)) return 0;

        CMTime display_time = kCMTimeInvalid;
        CVPixelBufferRef frame = [context->video_output
            copyPixelBufferForItemTime:item_time
            itemTimeForDisplay:&display_time];
        if (frame == NULL) return 0;
        OSType pixel_format = CVPixelBufferGetPixelFormatType(frame);
        const BOOL valid_format = context->hdr
            ? d2s_is_p010(pixel_format)
            : pixel_format == kCVPixelFormatType_32BGRA;
        if (!valid_format) {
            CVPixelBufferRelease(frame);
            d2s_media_set_error(
                context,
                context->hdr
                    ? @"AVPlayer did not provide a 10-bit P010 IOSurface for HDR media"
                    : @"AVPlayer did not provide a BGRA IOSurface for SDR media");
            return -2;
        }
        if (!CMTIME_IS_NUMERIC(display_time)) display_time = item_time;
        double pts = CMTimeGetSeconds(display_time);
        if (context->has_last_frame_pts && pts <= context->last_frame_pts + 1.0e-6) {
            CVPixelBufferRelease(frame);
            return 0;
        }
        context->has_last_frame_pts = YES;
        context->last_frame_pts = pts;
        *presentation_time = pts;
        *pixel_buffer = frame;
        return 1;
    }
}

void d2s_media_player_release_frame(void *pixel_buffer) {
    if (pixel_buffer != NULL) CVPixelBufferRelease((CVPixelBufferRef)pixel_buffer);
}

int32_t d2s_media_player_set_playing(void *handle, int32_t playing) {
    @autoreleasepool {
        D2SMediaPlayer *context = (D2SMediaPlayer *)handle;
        if (context == NULL || context->player == nil) return -1;
        if (playing) [context->player playImmediatelyAtRate:1.0f];
        else [context->player pause];
        return 0;
    }
}

int32_t d2s_media_player_seek(void *handle, double position) {
    @autoreleasepool {
        D2SMediaPlayer *context = (D2SMediaPlayer *)handle;
        if (context == NULL || context->player == nil || !isfinite(position)) return -1;
        [context->item cancelPendingSeeks];
        context->has_last_frame_pts = NO;
        CMTime target = CMTimeMakeWithSeconds(MAX(0.0, position), 600);
        if (!d2s_media_start_reader(context, target)) return -1;
        [context->player seekToTime:target
                    toleranceBefore:kCMTimeZero
                     toleranceAfter:kCMTimeZero];
        return 0;
    }
}

int32_t d2s_media_player_set_audio(void *handle, double volume, int32_t muted) {
    @autoreleasepool {
        D2SMediaPlayer *context = (D2SMediaPlayer *)handle;
        if (context == NULL || context->player == nil || !isfinite(volume)) return -1;
        context->player.volume = (float)fmin(1.0, fmax(0.0, volume));
        context->player.muted = muted != 0;
        return 0;
    }
}

double d2s_media_player_position(void *handle) {
    D2SMediaPlayer *context = (D2SMediaPlayer *)handle;
    if (context == NULL || context->player == nil) return 0.0;
    CMTime position = context->player.currentTime;
    return CMTIME_IS_NUMERIC(position) ? CMTimeGetSeconds(position) : 0.0;
}

double d2s_media_player_duration(void *handle) {
    D2SMediaPlayer *context = (D2SMediaPlayer *)handle;
    if (context == NULL || context->item == nil) return 0.0;
    CMTime duration = context->item.duration;
    return CMTIME_IS_NUMERIC(duration) ? CMTimeGetSeconds(duration) : 0.0;
}

int32_t d2s_media_player_status(void *handle) {
    D2SMediaPlayer *context = (D2SMediaPlayer *)handle;
    return context == NULL || context->item == nil
        ? -1 : (int32_t)context->item.status;
}

const char *d2s_media_player_last_error(void *handle) {
    D2SMediaPlayer *context = (D2SMediaPlayer *)handle;
    return context == NULL ? "AVPlayer is not initialized" : context->error;
}

void d2s_media_player_destroy(void *handle) {
    @autoreleasepool {
        D2SMediaPlayer *context = (D2SMediaPlayer *)handle;
        if (context == NULL) return;
        [context->player pause];
        d2s_media_reset_reader(context);
        [context->item removeOutput:context->video_output];
        context->player = nil;
        context->item = nil;
        context->asset = nil;
        context->video_output = nil;
        context->output_delegate = nil;
        context->output_queue = nil;
        free(context);
    }
}
