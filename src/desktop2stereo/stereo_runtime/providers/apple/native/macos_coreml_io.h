#ifndef D2S_MACOS_COREML_IO_H
#define D2S_MACOS_COREML_IO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    D2S_OUTPUT_HALF_SBS = 0,
    D2S_OUTPUT_FULL_SBS = 1,
    D2S_OUTPUT_HALF_TAB = 2,
    D2S_OUTPUT_FULL_TAB = 3,
    D2S_OUTPUT_MONO = 4,
    D2S_OUTPUT_DEPTH_MAP = 5,
    D2S_OUTPUT_ANAGLYPH = 6,
    D2S_OUTPUT_INTERLEAVED = 7,
    D2S_OUTPUT_LEIA = 8,
};

typedef struct {
    int32_t slot;
    int32_t source_width;
    int32_t source_height;
    int32_t input_width;
    int32_t input_height;
    int32_t depth_width;
    int32_t depth_height;
    int32_t input_shared;
    int32_t output_backing_used;
    int32_t output_zero_copy;
    int32_t finite_depth;
    uint32_t nonfinite_count;
    float normalize_lo;
    float normalize_hi;
    double preprocess_ms;
    double model_ms;
    double postprocess_ms;
    float raw_normalize_lo;
    float raw_normalize_hi;
    int32_t normalization_history_reset;
} D2SCoreMLIOResult;

typedef struct {
    float depth_strength;
    float max_disparity_px;
    float convergence;
    float edge_threshold;
    float fill_strength;
    int32_t fill_radius;
    int32_t mask_feather_radius;
    int32_t symmetric;
    int32_t layers;
    float softness;
    float foreground_scale;
    float midground_scale;
    float background_scale;
    int32_t edge_dilation;
    int32_t screen_edge_suppression;
    int32_t hole_fill_mode;
    int32_t occlusion_enabled;
    float depth_pop;
    float antialias_strength;
    int32_t anaglyph_method;
    int32_t edge_aa_enabled;
} D2SCoreMLIOWarpConfig;

void *d2s_coreml_io_create(const char *model_path, int32_t input_width,
                           int32_t input_height, int32_t compute_units,
                           char *error_buffer, size_t error_capacity);

int32_t d2s_coreml_io_set_smaa_luts(void *handle,
                                    const void *area, size_t area_size,
                                    const void *search, size_t search_size);

int32_t d2s_coreml_io_predict(void *handle, void *pixel_buffer,
                             uint64_t frame_id, D2SCoreMLIOResult *result);

int32_t d2s_coreml_io_pack(void *handle, int32_t slot, void *destination,
                           size_t destination_size, int32_t output_width,
                           int32_t output_height, int32_t output_format,
                           const D2SCoreMLIOWarpConfig *warp_config,
                           float eye_offset, float depth_strength,
                           float convergence, float smooth_texels);

int32_t d2s_coreml_io_pack_rgb(void *handle, int32_t slot, void *destination,
                               size_t destination_size, int32_t output_width,
                               int32_t output_height, int32_t output_format,
                               const D2SCoreMLIOWarpConfig *warp_config,
                               float eye_offset, float depth_strength,
                               float convergence, float smooth_texels);

int32_t d2s_coreml_io_release(void *handle, int32_t slot);
const char *d2s_coreml_io_last_error(void *handle);
void d2s_coreml_io_destroy(void *handle);

#ifdef __cplusplus
}
#endif

#endif
