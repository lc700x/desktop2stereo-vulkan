#ifndef D2S_MACOS_MEDIA_PLAYER_H
#define D2S_MACOS_MEDIA_PLAYER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void *d2s_media_player_create(const char *path, int32_t hdr, int32_t audio_enabled,
                              char *error_buffer, size_t error_capacity);
int32_t d2s_media_player_copy_frame(void *handle, void **pixel_buffer,
                                    double *presentation_time);
void d2s_media_player_release_frame(void *pixel_buffer);
int32_t d2s_media_player_set_playing(void *handle, int32_t playing);
int32_t d2s_media_player_seek(void *handle, double position);
int32_t d2s_media_player_set_audio(void *handle, double volume, int32_t muted);
double d2s_media_player_position(void *handle);
double d2s_media_player_duration(void *handle);
int32_t d2s_media_player_status(void *handle);
const char *d2s_media_player_last_error(void *handle);
void d2s_media_player_destroy(void *handle);

#ifdef __cplusplus
}
#endif

#endif
