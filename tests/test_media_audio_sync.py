"""Source-sample identity and video/DAC alignment; no audio/GPU hardware needed."""

import threading

import numpy as np
import pytest

from capture.media_audio import MediaAudioSync


def test_audio_does_not_start_before_first_presented_video():
    sync = MediaAudioSync(generation=1, start_pts=15.0)
    sync.enqueue(np.ones((480, 2), dtype=np.float32), threading.Event())
    output = np.empty((480, 2), dtype=np.float32)
    sync.render(output, dac_at=100.0, volume=1.0, muted=False)
    assert not output.any()
    assert sync.snapshot(100.0)["buffered_blocks"] == 1
    sync.observe_video(1, 0, 15.0, 100.0, 1.0 / 24.0)
    sync.render(output, dac_at=100.0, volume=1.0, muted=False)
    assert np.all(output == 1.0)
    assert sync.snapshot(100.0)["audio_pts"] == 15.0


def test_audio_uses_pcm_sample_count_and_dac_time_not_decoder_read_time():
    sync = MediaAudioSync(generation=1, start_pts=10.0)
    stop = threading.Event()
    for index in range(5):
        sync.enqueue(np.full((480, 2), index + 1, dtype=np.float32), stop)
    sync.observe_video(1, 10, 10.0, 100.0, 1.0 / 24.0)
    output = np.zeros((480, 2), dtype=np.float32)
    for index in range(4):
        sync.render(output, dac_at=100.0 + index * 0.01, volume=1.0, muted=False)
        assert np.all(output == index + 1)
        stats = sync.snapshot(100.0 + index * 0.01)
        assert stats["audio_pts"] == pytest.approx(10.0 + index * 0.01)
        assert stats["offset_ms"] == pytest.approx(0.0, abs=0.01)


def test_first_video_frame_aligns_audio_preroll_to_its_actual_pts():
    sync = MediaAudioSync(generation=1, start_pts=15.0)
    stop = threading.Event()
    # A ramp identifies exactly which original PCM samples reach the DAC.
    ramp = np.arange(4800, dtype=np.float32)
    for index in range(10):
        sync.enqueue(np.repeat(ramp[index * 480 : (index + 1) * 480, None], 2, axis=1), stop)
    sync.observe_video(1, 5, 15.05, 100.0, 1.0 / 24.0)
    output = np.zeros((480, 2), dtype=np.float32)
    sync.render(output, dac_at=100.0, volume=1.0, muted=False)
    assert np.array_equal(output[:, 0], ramp[2400:2880])
    assert sync.snapshot(100.0)["preroll_frames"] == 2400
    assert sync.snapshot(100.0)["audio_pts"] == pytest.approx(15.05)


def test_audio_holds_samples_when_picture_stalls_without_skipping_them():
    sync = MediaAudioSync(generation=1, start_pts=0.0)
    stop = threading.Event()
    for index in range(10):
        sync.enqueue(np.full((480, 2), index + 1, dtype=np.float32), stop)
    sync.observe_video(1, 0, 0.0, 100.0, 0.04)
    output = np.zeros((480, 2), dtype=np.float32)
    for index in range(4):
        sync.render(output, dac_at=100.0 + index * 0.01, volume=1.0, muted=False)
    sync.render(output, dac_at=100.5, volume=1.0, muted=False)
    assert not output.any()
    assert sync.snapshot(100.5)["buffered_blocks"] == 6
    sync.observe_video(1, 4, 0.04, 100.51, 0.04)
    sync.render(output, dac_at=100.51, volume=1.0, muted=False)
    assert np.all(output == 5.0)
    assert sync.snapshot(100.51)["audio_pts"] == pytest.approx(0.04)


def test_audio_rejects_seek_generation_and_repeated_frame_feedback():
    sync = MediaAudioSync(generation=2, start_pts=20.0)
    sync.observe_video(1, 100, 100.0, 100.0, 0.04)
    assert sync.snapshot(100.0)["video_pts"] is None
    sync.observe_video(2, 0, 20.0, 100.0, 0.04)
    sync.observe_video(2, 0, 20.0, 110.0, 0.04)
    assert sync.snapshot(110.0)["video_pts"] == pytest.approx(20.04)


def test_audio_mute_changes_gain_but_preserves_sample_timestamp():
    sync = MediaAudioSync(generation=1, start_pts=0.0)
    sync.enqueue(np.ones((480, 2), dtype=np.float32), threading.Event())
    sync.observe_video(1, 0, 0.0, 100.0, 0.04)
    output = np.ones((480, 2), dtype=np.float32)
    sync.render(output, dac_at=100.0, volume=0.5, muted=True)
    assert not output.any()
    assert sync.snapshot(100.0)["audio_pts"] == 0.0
    assert sync.snapshot(100.0)["buffered_blocks"] == 0


def test_audio_close_unblocks_full_decoder_queue():
    sync = MediaAudioSync(generation=1, start_pts=0.0)
    stop = threading.Event()
    for _ in range(20):
        assert sync.enqueue(np.zeros((480, 2), dtype=np.float32), stop)
    done = threading.Event()
    results = []
    worker = threading.Thread(target=lambda: (results.append(sync.enqueue(np.zeros((480, 2)), stop)), done.set()))
    worker.start()
    try:
        assert not done.wait(0.03)
        sync.close()
        assert done.wait(1.0)
        assert results == [False]
    finally:
        sync.close()
        worker.join(timeout=1.0)


def test_audio_snapshot_uses_current_dac_block_not_future_queued_audio():
    sync = MediaAudioSync(generation=1, start_pts=10.0)
    stop = threading.Event()
    for _ in range(5):
        sync.enqueue(np.ones((480, 2), dtype=np.float32), stop)
    sync.observe_video(1, 0, 10.0, 100.0, 0.04)
    output = np.zeros((480, 2), dtype=np.float32)
    sync.render(output, dac_at=100.0, callback_at=99.99, volume=1.0, muted=False)
    sync.render(output, dac_at=100.01, callback_at=100.0, volume=1.0, muted=False)
    assert sync.snapshot(100.005)["audio_pts"] == pytest.approx(10.005)
    assert sync.snapshot(99.99)["audio_pts"] is None


def test_audio_recovers_measured_lateness_without_changing_sample_rate():
    sync = MediaAudioSync(generation=1, start_pts=0.0)
    stop = threading.Event()
    for index in range(10):
        sync.enqueue(np.full((480, 2), index + 1, dtype=np.float32), stop)
    sync.observe_video(1, 0, 0.0, 100.0, 0.04)
    output = np.zeros((480, 2), dtype=np.float32)
    sync.render(output, dac_at=100.0, volume=1.0, muted=False)
    sync.observe_video(1, 5, 0.05, 100.05, 0.04)
    sync.render(output, dac_at=100.05, volume=1.0, muted=False)
    assert np.all(output == 6.0)
    assert sync.snapshot(100.05)["audio_pts"] == pytest.approx(0.05)
    assert sync.snapshot(100.05)["resync_frames"] == 1920


def test_audio_can_cross_video_tick_in_future_dac_buffer_without_repeated_silence():
    sync = MediaAudioSync(generation=1, start_pts=0.0)
    stop = threading.Event()
    for _ in range(10):
        sync.enqueue(np.ones((480, 2), dtype=np.float32), stop)
    sync.observe_video(1, 0, 0.0, 100.0, 1.0 / 24.0)
    output = np.zeros((480, 2), dtype=np.float32)
    for index in range(6):
        sync.render(output, dac_at=100.0 + index * 0.01, callback_at=99.99 + index * 0.01, volume=1.0, muted=False)
        assert np.all(output == 1.0)
    assert sync.snapshot(100.05)["held_frames"] == 0


def test_predicted_future_video_maps_to_preceding_audio_samples_until_display():
    sync = MediaAudioSync(generation=1, start_pts=10.0)
    stop = threading.Event()
    for index in range(8):
        sync.enqueue(np.full((480, 2), index + 1, dtype=np.float32), stop)
    output = np.zeros((480, 2), dtype=np.float32)
    sync.observe_video(1, 0, 10.0, 100.0, 0.04)
    for index in range(3):
        sync.render(output, dac_at=100.0 + index * 0.01, callback_at=99.99 + index * 0.01, volume=1.0, muted=False)
    # Runtime has submitted the NEXT frame 20 ms ahead of its display. It
    # must not force audio forward to that future frame's PTS prematurely.
    sync.observe_video(1, 4, 10.04, 100.04, 0.04)
    sync.render(output, dac_at=100.03, callback_at=100.02, volume=1.0, muted=False)
    assert np.all(output == 4.0)
    assert sync.snapshot(100.03)["audio_pts"] == pytest.approx(10.03)
    assert sync.snapshot(100.03)["video_pts"] == pytest.approx(10.03)
    assert sync.snapshot(100.03)["resync_frames"] == 0
    assert sync.snapshot(100.03)["held_frames"] == 0


def test_media_sample_clock_stays_continuous_under_small_video_phase_jitter():
    sync = MediaAudioSync(generation=1, start_pts=0.0)
    stop = threading.Event()
    output = np.zeros((480, 2), dtype=np.float32)
    sync.observe_video(1, 0, 0.0, 100.0, 1.0 / 24.0)
    last_index = 0
    for block in range(200):
        sync.enqueue(np.ones((480, 2), dtype=np.float32), stop)
        now = 100.0 + block * 0.01
        index = int((block * 0.01 + 0.01) * 24)
        if index > last_index:
            jitter = (0.002, -0.002, 0.0)[index % 3]
            sync.observe_video(1, index, index / 24.0, 100.0 + index / 24.0 + jitter, 1.0 / 24.0)
            last_index = index
        sync.render(output, dac_at=now, callback_at=now - 0.01, volume=1.0, muted=False)
        assert np.all(output == 1.0)
    assert sync.snapshot(101.99)["held_frames"] == 0
    assert sync.snapshot(101.99)["resync_frames"] == 0


def test_stopped_audio_clock_never_plays_pre_seek_pcm_in_new_session():
    old = MediaAudioSync(generation=1, start_pts=0.0)
    old.enqueue(np.ones((480, 2), dtype=np.float32), threading.Event())
    old.observe_video(1, 0, 0.0, 100.0, 0.04)
    old.close()
    current = MediaAudioSync(generation=2, start_pts=20.0)
    current.enqueue(np.full((480, 2), 2.0, dtype=np.float32), threading.Event())
    current.observe_video(1, 100, 100.0, 110.0, 0.04)
    output = np.ones((480, 2), dtype=np.float32)
    current.render(output, dac_at=110.0, volume=1.0, muted=False)
    assert not output.any()
    current.observe_video(2, 0, 20.0, 110.0, 0.04)
    current.render(output, dac_at=110.0, volume=1.0, muted=False)
    assert np.all(output == 2.0)
    assert current.snapshot(110.0)["audio_pts"] == 20.0


def test_audio_device_underflows_are_reported_separately_from_pcm_underruns():
    sync = MediaAudioSync(generation=1, start_pts=0.0)
    sync.note_device_underflow()
    stats = sync.snapshot(100.0)
    assert stats["device_underflows"] == 1
    assert stats["underruns"] == 0


def test_natural_audio_eof_retains_final_samples_and_distinguishes_trailing_silence():
    sync = MediaAudioSync(generation=1, start_pts=10.0)
    sync.enqueue(np.ones((240, 2), dtype=np.float32), threading.Event())
    sync.observe_video(1, 0, 10.0, 100.0, 1.0 / 30.0)
    sync.note_source_eof()
    output = np.empty((480, 2), dtype=np.float32)
    sync.render(output, dac_at=100.0, volume=1.0, muted=False)
    assert np.all(output[:240] == 1.0)
    assert not output[240:].any()
    metrics = sync.snapshot(100.0)
    assert metrics["underruns"] == 0
    assert metrics["source_eof"] is True
    assert metrics["trailing_silence_frames"] == 240


def test_empty_audio_before_decoder_eof_still_counts_as_real_underrun():
    sync = MediaAudioSync(generation=1, start_pts=10.0)
    sync.observe_video(1, 0, 10.0, 100.0, 1.0 / 30.0)
    output = np.empty((480, 2), dtype=np.float32)
    sync.render(output, dac_at=100.0, volume=1.0, muted=False)
    assert sync.snapshot(100.0)["underruns"] == 1
    assert not sync.snapshot(100.0)["source_eof"]
