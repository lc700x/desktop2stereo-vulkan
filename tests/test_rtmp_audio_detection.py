import logging
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest


DSHOW_OUTPUT = """
[dshow @ 000001] "screen-capture-recorder" (video)
[dshow @ 000001] "Stereo Mix (Realtek(R) Audio)" (audio)
[dshow @ 000001] "virtual-audio-capturer" (audio)
[dshow @ 000001]   Alternative name "@device_cm_..."
"""

NEW_FFMPEG_DSHOW_OUTPUT = """
[in#0 @ 000001] "screen-capture-recorder" (video)
[in#0 @ 000001] "virtual-audio-capturer" (audio)
[in#0 @ 000001]   Alternative name "@device_sw_..."
"""


def test_windows_soundcard_version_supports_pinned_numpy_binary_buffers() -> None:
    if sys.platform != "win32":
        pytest.skip("reads the Windows venv's site-packages layout (Lib/)")
    requirements = (
        Path(__file__).resolve().parents[1] / "src/env_install/requirements.txt"
    ).read_text(encoding="utf-8")
    mediafoundation = (
        Path(__file__).resolve().parents[1]
        / "src/python3/Lib/site-packages/soundcard/mediafoundation.py"
    ).read_text(encoding="utf-8")

    assert "numpy==2.5.0" in requirements
    assert 'soundcard==0.4.6; platform_system == "Windows"' in requirements
    assert "numpy.frombuffer" in mediafoundation
    assert "numpy.fromstring" not in mediafoundation


def test_soundcard_loopback_lists_windows_default_speaker_first(monkeypatch):
    from streaming.wasapi_audio import query_soundcard_loopback_devices

    class Speaker:
        def __init__(self, speaker_id, name):
            self.id = speaker_id
            self.name = name

    default = Speaker("default-id", "Default Headphones")
    other = Speaker("other-id", "Other Speakers")
    fake_soundcard = types.SimpleNamespace(
        default_speaker=lambda: default,
        all_speakers=lambda: [other, default],
        get_microphone=lambda **kwargs: types.SimpleNamespace(isloopback=True),
    )
    monkeypatch.setitem(sys.modules, "soundcard", fake_soundcard)

    assert query_soundcard_loopback_devices() == [
        "Default Headphones",
        "Other Speakers",
    ]


def _target():
    from gui.handlers import GUIHandlerMixin

    return object.__new__(GUIHandlerMixin)


def test_parse_ffmpeg_dshow_audio_devices() -> None:
    from streaming.audio import parse_ffmpeg_dshow_audio_devices

    assert parse_ffmpeg_dshow_audio_devices(DSHOW_OUTPUT) == [
        "Stereo Mix (Realtek(R) Audio)",
        "virtual-audio-capturer",
    ]


def test_parse_new_ffmpeg_dshow_audio_devices() -> None:
    from streaming.audio import parse_ffmpeg_dshow_audio_devices

    assert parse_ffmpeg_dshow_audio_devices(NEW_FFMPEG_DSHOW_OUTPUT) == [
        "virtual-audio-capturer",
    ]


def test_query_ffmpeg_dshow_audio_devices(monkeypatch, tmp_path) -> None:
    from streaming import audio

    ffmpeg = tmp_path / "ffmpeg.exe"
    ffmpeg.write_bytes(b"")
    result = types.SimpleNamespace(stdout="", stderr=DSHOW_OUTPUT)
    monkeypatch.setattr(audio.subprocess, "run", lambda *args, **kwargs: result)

    assert audio.query_ffmpeg_dshow_audio_devices(ffmpeg) == [
        "Stereo Mix (Realtek(R) Audio)",
        "virtual-audio-capturer",
    ]


def test_windows_detection_prefers_dshow_virtual_audio(
    monkeypatch, caplog
) -> None:
    from gui import handlers

    caplog.set_level(logging.WARNING, logger="gui.handlers")
    monkeypatch.setattr(handlers, "OS_NAME", "Windows")
    monkeypatch.setattr(handlers, "query_soundcard_loopback_devices", lambda: None)
    monkeypatch.setattr(
        handlers,
        "query_ffmpeg_dshow_audio_devices",
        lambda path: ["virtual-audio-capturer"],
    )
    fake_sounddevice = types.SimpleNamespace(
        query_devices=lambda: (_ for _ in ()).throw(
            AssertionError("sounddevice fallback must not run")
        )
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    target = _target()
    target._populate_audio_generic()

    assert target.audio_devices == ["virtual-audio-capturer"]
    assert not caplog.messages


def test_windows_detection_prefers_soundcard_loopback(
    monkeypatch, caplog
) -> None:
    from gui import handlers

    caplog.set_level(logging.WARNING, logger="gui.handlers")
    monkeypatch.setattr(handlers, "OS_NAME", "Windows")
    monkeypatch.setattr(
        handlers,
        "query_soundcard_loopback_devices",
        lambda: ["Default speakers"],
    )
    monkeypatch.setattr(
        handlers,
        "query_ffmpeg_wasapi_audio_devices",
        lambda path: (_ for _ in ()).throw(
            AssertionError("FFmpeg probing must not run when SoundCard loopback works")
        ),
    )
    monkeypatch.setattr(
        handlers,
        "query_ffmpeg_dshow_audio_devices",
        lambda path: ["virtual-audio-capturer"],
    )

    target = _target()
    target._populate_audio_generic()
    target.audio_dd = types.SimpleNamespace(value=None, update=lambda: None)
    target.auto_select_stereo_mix()

    assert target.audio_devices == [
        "soundcard:Default speakers",
        "virtual-audio-capturer",
    ]
    assert target.audio_dd.value == "soundcard:Default speakers"
    assert not caplog.messages


def test_existing_audio_devices_are_reselected_after_gui_reset() -> None:
    target = _target()
    target.audio_devices = [
        "soundcard:Default speakers",
        "virtual-audio-capturer",
    ]
    target.audio_dd = types.SimpleNamespace(
        options=list(target.audio_devices),
        value="",
        update=lambda: None,
    )

    target._ensure_audio_device_selected()

    assert target.audio_dd.value == "soundcard:Default speakers"


def test_existing_valid_audio_selection_survives_stream_mode_change() -> None:
    target = _target()
    target.audio_devices = [
        "soundcard:Default speakers",
        "virtual-audio-capturer",
    ]
    target.audio_dd = types.SimpleNamespace(
        options=[],
        value="virtual-audio-capturer",
        update=lambda: None,
    )

    target._ensure_audio_device_selected()

    assert target.audio_dd.options == target.audio_devices
    assert target.audio_dd.value == "virtual-audio-capturer"


def test_windows_detection_silently_falls_back_when_dshow_has_no_loopback(
    monkeypatch, caplog
) -> None:
    from gui import handlers

    caplog.set_level(logging.WARNING, logger="gui.handlers")
    monkeypatch.setattr(handlers, "OS_NAME", "Windows")
    monkeypatch.setattr(handlers, "query_soundcard_loopback_devices", lambda: None)
    monkeypatch.setattr(
        handlers,
        "query_ffmpeg_dshow_audio_devices",
        lambda path: ["Microphone (USB Audio)"],
    )

    target = _target()
    target._populate_audio_generic()

    assert target.audio_devices == ["virtual-audio-capturer"]
    assert not caplog.messages


def test_windows_detection_falls_back_to_sounddevice(
    monkeypatch, caplog
) -> None:
    from gui import handlers

    caplog.set_level(logging.WARNING, logger="gui.handlers")
    monkeypatch.setattr(handlers, "OS_NAME", "Windows")
    monkeypatch.setattr(handlers, "query_soundcard_loopback_devices", lambda: None)
    monkeypatch.setattr(
        handlers,
        "query_ffmpeg_dshow_audio_devices",
        lambda path: None,
    )
    fake_sounddevice = types.SimpleNamespace(
        query_devices=lambda: [
            {
                "name": "Stereo Mix (Realtek(R) Audio)",
                "max_input_channels": 2,
                "max_output_channels": 0,
            }
        ]
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    target = _target()
    target._populate_audio_generic()

    assert target.audio_devices == ["Stereo Mix (Realtek(R) Audio)"]
    assert not caplog.messages


def test_soundcard_sender_resolves_speaker_to_loopback_microphone(monkeypatch) -> None:
    from streaming.wasapi_audio import SoundcardLoopbackSender

    speaker = types.SimpleNamespace(id="speaker-id", name="Default speakers")
    loopback = types.SimpleNamespace(isloopback=True, name="Default speakers loopback")
    fake_soundcard = types.SimpleNamespace(
        all_speakers=lambda: [speaker],
        default_speaker=lambda: speaker,
        get_microphone=lambda *, id, include_loopback: (
            loopback if id == speaker.id and include_loopback else None
        ),
    )
    monkeypatch.setitem(sys.modules, "soundcard", fake_soundcard)

    sender = SoundcardLoopbackSender("virtual-audio-capturer")
    try:
        assert sender._loopback is loopback
    finally:
        sender.close()


def test_soundcard_sender_stays_healthy_during_continuous_capture(monkeypatch) -> None:
    from streaming.wasapi_audio import SoundcardLoopbackSender

    class Recorder:
        calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def record(self, *, numframes):
            self.calls += 1
            time.sleep(0.001)
            return np.zeros((numframes, 2), dtype=np.float32)

    recorder = Recorder()
    speaker = types.SimpleNamespace(id="speaker-id", name="Default speakers")
    loopback = types.SimpleNamespace(
        isloopback=True,
        recorder=lambda **_kwargs: recorder,
    )
    fake_soundcard = types.SimpleNamespace(
        all_speakers=lambda: [speaker],
        default_speaker=lambda: speaker,
        get_microphone=lambda **_kwargs: loopback,
    )
    monkeypatch.setitem(sys.modules, "soundcard", fake_soundcard)

    sender = SoundcardLoopbackSender()
    try:
        sender.start()
        for _ in range(100):
            if recorder.calls >= 3:
                break
            time.sleep(0.002)
        assert recorder.calls >= 3
        assert sender._thread is not None and sender._thread.is_alive()
        assert sender._startup_error is None
        assert sender._runtime_error is None
    finally:
        sender.close()


def test_soundcard_sender_rejects_failure_before_first_pcm(monkeypatch) -> None:
    from streaming.wasapi_audio import SoundcardLoopbackSender

    class FailingRecorder:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def record(self, *, numframes):
            raise RuntimeError(f"failed before {numframes} frames")

    speaker = types.SimpleNamespace(id="speaker-id", name="Default speakers")
    loopback = types.SimpleNamespace(
        isloopback=True,
        recorder=lambda **_kwargs: FailingRecorder(),
    )
    fake_soundcard = types.SimpleNamespace(
        all_speakers=lambda: [speaker],
        default_speaker=lambda: speaker,
        get_microphone=lambda **_kwargs: loopback,
    )
    monkeypatch.setitem(sys.modules, "soundcard", fake_soundcard)

    sender = SoundcardLoopbackSender()
    try:
        with pytest.raises(RuntimeError, match="failed before 4096 frames"):
            sender.start()
        assert isinstance(sender._startup_error, RuntimeError)
        assert sender._runtime_error is None
        assert sender._stop.is_set()
    finally:
        sender.close()


def test_soundcard_sender_keeps_ffmpeg_alive_with_silence_after_capture_failure(
    monkeypatch, capsys
) -> None:
    from streaming.wasapi_audio import SoundcardLoopbackSender

    class FailingRecorder:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def record(self, *, numframes):
            if getattr(self, "sent_once", False):
                raise RuntimeError("capture endpoint changed")
            self.sent_once = True
            return np.zeros((numframes, 2), dtype=np.float32)

    speaker = types.SimpleNamespace(id="speaker-id", name="Default speakers")
    loopback = types.SimpleNamespace(
        isloopback=True,
        recorder=lambda **_kwargs: FailingRecorder(),
    )
    fake_soundcard = types.SimpleNamespace(
        all_speakers=lambda: [speaker],
        default_speaker=lambda: speaker,
        get_microphone=lambda **_kwargs: loopback,
    )
    monkeypatch.setitem(sys.modules, "soundcard", fake_soundcard)

    sender = SoundcardLoopbackSender()
    try:
        sender.start()
        for _ in range(100):
            if sender._runtime_error is not None:
                break
            time.sleep(0.005)
        assert sender._startup_error is None
        assert isinstance(sender._runtime_error, RuntimeError)
        assert sender._thread is not None and sender._thread.is_alive()
        assert not sender._stop.is_set()
        assert "continuing with silent audio" in capsys.readouterr().out
    finally:
        sender.close()


def test_darwin_persists_explicit_stereo_mix_selection(monkeypatch) -> None:
    from gui import handlers

    target = _target()
    target._config = {}
    monkeypatch.setattr(handlers, "OS_NAME", "Darwin")

    # A real device selection is persisted (v2.5.0 parity: the runtime then
    # captures the chosen device by name instead of auto-selecting).
    ev = types.SimpleNamespace(control=types.SimpleNamespace(value="Virtual Desktop Speakers"))
    handlers.GUIHandlerMixin.on_audio_device_change(target, ev)
    assert target._config.get("Stereo Mix") == "Virtual Desktop Speakers"

    # Placeholder labels are never persisted.
    for placeholder in (
        "No Stereo Mix device found",
        "sounddevice not available",
        "No audio capture devices found",
    ):
        ev = types.SimpleNamespace(control=types.SimpleNamespace(value=placeholder))
        target._config["Stereo Mix"] = "old"
        handlers.GUIHandlerMixin.on_audio_device_change(target, ev)
        assert "Stereo Mix" not in target._config


def test_non_darwin_never_persists_stereo_mix(monkeypatch) -> None:
    from gui import handlers

    target = _target()
    target._config = {}
    monkeypatch.setattr(handlers, "OS_NAME", "Windows")

    ev = types.SimpleNamespace(control=types.SimpleNamespace(value="Speakers"))
    handlers.GUIHandlerMixin.on_audio_device_change(target, ev)

    assert "Stereo Mix" not in target._config


class _FakeDropdown:
    def __init__(self):
        self.options = []
        self.value = ""
        self._updates = 0

    def update(self):
        self._updates += 1


def test_saved_stereo_mix_wins_over_first_discovered_device() -> None:
    from gui import handlers

    target = _target()
    target._config = {"Stereo Mix": "Virtual Desktop Speakers"}
    target.audio_devices = ["BlackHole 2ch", "Virtual Desktop Speakers"]

    assert handlers.GUIHandlerMixin._saved_audio_device(target) == "Virtual Desktop Speakers"


def test_saved_stereo_mix_matches_prefixed_label() -> None:
    from gui import handlers

    target = _target()
    target._config = {"Stereo Mix": "soundcard:BlackHole 2ch"}
    target.audio_devices = ["BlackHole 2ch", "Virtual Desktop Speakers"]

    assert handlers.GUIHandlerMixin._saved_audio_device(target) == "BlackHole 2ch"


def test_saved_stereo_mix_ignored_when_device_gone() -> None:
    from gui import handlers

    target = _target()
    target._config = {"Stereo Mix": "Gone Device"}
    target.audio_devices = ["BlackHole 2ch"]

    assert handlers.GUIHandlerMixin._saved_audio_device(target) is None


def test_audio_dropdown_prefers_saved_device() -> None:
    from gui import handlers

    target = _target()
    target._config = {"Stereo Mix": "Virtual Desktop Speakers"}
    target.audio_devices = ["BlackHole 2ch", "Virtual Desktop Speakers"]
    target.audio_dd = _FakeDropdown()

    handlers.GUIHandlerMixin._apply_audio_devices_to_dropdown(target)

    assert target.audio_dd.value == "Virtual Desktop Speakers"
    assert target.audio_dd.options == target.audio_devices
    assert target.audio_dd._updates >= 1


def test_auto_select_stereo_mix_keeps_saved_device() -> None:
    from gui import handlers

    target = _target()
    target._config = {"Stereo Mix": "Virtual Desktop Speakers"}
    target.audio_devices = ["BlackHole 2ch", "Virtual Desktop Speakers"]
    target.audio_dd = _FakeDropdown()

    handlers.GUIHandlerMixin.auto_select_stereo_mix(target)

    assert target.audio_dd.value == "Virtual Desktop Speakers"
