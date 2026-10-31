from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import flet as ft
from gui.handlers import GUIHandlerMixin
from gui.process import GUIProcessMixin
from gui.localization import UI_MESSAGES


class _FakePage:
    def show_dialog(self, dialog):
        self.dialog = dialog

    def pop_dialog(self):
        self.dialog = None


class _FakeProcess:
    returncode = None


def _gui_harness():
    gui = object.__new__(GUIHandlerMixin)
    gui.locale = "EN"
    gui.page = _FakePage()
    gui.moonlight_pin_tf = ft.TextField(
        label="Pairing code",
        max_length=4,
        autofocus=True,
    )
    gui._closed = False
    gui._moonlight_pair_dialog_open = False
    gui._moonlight_selected_pairing_id = None
    return gui


def test_pairing_dialog_identifies_waiting_moonlight_client():
    gui = _gui_harness()
    pairing = {
        "id": "client-123",
        "name": "Living room",
        "address": "192.168.1.25",
    }

    gui.open_moonlight_pairing_pin_dialog(pairing=pairing)

    assert gui._moonlight_pair_dialog_open is True
    assert gui.moonlight_pin_tf.on_submit == gui.submit_moonlight_pairing
    assert gui._moonlight_selected_pairing_id == "client-123"
    assert gui._moonlight_pairing_device_text.value == (
        "Pairing request from: Living room (192.168.1.25)"
    )
    assert gui.page.dialog.modal is True


def test_enter_submits_pairing_pin_once(tmp_path):
    gui = _gui_harness()
    request_path = tmp_path / "pairing-request.json"
    gui.moonlight_pin_tf.value = "1234"
    gui._moonlight_pair_dialog_open = True
    gui._moonlight_pair_submission_pending = False
    gui._moonlight_pairing_request_path = str(request_path)
    gui.process = _FakeProcess()
    gui.set_status = lambda *_args, **_kwargs: None
    gui._safe_update = lambda *_controls: None

    event = SimpleNamespace(key="Enter")
    GUIProcessMixin._on_key(gui, event)
    first_request = request_path.read_text(encoding="utf-8")
    GUIProcessMixin._on_key(gui, event)

    assert '"pin": "1234"' in first_request
    assert request_path.read_text(encoding="utf-8") == first_request
    assert gui.page.dialog is None


def test_waiting_pairing_request_automatically_opens_dialog():
    gui = _gui_harness()
    process = _FakeProcess()
    opened = []
    pairing = {"id": "client-456", "name": "Bedroom"}

    def open_dialog(*, pairing):
        opened.append(pairing)
        process.returncode = 0

    gui.open_moonlight_pairing_pin_dialog = open_dialog
    with patch(
        "streaming.moonlight_output.request_sunshine_api",
        return_value={"pairings": [pairing]},
    ):
        asyncio.run(gui._watch_moonlight_pairing_requests(process))

    assert opened == [pairing]


def test_moonlight_buttons_require_a_running_moonlight_process():
    gui = object.__new__(GUIHandlerMixin)
    gui.run_mode_key = "Moonlight Streamer"
    gui.process = None
    gui.moonlight_pairing_btn = ft.Button(disabled=True)
    gui.moonlight_pair_btn = ft.Button(disabled=True)

    gui._sync_moonlight_button_state()
    assert gui.moonlight_pairing_btn.disabled is True
    assert gui.moonlight_pair_btn.disabled is True

    gui.process = _FakeProcess()
    gui._sync_moonlight_button_state()
    assert gui.moonlight_pairing_btn.disabled is False
    assert gui.moonlight_pair_btn.disabled is False

    gui.run_mode_key = "RTMP Streamer"
    gui._sync_moonlight_button_state()
    assert gui.moonlight_pairing_btn.disabled is True
    assert gui.moonlight_pair_btn.disabled is True

    gui.run_mode_key = "Moonlight Streamer"
    gui.process.returncode = 0
    gui._sync_moonlight_button_state()
    assert gui.moonlight_pairing_btn.disabled is True
    assert gui.moonlight_pair_btn.disabled is True


def test_macos_moonlight_setup_help_is_a_mode_specific_tooltip():
    gui = object.__new__(GUIHandlerMixin)
    gui.locale = "EN"
    gui.run_mode_key = "Moonlight Streamer"
    gui.moonlight_macos_help = ft.IconButton(icon=ft.Icons.INFO_OUTLINE)

    with patch("gui.handlers.OS_NAME", "Darwin"):
        gui._sync_moonlight_macos_help()

    assert gui.moonlight_macos_help.visible is True
    assert gui.moonlight_macos_help.tooltip == UI_MESSAGES["EN"][
        "tooltip_moonlight_macos_setup"
    ]

    gui.locale = "CN"
    with patch("gui.handlers.OS_NAME", "Darwin"):
        gui._sync_moonlight_macos_help()
    assert gui.moonlight_macos_help.tooltip == UI_MESSAGES["CN"][
        "tooltip_moonlight_macos_setup"
    ]

    gui.run_mode_key = "RTMP Streamer"
    with patch("gui.handlers.OS_NAME", "Darwin"):
        gui._sync_moonlight_macos_help()
    assert gui.moonlight_macos_help.visible is False

    gui.run_mode_key = "Moonlight Streamer"
    with patch("gui.handlers.OS_NAME", "Windows"):
        gui._sync_moonlight_macos_help()
    assert gui.moonlight_macos_help.visible is False


def test_moonlight_encoder_options_match_the_selected_gpu_backend():
    gui = object.__new__(GUIHandlerMixin)
    gui.locale = "EN"
    gui.run_mode_key = "Moonlight Streamer"
    gui.device_dd = SimpleNamespace(value="CUDA 0: NVIDIA GeForce RTX 2070")
    gui.video_backend_dd = SimpleNamespace(
        options=["Auto", "Intel QSV (D3D11)", "FFmpeg", "Vulkan Video", "PyNvVideoCodec", "AMD AMF"],
        value="Vulkan Video",
    )
    gui.set_status = lambda *_args, **_kwargs: None
    gui._safe_update = lambda *_controls: None

    with patch("gui.handlers.OS_NAME", "Windows"), patch(
        "gui.handlers.devices_module.IS_ROCM", False
    ):
        gui._sync_moonlight_encoder_options()
        assert gui.video_backend_dd.options == ["Auto", "PyNvVideoCodec"]
        assert gui.video_backend_dd.value == "Auto"

        gui.video_backend_dd.value = "PyNvVideoCodec"
        gui._sync_moonlight_encoder_options()
        assert gui.video_backend_dd.value == "PyNvVideoCodec"

    gui.device_dd.value = "CUDA 0: AMD Radeon RX 9060 XT"
    with patch("gui.handlers.OS_NAME", "Windows"), patch(
        "gui.handlers.devices_module.IS_ROCM", True
    ):
        gui._sync_moonlight_encoder_options()
        assert gui.video_backend_dd.options == ["Auto", "AMD AMF"]
        assert gui.video_backend_dd.value == "Auto"

        gui.video_backend_dd.value = "AMD AMF"
        gui._sync_moonlight_encoder_options()
        assert gui.video_backend_dd.value == "AMD AMF"

    gui.run_mode_key = "RTMP Streamer"
    gui._sync_moonlight_encoder_options()
    assert "Vulkan Video" in gui.video_backend_dd.options


def test_video_encoder_label_has_chinese_translation():
    assert UI_MESSAGES["CN"]["Video Encoder:"] == "视频编码器:"


def test_moonlight_encoder_options_expose_videotoolbox_on_macos():
    gui = object.__new__(GUIHandlerMixin)
    gui.locale = "EN"
    gui.run_mode_key = "Moonlight Streamer"
    gui.device_dd = SimpleNamespace(value="MPS")
    gui.video_backend_dd = SimpleNamespace(
        options=["Auto", "FFmpeg"], value="Auto"
    )
    gui.set_status = lambda *_args, **_kwargs: None
    gui._safe_update = lambda *_controls: None

    with patch("gui.handlers.OS_NAME", "Darwin"):
        gui._sync_moonlight_encoder_options()

    assert gui.video_backend_dd.options == ["Auto", "Apple VideoToolbox"]
    assert gui.video_backend_dd.value == "Auto"

    gui.video_backend_dd.value = "Apple VideoToolbox"
    gui.run_mode_key = "RTMP Streamer"
    gui._sync_moonlight_encoder_options()
    assert "Apple VideoToolbox" not in gui.video_backend_dd.options
    assert gui.video_backend_dd.value == "Auto"


def test_sunshine_settings_manage_paired_clients(tmp_path):
    gui = _gui_harness()
    gui._show_sunshine_settings_dialog(
        tmp_path,
        47990,
        {"status": True},
        {"username": "d2s", "password": "test-password"},
        [{"name": "Living room", "uuid": "client-123", "enabled": True}],
    )
    dialog = gui.page.dialog
    dialog.update = lambda: None

    def walk_controls(control):
        yield control
        for child_name in ("content",):
            child = getattr(control, child_name, None)
            if child is not None and not isinstance(child, str):
                yield from walk_controls(child)
        for child_name in ("controls", "actions"):
            for child in getattr(control, child_name, None) or []:
                yield from walk_controls(child)

    paired_list = next(
        control
        for control in walk_controls(dialog)
        if isinstance(control, ft.ListView) and control.height == 128
    )
    device_row = paired_list.controls[0]
    enabled_switch = next(
        control for control in device_row.controls if isinstance(control, ft.Switch)
    )
    unpair_button = next(
        control
        for control in device_row.controls
        if isinstance(control, ft.TextButton)
    )
    enabled_switch.value = False
    calls = []

    def fake_api(_state_dir, method, endpoint, payload=None):
        calls.append((method, endpoint, payload))
        if endpoint == "/api/clients/list":
            return {"status": True, "named_certs": []}
        return {"status": True}

    with patch("streaming.moonlight_output.request_sunshine_api", side_effect=fake_api):
        asyncio.run(
            enabled_switch.on_change(SimpleNamespace(control=enabled_switch))
        )
        asyncio.run(unpair_button.on_click(None))

    assert calls == [
        (
            "POST",
            "/api/clients/update",
            {"uuid": "client-123", "enabled": False},
        ),
        ("GET", "/api/clients/list", None),
        ("POST", "/api/clients/unpair", {"uuid": "client-123"}),
        ("GET", "/api/clients/list", None),
    ]
