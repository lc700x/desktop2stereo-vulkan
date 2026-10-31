from __future__ import annotations

from unittest.mock import patch

from streaming.moonlight_output import MoonlightSbsOutput, read_sunshine_base_port


class _OccupiedSocket:
    def __init__(self, occupied_ports):
        self.occupied_ports = occupied_ports

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def bind(self, address):
        if address[1] in self.occupied_ports:
            raise OSError("address already in use")


def test_moonlight_selects_alternate_port_family_when_default_is_busy():
    with patch(
        "streaming.moonlight_output.socket.socket",
        side_effect=lambda _family, _socket_type: _OccupiedSocket({47984}),
    ):
        assert MoonlightSbsOutput._select_sunshine_port(47989) == 48089


def test_sunshine_service_config_keeps_custom_port_and_host_name(tmp_path):
    output = MoonlightSbsOutput(
        base_dir=tmp_path,
        fps=60,
        display_mode="Half-SBS",
        fit_mode="contain",
        input_size=(1920, 1080),
        input_bounds=(0, 0, 1920, 1080, 1920, 1080),
        device="cuda:0",
    )
    output._state_dir = tmp_path
    output._sunshine_config = tmp_path / "sunshine.conf"
    output._sunshine_apps = tmp_path / "apps.json"
    output._sunshine_credentials = tmp_path / "web_credentials"
    output._sunshine_account_file = tmp_path / "web_account.json"
    output._sunshine_state = tmp_path / "sunshine_state.json"
    output._sunshine_pki_dir = tmp_path / "credentials"
    output._sunshine_private_key = output._sunshine_pki_dir / "cakey.pem"
    output._sunshine_certificate = output._sunshine_pki_dir / "cacert.pem"
    output._sunshine_config.write_text(
        "port = 47989\nupnp = disabled\nsunshine_name = Old Host\n",
        encoding="utf-8",
    )

    output._ensure_service_files(48089)

    config = output._sunshine_config.read_text(encoding="utf-8")
    assert "port = 48089" in config
    assert "sunshine_name = D2S Sunshine" in config
    assert "upnp = disabled" in config
    assert read_sunshine_base_port(output._sunshine_config) == 48089
    output._sunshine_port = 48089
    assert output.pairing_url == "https://localhost:48090"


def test_sunshine_service_setup_preserves_paired_client_state(tmp_path):
    output = MoonlightSbsOutput(
        base_dir=tmp_path,
        fps=60,
        display_mode="Half-SBS",
        fit_mode="contain",
        input_size=(1920, 1080),
        input_bounds=(0, 0, 1920, 1080, 1920, 1080),
        device="cuda:0",
    )
    output._state_dir = tmp_path
    output._sunshine_config = tmp_path / "sunshine.conf"
    output._sunshine_apps = tmp_path / "apps.json"
    output._sunshine_state = tmp_path / "sunshine_state.json"
    output._sunshine_pki_dir = tmp_path / "credentials"
    output._sunshine_private_key = output._sunshine_pki_dir / "cakey.pem"
    output._sunshine_certificate = output._sunshine_pki_dir / "cacert.pem"
    saved_state = '{"root":{"uniqueid":"stable-host-id","named_devices":[]}}\n'
    output._sunshine_state.write_text(saved_state, encoding="utf-8")

    output._ensure_service_files(48089)

    assert output._sunshine_state.read_text(encoding="utf-8") == saved_state


def test_sunshine_pairing_targets_selected_request_when_multiple_are_waiting(tmp_path):
    output = MoonlightSbsOutput(
        base_dir=tmp_path,
        fps=60,
        display_mode="Half-SBS",
        fit_mode="contain",
        input_size=(1920, 1080),
        input_bounds=(0, 0, 1920, 1080, 1920, 1080),
        device="cuda:0",
    )
    pairings = [
        {"id": "client-one", "name": "Living room"},
        {"id": "client-two", "name": "Bedroom"},
    ]
    with patch.object(
        output,
        "_sunshine_api_request",
        side_effect=[{"pairings": pairings}, {"status": True}],
    ) as request:
        output._pair_sunshine_client("1234", "client-two")

    assert request.call_args_list[1].args[1]["pairing_id"] == "client-two"
    assert request.call_args_list[1].args[1]["name"] == "Bedroom"
