"""Host frame packer gate tests (ported macOS pipeline)."""

import sys

from viewer.host_frame_packer import maybe_install_local_viewer_packer


def test_packer_gated_off_off_darwin(monkeypatch) -> None:
    if sys.platform == "darwin":
        monkeypatch.delenv("D2S_VIEWER_HOST_FRAME", raising=False)
        monkeypatch.setenv("D2S_VIEWER_HOST_FRAME", "0")
    kwargs: dict = {"runtime_q": object()}
    installed = maybe_install_local_viewer_packer(
        pipeline_q=object(),
        local_viewer_kwargs=kwargs,
        os_name="Windows" if sys.platform != "win32" else "Linux",
    )
    assert installed is False
