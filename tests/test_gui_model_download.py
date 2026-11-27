import os
from pathlib import Path

from gui.handlers import GUIHandlerMixin
from gui.model_download import download_model_to_gui_cache


def test_depthpro2_has_no_in_app_research_gate():
    assert not hasattr(GUIHandlerMixin, "_confirm_research_model_if_needed")


def test_gui_model_download_uses_runtime_cache_and_scopes_token(monkeypatch, tmp_path: Path):
    observed = {}

    def fake_ensure_model_downloaded(spec, *, cache_dir, force_download):
        observed["model_id"] = spec.model_id
        observed["cache_dir"] = Path(cache_dir)
        observed["force_download"] = force_download
        observed["token"] = os.environ.get("HF_TOKEN")
        return Path(cache_dir)

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        "gui.model_download.ensure_model_downloaded",
        fake_ensure_model_downloaded,
    )

    result = download_model_to_gui_cache(
        "DepthPro2-S",
        cache_dir=tmp_path,
        token="hf_test_token",
    )

    assert result == tmp_path
    assert observed == {
        "model_id": "lc700x/DepthPro2",
        "cache_dir": tmp_path,
        "force_download": False,
        "token": "hf_test_token",
    }
    assert "HF_TOKEN" not in os.environ


def test_gui_model_download_restores_existing_token(monkeypatch, tmp_path: Path):
    def fake_ensure_model_downloaded(spec, *, cache_dir, force_download):
        assert os.environ["HF_TOKEN"] == "hf_session_token"
        return Path(cache_dir)

    monkeypatch.setenv("HF_TOKEN", "hf_existing_token")
    monkeypatch.setattr(
        "gui.model_download.ensure_model_downloaded",
        fake_ensure_model_downloaded,
    )

    download_model_to_gui_cache(
        "DepthPro2-S",
        cache_dir=tmp_path,
        token="hf_session_token",
    )

    assert os.environ["HF_TOKEN"] == "hf_existing_token"


def test_gui_detects_hugging_face_authorization_errors():
    assert GUIHandlerMixin._looks_like_hf_authorization_error(
        RuntimeError("401 Client Error: gated repository; access token required")
    )
    assert not GUIHandlerMixin._looks_like_hf_authorization_error(
        RuntimeError("connection timed out")
    )
