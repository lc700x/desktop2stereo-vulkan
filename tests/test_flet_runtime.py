from __future__ import annotations

import zipfile

from gui import flet_runtime


def _write_client_archive(path, content: str) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("flet/flet.exe", content)


def test_vendored_flet_cache_tracks_archive_content(tmp_path, monkeypatch) -> None:
    packages_dir = tmp_path / "packages"
    clients_dir = tmp_path / "clients"
    packages_dir.mkdir()
    archive_path = packages_dir / "flet-windows.zip"
    _write_client_archive(archive_path, "0.85.3")

    monkeypatch.setattr(flet_runtime, "PACKAGES_DIR", packages_dir)
    monkeypatch.setattr(flet_runtime, "CLIENTS_DIR", clients_dir)
    monkeypatch.setattr(flet_runtime, "_current_artifact_name", lambda: archive_path.name)
    monkeypatch.setattr(flet_runtime, "_is_linux", lambda: False)
    monkeypatch.setattr(flet_runtime, "get_os_name", lambda: "Windows")

    view_path = flet_runtime.ensure_vendored_flet_view()
    executable = clients_dir / "flet-windows" / "flet" / "flet.exe"
    assert view_path == str(executable.parent)
    assert executable.read_text() == "0.85.3"

    _write_client_archive(archive_path, "0.86.5")
    flet_runtime.ensure_vendored_flet_view()

    assert executable.read_text() == "0.86.5"


def test_run_active_reflects_process_state() -> None:
    from gui import process as gui_process

    target = object.__new__(gui_process.GUIProcessMixin)
    target._starting = False
    target.process = None
    assert target._run_active() is False

    target._starting = True
    assert target._run_active() is True

    target._starting = False
    target.process = type("P", (), {"returncode": None})()
    assert target._run_active() is True

    target.process = type("P", (), {"returncode": 0})()
    assert target._run_active() is False


def test_esc_poll_task_darwin_starts_listener_and_exits_on_close(monkeypatch) -> None:
    import asyncio

    from gui import process as gui_process

    monkeypatch.setattr(gui_process, "OS_NAME", "Darwin")
    target = object.__new__(gui_process.GUIProcessMixin)
    target._closed = False
    target._esc_down = None
    target._esc_stopped = False
    target._starting = False
    target.process = None
    target.set_status = lambda *args, **kwargs: None

    async def run() -> None:
        task = asyncio.ensure_future(target._esc_poll_task())
        await asyncio.sleep(0.5)
        target._closed = True
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(run())
    assert target._closed is True
