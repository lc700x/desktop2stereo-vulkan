from __future__ import annotations

import zipfile
from pathlib import Path

from gui import flet_runtime


def test_flet_process_tree_targets_only_client_descendants() -> None:
    processes = [
        (10, 1, "desktop2stereo.exe"),
        (11, 10, "python.exe"),
        (12, 11, "flet.exe"),
        (13, 12, "renderer.exe"),
        (20, 1, "other-app.exe"),
        (21, 20, "flet.exe"),
    ]

    assert flet_runtime._flet_process_ids(processes, 10) == (12,)


def test_main_gui_reaps_its_flet_client_when_flet_run_raises(monkeypatch) -> None:
    import os

    import pytest
    from gui import gui

    stopped = []
    monkeypatch.setattr(gui, "_setup_console_logging", lambda: None)
    monkeypatch.setattr(
        gui,
        "stop_flet_descendants",
        lambda pid: stopped.append(pid) or (),
    )

    def fail_run(*args, **kwargs):
        raise RuntimeError("simulated Flet close")

    monkeypatch.setattr(gui.ft, "run", fail_run)
    with pytest.raises(RuntimeError, match="simulated Flet close"):
        gui.main()

    assert stopped == [os.getpid()]


def test_stop_flet_descendants_terminates_only_owned_client(tmp_path) -> None:
    import os
    import shutil
    import subprocess
    import sys
    import time

    import pytest

    if os.name != "nt":
        pytest.skip("Windows Flet client process cleanup is Windows-specific")

    client_path = tmp_path / "flet.exe"
    shutil.copy2(sys.executable, client_path)
    child_env = os.environ.copy()
    child_env["PATH"] = (
        str(Path(sys.executable).parent)
        + os.pathsep
        + child_env.get("PATH", "")
    )
    client = subprocess.Popen(
        [str(client_path), "-c", "import time; time.sleep(60)"],
        env=child_env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if client.pid in {pid for pid, _, _ in flet_runtime._windows_process_snapshot()}:
                break
            time.sleep(0.05)
        else:
            pytest.fail("the test Flet client process did not start")

        assert flet_runtime.stop_flet_descendants(os.getpid(), timeout_s=2.0) == ()
        client.wait(timeout=5.0)
    finally:
        if client.poll() is None:
            client.kill()
            client.wait(timeout=2.0)


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
