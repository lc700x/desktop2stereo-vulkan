import json
import threading

import pytest

import stereo_runtime.progress as progress_module
from stereo_runtime.progress import (
    DownloadProgress,
    activity_progress,
    file_size_progress,
    progress_write,
    status_write,
    write_bytes_with_progress,
)
from tqdm.contrib.concurrent import thread_map


def test_write_bytes_with_progress_writes_file_and_reports_total(tmp_path, capsys):
    target = tmp_path / "model.trt"

    write_bytes_with_progress(target, b"trt-bytes", "Saving TensorRT engine: model.trt", chunk_size=3)

    assert target.read_bytes() == b"trt-bytes"
    out = capsys.readouterr().out
    assert "Saving TensorRT engine: model.trt" in out
    assert "[D2S_PROGRESS]" in out
    assert '"percent":100.0' in out


def test_file_size_progress_uses_file_growth_as_approximation(tmp_path, capsys):
    target = tmp_path / "model.trt"

    with file_size_progress("Building TensorRT engine: model.trt", target, total_bytes=9, interval_s=999):
        target.write_bytes(b"123456789")

    out = capsys.readouterr().out
    assert "Building TensorRT engine: model.trt" in out
    assert "[D2S_PROGRESS]" in out
    assert '"percent":100.0' in out


def test_download_progress_emits_structured_progress(capsys):
    progress = DownloadProgress(total=100, desc="model.safetensors", mininterval=0)
    progress.update(50)
    progress.close()

    out = capsys.readouterr().out
    assert "[D2S_PROGRESS]" in out
    assert '"desc":"model.safetensors"' in out
    assert '"percent":50.0' in out


def test_progress_write_keeps_long_messages_single_line(capsys):
    message = "[Main] Preparing depth model download: lc700x/InfiniDepth-Large/model.safetensors to models. First download may take several minutes."

    progress_write(message)

    assert capsys.readouterr().out == message + "\n"


def test_status_write_emits_status_marker(capsys):
    status_write("Downloading weights")

    assert capsys.readouterr().out == "[D2S_STATUS] Downloading weights\n"


def test_download_progress_exposes_tqdm_lock():
    with DownloadProgress.get_lock():
        pass


def test_download_progress_allows_tqdm_lock_replacement():
    original = DownloadProgress.get_lock()
    replacement = type(original)()
    try:
        DownloadProgress.set_lock(replacement)
        assert DownloadProgress.get_lock() is replacement
    finally:
        DownloadProgress.set_lock(original)


def test_download_progress_wraps_iterables(capsys):
    assert list(DownloadProgress([1, 2], total=2, desc="files", mininterval=0)) == [1, 2]

    out = capsys.readouterr().out
    assert '"desc":"files"' in out
    assert '"percent":100.0' in out
    assert '"unit":"steps"' in out


def test_download_progress_works_with_thread_map(capsys):
    assert thread_map(lambda value: value + 1, [1, 2], tqdm_class=DownloadProgress, desc="files") == [2, 3]

    out = capsys.readouterr().out
    assert '"desc":"files"' in out
    assert '"percent":100.0' in out
    assert '"unit":"steps"' in out


def test_activity_progress_emits_live_updates_and_stops_when_complete(monkeypatch):
    events = []
    pulsed = threading.Event()
    desc = "Building MIGraphX graph: model.mgx"

    def record(message):
        events.append(json.loads(message.removeprefix("[D2S_PROGRESS] ")))
        if len(events) >= 2 and events[-1]["state"] == "running":
            pulsed.set()

    monkeypatch.setattr(progress_module, "progress_write", record)
    with activity_progress(desc, interval_s=0.001):
        assert events[0]["desc"] == desc
        assert events[0]["percent"] is None
        assert events[0]["unit"] == "activity"
        assert pulsed.wait(timeout=5.0)

    assert events[-1]["state"] == "complete"
    assert events[-1]["percent"] == 100.0
    assert all(event["percent"] is None for event in events[:-1])
    assert not any(thread.name == f"Progress:{desc}" for thread in threading.enumerate())


def test_activity_progress_reports_failure_without_claiming_completion(capsys):
    desc = "Building MIGraphX graph: failed.mgx"
    with pytest.raises(RuntimeError, match="compile failed"):
        with activity_progress(desc, interval_s=999):
            raise RuntimeError("compile failed")

    events = [
        json.loads(line.removeprefix("[D2S_PROGRESS] "))
        for line in capsys.readouterr().out.splitlines() if line.startswith("[D2S_PROGRESS] ")
    ]
    assert [event["state"] for event in events] == ["running", "failed"]
    assert all(event["percent"] is None for event in events)
    assert not any(thread.name == f"Progress:{desc}" for thread in threading.enumerate())
