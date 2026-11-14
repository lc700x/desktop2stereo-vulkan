from __future__ import annotations

import time

from capture.media_browser import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    MediaBrowser,
    media_kind_for_path,
)


def _wait_for_scan(browser: MediaBrowser, deadline: float = 2.0) -> dict:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        state = browser.snapshot()
        if state["browser_entries"]:
            return state
        time.sleep(0.02)
    return browser.snapshot()


def test_oku3d_compatible_extension_inventory():
    assert len(VIDEO_EXTENSIONS) == 59
    assert len(IMAGE_EXTENSIONS) == 49
    assert {".mk3d", ".ssif", ".insv", ".hevc", ".rmvb"} <= VIDEO_EXTENSIONS
    assert {".heic", ".avif", ".psd", ".exr", ".jxl", ".svg", ".dpx"} <= IMAGE_EXTENSIONS


def test_media_kind_matches_video_image_and_disc(tmp_path):
    video = tmp_path / "movie.mp4"
    image = tmp_path / "still.png"
    iso = tmp_path / "movie.iso"
    disc = tmp_path / "disc"
    disc.mkdir()
    (disc / "BDMV").mkdir()
    video.write_bytes(b"v")
    image.write_bytes(b"i")
    iso.write_bytes(b"d")

    assert media_kind_for_path(video) == "video"
    assert media_kind_for_path(image) == "image"
    assert media_kind_for_path(iso) == "disc"
    assert media_kind_for_path(disc) == "disc"
    assert media_kind_for_path(tmp_path / "movie.mk3d") == "video"
    assert media_kind_for_path(tmp_path / "photo.heic") == "image"
    assert media_kind_for_path(tmp_path / "vector.svg") == "image"


def test_browser_orders_media_and_creates_playlist(tmp_path):
    (tmp_path / "b.mp4").write_bytes(b"b")
    (tmp_path / "a.mkv").write_bytes(b"a")
    (tmp_path / "still.png").write_bytes(b"p")
    (tmp_path / "notes.txt").write_text("skip", encoding="utf-8")
    browser = MediaBrowser(initial_directory=str(tmp_path))

    assert browser.open() is True
    state = _wait_for_scan(browser)

    assert [entry["name"] for entry in state["browser_entries"]] == [
        "a.mkv",
        "b.mp4",
        "still.png",
    ]
    assert browser.open_path(str(tmp_path / "b.mp4")) == (
        "media",
        str((tmp_path / "b.mp4").resolve()),
    )
    state = browser.snapshot()
    assert [entry["name"] for entry in state["playlist_entries"]] == [
        "a.mkv",
        "b.mp4",
        "still.png",
    ]
    assert state["playlist_index"] == 1
    assert browser.playlist_step(1) == str((tmp_path / "still.png").resolve())
    assert browser.playlist_step(-1) == str((tmp_path / "b.mp4").resolve())
    browser.close_and_join()


def test_media_playlist_can_be_primed_for_autoplay(tmp_path):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"1")
    second.write_bytes(b"2")
    browser = MediaBrowser(initial_directory=str(tmp_path))

    browser.prime_media_playlist(str(second))
    state = browser.snapshot()

    assert [entry["name"] for entry in state["playlist_entries"]] == [
        "first.mp4",
        "second.mp4",
    ]
    assert state["playlist_index"] == 1
    browser.close_and_join()


def test_browser_select_is_separate_from_playlist_current(tmp_path):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"1")
    second.write_bytes(b"2")
    browser = MediaBrowser(initial_directory=str(tmp_path))
    browser.open()
    _wait_for_scan(browser)
    browser.open_path(str(first))
    browser.select(str(second))

    state = browser.snapshot()
    assert state["browser_selection"] == str(second)
    assert state["playlist_index"] == 0
    browser.close_and_join()


def test_recent_media_history_persists_and_can_be_removed(tmp_path):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.png"
    first.write_bytes(b"1")
    second.write_bytes(b"2")
    history_path = tmp_path / "media-recent.json"

    browser = MediaBrowser(
        initial_directory=str(tmp_path), history_path=str(history_path)
    )
    assert browser.add_recent(str(first)) is True
    assert browser.add_recent(str(second)) is True
    assert [entry["path"] for entry in browser.snapshot()["recent_entries"]] == [
        str(second.resolve()),
        str(first.resolve()),
    ]
    browser.close_and_join()

    reloaded = MediaBrowser(history_path=str(history_path))
    try:
        recent = reloaded.snapshot()["recent_entries"]
        assert [entry["path"] for entry in recent] == [
            str(second.resolve()),
            str(first.resolve()),
        ]
        assert reloaded.remove_recent(str(second)) is True
        assert [entry["path"] for entry in reloaded.snapshot()["recent_entries"]] == [
            str(first.resolve()),
        ]
    finally:
        reloaded.close_and_join()
