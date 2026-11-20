from __future__ import annotations

import ctypes
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Oku3D-compatible media inventory (59 video and 49 image extension types).
# Extension acceptance is deliberately broader than the preferred demuxer path:
# FFmpeg probes the actual stream and unsupported hardware decoders fall back
# without making the output browser hide a valid local file.
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".wmv", ".m4v", ".ts",
    ".m2ts", ".mpg", ".mpeg", ".vob", ".flv", ".3gp", ".mxf", ".mk3d",
    ".iso", ".bdmv", ".ssif", ".mts", ".f4v", ".3g2", ".ogv", ".m2v",
    ".divx", ".asf", ".m3u8", ".mpd", ".ism", ".f4m", ".img", ".m2t",
    ".tp", ".trp", ".mod", ".tod", ".vro", ".dv", ".dif", ".m1v",
    ".m2p", ".mpe", ".mpv", ".qt", ".ogm", ".rm", ".rmvb", ".wtv",
    ".dvr-ms", ".nut", ".ivf", ".y4m", ".264", ".h264", ".avc", ".265",
    ".h265", ".hevc", ".insv",
}
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".avif", ".heic", ".heif",
    ".tiff", ".tif", ".bmp", ".gif", ".psd", ".exr", ".jxl", ".svg",
    ".jpe", ".jfif", ".pjp", ".pjpeg", ".apng", ".avifs", ".jp2",
    ".j2k", ".jpf", ".jpx", ".dpx", ".jls", ".tga", ".dds", ".hdr",
    ".qoi", ".pbm", ".pgm", ".ppm", ".pam", ".pnm", ".ico", ".heics",
    ".hif", ".insp", ".pcx", ".sgi", ".ras", ".sun", ".xwd", ".fits",
    ".pfm", ".dcm", ".svgz",
}
DISC_MARKERS = ("BDMV", "VIDEO_TS")
_MAX_ENTRIES = 2000
_MAX_RECENT = 12


@dataclass(frozen=True)
class MediaBrowserEntry:
    name: str
    path: str
    kind: str
    size: int = 0


def media_kind_for_path(path: str | os.PathLike[str]) -> str:
    """Return ``video``, ``image``, ``disc`` or an empty string."""
    candidate = Path(path)
    if candidate.is_dir():
        return "disc" if is_disc_directory(candidate) else "directory"
    suffix = candidate.suffix.casefold()
    if suffix == ".iso":
        return "disc"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return ""


def is_disc_directory(path: str | os.PathLike[str]) -> bool:
    try:
        root = Path(path)
        return root.is_dir() and any((root / marker).is_dir() for marker in DISC_MARKERS)
    except OSError:
        return False


def _windows_drives() -> list[MediaBrowserEntry]:
    if os.name != "nt":
        return []
    try:
        mask = int(ctypes.windll.kernel32.GetLogicalDrives())
        drive_type = ctypes.windll.kernel32.GetDriveTypeW
        drive_type.argtypes = [ctypes.c_wchar_p]
        drive_type.restype = ctypes.c_uint
    except Exception:
        return []
    entries: list[MediaBrowserEntry] = []
    for index in range(26):
        if not (mask & (1 << index)):
            continue
        letter = chr(ord("A") + index)
        root = f"{letter}:\\"
        kind = int(drive_type(root))
        if kind not in {2, 3, 4, 5}:
            continue
        try:
            if not Path(root).exists():
                continue
        except OSError:
            continue
        label = "光驱" if kind == 5 else "磁盘"
        entries.append(MediaBrowserEntry(f"{label} {letter}:", root, "drive"))
    return entries


def _mount_entries() -> list[MediaBrowserEntry]:
    drives = _windows_drives()
    if drives:
        return drives
    if os.name != "nt":
        candidates = ("/", "/mnt", "/media", str(Path.home()))
        return [
            MediaBrowserEntry(str(Path(path)), str(Path(path)), "drive")
            for path in candidates
            if Path(path).is_dir()
        ]
    return []


def _sort_entries(entries: Iterable[MediaBrowserEntry]) -> list[MediaBrowserEntry]:
    order = {"directory": 0, "disc": 1, "video": 2, "image": 3, "other": 4}
    return sorted(
        entries,
        key=lambda item: (order.get(item.kind, 9), item.name.casefold()),
    )


class MediaBrowser:
    """Threaded local-media browser shared by the GUI, viewer and OpenXR."""

    def __init__(self, *, initial_directory: str | None = None, history_path: str | None = None) -> None:
        self._lock = threading.RLock()
        self._closed = False
        self._visible = False
        self._path = str(initial_directory or Path.home())
        self._parent = ""
        self._entries: list[MediaBrowserEntry] = []
        self._mounts: list[MediaBrowserEntry] = []
        self._selected = ""
        self._playlist: list[MediaBrowserEntry] = []
        self._playlist_index = -1
        self._playlist_managed = False
        self._playlist_offset = 0
        self._playlist_sort_descending: bool | None = None
        self._recent: list[MediaBrowserEntry] = []
        self._history_path = str(history_path or "").strip()
        self._recent = self._load_recent()
        self._error = ""
        self._revision = 0
        self._generation = 0
        self._worker: threading.Thread | None = None

    @staticmethod
    def _same_path(left: str, right: str) -> bool:
        return os.path.normcase(os.path.abspath(left)) == os.path.normcase(
            os.path.abspath(right)
        )

    def _load_recent(self) -> list[MediaBrowserEntry]:
        if not self._history_path:
            return []
        try:
            with open(self._history_path, "r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, UnicodeError, ValueError):
            return []
        items = payload.get("entries", []) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            return []
        recent: list[MediaBrowserEntry] = []
        for item in items:
            raw_path = item.get("path") if isinstance(item, dict) else item
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            try:
                resolved = str(Path(raw_path).expanduser().resolve())
            except OSError:
                resolved = str(Path(raw_path).expanduser())
            kind = media_kind_for_path(resolved)
            if kind not in {"video", "image", "disc"}:
                continue
            if any(self._same_path(entry.path, resolved) for entry in recent):
                continue
            recent.append(
                MediaBrowserEntry(Path(resolved).name or resolved, resolved, kind)
            )
            if len(recent) >= _MAX_RECENT:
                break
        return recent

    def _save_recent_locked(self) -> None:
        if not self._history_path:
            return
        payload = {
            "version": 1,
            "entries": [
                {"path": entry.path, "kind": entry.kind}
                for entry in self._recent[:_MAX_RECENT]
            ],
        }
        parent = Path(self._history_path).parent
        temporary = f"{self._history_path}.{os.getpid()}.tmp"
        try:
            parent.mkdir(parents=True, exist_ok=True)
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
            os.replace(temporary, self._history_path)
        except OSError:
            try:
                os.unlink(temporary)
            except OSError:
                pass

    def add_recent(self, path: str) -> bool:
        candidate = str(path or "").strip()
        if not candidate:
            return False
        try:
            resolved = str(Path(candidate).expanduser().resolve())
        except OSError:
            return False
        kind = media_kind_for_path(resolved)
        if kind not in {"video", "image", "disc"}:
            return False
        entry = MediaBrowserEntry(Path(resolved).name or resolved, resolved, kind)
        with self._lock:
            self._recent = [
                entry,
                *(
                    item
                    for item in self._recent
                    if not self._same_path(item.path, resolved)
                ),
            ][:_MAX_RECENT]
            self._save_recent_locked()
            self._revision += 1
        return True

    def remove_recent(self, path: str) -> bool:
        candidate = str(path or "").strip()
        if not candidate:
            return False
        with self._lock:
            before = len(self._recent)
            self._recent = [
                item for item in self._recent if not self._same_path(item.path, candidate)
            ]
            if len(self._recent) == before:
                return False
            self._save_recent_locked()
            self._revision += 1
            return True

    @property
    def visible(self) -> bool:
        with self._lock:
            return self._visible

    def open(self, value=None) -> bool:
        with self._lock:
            self._visible = True
            self._error = ""
            if isinstance(value, dict):
                mode = str(value.get("mode") or "").casefold()
                path = value.get("path")
                if path:
                    return self._begin_navigate(str(path), selected="")
                if mode == "disc":
                    disc = next(
                        (entry for entry in _mount_entries() if "光驱" in entry.name),
                        None,
                    )
                    return self._begin_navigate(
                        disc.path if disc is not None else self._path,
                        selected="",
                    )
            elif value:
                return self._begin_navigate(str(value), selected="")
            return self._begin_navigate(self._path, selected="")
        return True

    def close(self) -> None:
        with self._lock:
            self._visible = False
            self._generation += 1
            self._error = ""
            self._revision += 1

    def up(self) -> bool:
        with self._lock:
            parent = self._parent
        return self._begin_navigate(parent, selected=self._path) if parent else False

    def refresh(self) -> bool:
        with self._lock:
            path = self._path
            selected = self._selected
        return self._begin_navigate(path, selected=selected)

    def navigate(self, path: str) -> bool:
        return self._begin_navigate(path, selected="")

    def set_error(self, message: str) -> None:
        with self._lock:
            self._error = str(message or "")
            self._revision += 1

    def select(self, path: str) -> bool:
        candidate = str(path or "").strip()
        if not candidate:
            return False
        with self._lock:
            if not any(
                os.path.normcase(entry.path) == os.path.normcase(candidate)
                for entry in self._entries
            ):
                return False
            self._selected = candidate
            self._revision += 1
        return True

    def prime_media_playlist(self, path: str) -> None:
        """Build a lightweight playlist from the selected media's folder."""
        candidate = Path(path)
        with self._lock:
            if self._playlist_managed:
                self._playlist_index = next(
                    (index for index, entry in enumerate(self._playlist) if self._same_path(entry.path, path)), -1
                )
                self._revision += 1
                return
        try:
            if candidate.is_dir():
                entries = [
                    MediaBrowserEntry(
                        candidate.name or str(candidate),
                        str(candidate.resolve()),
                        "disc" if is_disc_directory(candidate) else "directory",
                    )
                ]
            else:
                entries = []
                for child in candidate.parent.iterdir():
                    try:
                        kind = media_kind_for_path(child)
                    except OSError:
                        continue
                    if kind in {"video", "image", "disc"}:
                        entries.append(
                            MediaBrowserEntry(child.name, str(child.resolve()), kind)
                        )
                    if len(entries) >= _MAX_ENTRIES:
                        break
                entries = _sort_entries(entries)
            with self._lock:
                self._playlist = entries
                self._playlist_index = next(
                    (
                        index
                        for index, entry in enumerate(entries)
                        if os.path.normcase(entry.path)
                        == os.path.normcase(str(candidate.resolve()))
                    ),
                    0 if entries else -1,
                )
                self._revision += 1
        except OSError:
            return

    def playlist_step(self, offset: int) -> str | None:
        with self._lock:
            if not self._playlist:
                return None
            index = self._playlist_index + int(offset)
            if index < 0 or index >= len(self._playlist):
                return None
            self._playlist_index = index
            self._selected = self._playlist[index].path
            self._revision += 1
            return self._selected

    def playlist_select(self, path: str) -> str | None:
        with self._lock:
            for index, entry in enumerate(self._playlist):
                if os.path.normcase(entry.path) == os.path.normcase(str(path)):
                    self._playlist_index = index
                    self._selected = entry.path
                    self._revision += 1
                    return entry.path
        return None

    def playlist_add(self, path: str) -> bool:
        candidate = Path(str(path or "")).expanduser()
        try:
            if not str(path or "").strip() or not candidate.exists():
                return False
            resolved = str(candidate.resolve())
            kind = media_kind_for_path(resolved)
            if kind not in {"video", "image", "disc"}:
                return False
        except OSError:
            return False
        with self._lock:
            if any(self._same_path(entry.path, resolved) for entry in self._playlist):
                return False
            self._playlist.append(MediaBrowserEntry(candidate.name, resolved, kind))
            self._playlist_managed = True
            self._playlist_sort_descending = None
            self._revision += 1
        return True

    def playlist_move(self, path: str, offset: int) -> bool:
        with self._lock:
            index = next((i for i, entry in enumerate(self._playlist) if self._same_path(entry.path, path)), -1)
            target = index + int(offset)
            if index < 0 or target < 0 or target >= len(self._playlist) or target == index:
                return False
            current = self._playlist[self._playlist_index] if self._playlist_index >= 0 else None
            self._playlist.insert(target, self._playlist.pop(index))
            self._playlist_index = self._playlist.index(current) if current is not None else -1
            self._playlist_offset = (target // 4) * 4
            self._playlist_managed = True
            self._playlist_sort_descending = None
            self._revision += 1
        return True

    def playlist_sort(self, *, descending: bool = False) -> None:
        with self._lock:
            current = self._playlist[self._playlist_index] if self._playlist_index >= 0 else None
            self._playlist.sort(key=lambda entry: (entry.name.casefold(), entry.path.casefold()), reverse=descending)
            self._playlist_index = self._playlist.index(current) if current is not None else -1
            self._playlist_offset = 0
            self._playlist_sort_descending = bool(descending)
            self._playlist_managed = True
            self._revision += 1

    def playlist_page(self, offset: int) -> None:
        with self._lock:
            self._playlist_offset = max(0, min(max(0, ((len(self._playlist) - 1) // 4) * 4),
                self._playlist_offset + int(offset)))
            self._revision += 1

    def playlist_remove(self, path: str) -> bool:
        """Remove a queue entry only; never delete a media file or stop playback."""
        with self._lock:
            for index, entry in enumerate(self._playlist):
                if not self._same_path(entry.path, str(path)):
                    continue
                self._playlist.pop(index)
                if index <= self._playlist_index:
                    self._playlist_index -= 1
                self._playlist_offset = min(self._playlist_offset, max(0, ((len(self._playlist) - 1) // 4) * 4))
                self._playlist_managed = True
                self._revision += 1
                return True
        return False

    def playlist_clear(self) -> None:
        with self._lock:
            self._playlist.clear()
            self._playlist_index = -1
            self._playlist_offset = 0
            self._playlist_managed = True
            self._revision += 1

    def open_path(self, path: str) -> tuple[str, str]:
        """Return ``("media"|"directory"|"error", resolved_path)``."""
        candidate = str(path or "").strip()
        if not candidate:
            return "error", "没有选择项目"
        try:
            resolved = str(Path(candidate).expanduser().resolve())
        except OSError as exc:
            return "error", str(exc)
        if not Path(resolved).exists():
            return "error", f"路径不存在：{resolved}"
        if Path(resolved).is_dir():
            if is_disc_directory(resolved):
                with self._lock:
                    self._selected = resolved
                    if self._playlist_managed:
                        self.playlist_select(resolved)
                    else:
                        self._playlist = [MediaBrowserEntry(Path(resolved).name, resolved, "disc")]
                        self._playlist_index = 0
                    self._revision += 1
                return "media", resolved
            self._begin_navigate(resolved, selected="")
            return "directory", resolved
        kind = media_kind_for_path(resolved)
        if kind not in {"video", "image", "disc"}:
            return "error", f"不支持的文件类型：{Path(resolved).suffix or resolved}"
        with self._lock:
            self._selected = resolved
            queued = [
                entry
                for entry in self._entries
                if entry.kind in {"video", "image", "disc"}
            ]
            if not self._playlist_managed:
                self._playlist = queued or [MediaBrowserEntry(Path(resolved).name, resolved, kind)]
            self._playlist_index = next(
                (
                    index
                    for index, entry in enumerate(self._playlist)
                    if os.path.normcase(entry.path) == os.path.normcase(resolved)
                ),
                -1 if self._playlist_managed else 0,
            )
            self._revision += 1
        return "media", resolved

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "browser_visible": self._visible,
                "browser_path": self._path,
                "browser_parent": self._parent,
                "browser_selection": self._selected,
                "browser_error": self._error,
                "browser_revision": self._revision,
                "browser_entries": [
                    {
                        "name": entry.name,
                        "path": entry.path,
                        "kind": entry.kind,
                        "size": entry.size,
                    }
                    for entry in self._entries
                ],
                "browser_mounts": [
                    {"name": entry.name, "path": entry.path, "kind": entry.kind}
                    for entry in self._mounts
                ],
                "playlist_entries": [
                    {"name": entry.name, "path": entry.path, "kind": entry.kind}
                    for entry in self._playlist
                ],
                "playlist_index": self._playlist_index,
                "playlist_offset": self._playlist_offset,
                "playlist_sort_descending": self._playlist_sort_descending,
                "playlist_visible": bool(self._playlist),
                "recent_entries": [
                    {"name": entry.name, "path": entry.path, "kind": entry.kind}
                    for entry in self._recent[:_MAX_RECENT]
                ],
                "recent_visible": bool(self._recent),
            }

    def close_and_join(self) -> None:
        self.close()
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=1.0)
        with self._lock:
            self._closed = True

    def _begin_navigate(self, path: str, *, selected: str) -> bool:
        candidate = str(path or "").strip()
        if not candidate:
            return False
        with self._lock:
            if self._closed:
                return False
            self._generation += 1
            generation = self._generation
            self._visible = True
            self._error = ""
            self._selected = selected
            self._allowed_extensions = set(VIDEO_EXTENSIONS | IMAGE_EXTENSIONS)
            worker = threading.Thread(
                target=self._scan_worker,
                args=(generation, candidate, selected),
                name="MediaBrowserScan",
                daemon=True,
            )
            self._worker = worker
            worker.start()
        return True

    def _scan_worker(self, generation: int, candidate: str, selected: str) -> None:
        try:
            root = Path(candidate).expanduser()
            if root.is_file():
                selected = str(root.resolve())
                root = root.parent
            if not root.is_dir():
                raise FileNotFoundError(str(root))
            root = root.resolve()
            entries: list[MediaBrowserEntry] = []
            for child in root.iterdir():
                try:
                    if child.is_dir():
                        kind = "disc" if is_disc_directory(child) else "directory"
                        entries.append(MediaBrowserEntry(child.name, str(child), kind))
                        continue
                    kind = media_kind_for_path(child)
                    if kind in {"video", "image", "disc"}:
                        entries.append(
                            MediaBrowserEntry(
                                child.name,
                                str(child),
                                kind,
                                int(child.stat().st_size),
                            )
                        )
                except OSError:
                    continue
                if len(entries) >= _MAX_ENTRIES:
                    break
            entries = _sort_entries(entries)
            mounts = _mount_entries()
            with self._lock:
                if generation != self._generation or self._closed:
                    return
                self._path = str(root)
                self._parent = str(root.parent) if root.parent != root else ""
                self._entries = entries
                self._mounts = mounts
                self._selected = selected
                self._error = ""
                self._revision += 1
        except Exception as exc:
            with self._lock:
                if generation != self._generation or self._closed:
                    return
                self._error = f"{type(exc).__name__}: {exc}"
                self._revision += 1
