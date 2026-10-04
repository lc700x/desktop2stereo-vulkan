"""Vendored Flet desktop client provisioning."""
from __future__ import annotations

import os
import hashlib
import logging
import subprocess
import shutil
import tarfile
import time
import zipfile
from pathlib import Path

from utils.platform_info import get_arch, get_flet_desktop_artifact_name, get_linux_distro_id, get_os_name

from .paths import GUI_DIR

CLIENTS_DIR = Path(GUI_DIR) / "flet_clients"
PACKAGES_DIR = Path(GUI_DIR) / "flet_packages"
logger = logging.getLogger(__name__)
_ARCHIVE_DIGEST_FILE = ".archive.sha256"

_LINUX_FALLBACK_ARTIFACTS = (
    "flet-linux-ubuntu22.04-light-amd64.tar.gz",
)


def _flet_process_ids(entries: list[tuple[int, int, str]], parent_pid: int) -> tuple[int, ...]:
    children: dict[int, list[int]] = {}
    process_info: dict[int, tuple[int, str]] = {}
    for pid, parent, name in entries:
        children.setdefault(parent, []).append(pid)
        process_info[pid] = (parent, name)

    pending = list(children.get(int(parent_pid), ()))
    descendants: set[int] = set()
    flet_processes: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in descendants:
            continue
        descendants.add(pid)
        info = process_info.get(pid)
        if info is None:
            continue
        _, name = info
        if name.casefold() == "flet.exe":
            flet_processes.add(pid)
        pending.extend(children.get(pid, ()))
    return tuple(sorted(flet_processes))


def _windows_process_snapshot() -> list[tuple[int, int, str]]:
    if os.name != "nt":
        return []

    import ctypes
    from ctypes import wintypes

    class ProcessEntry32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessEntry32W),
    ]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessEntry32W),
    ]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if not snapshot or snapshot == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())

    entries: list[tuple[int, int, str]] = []
    try:
        entry = ProcessEntry32W()
        entry.dwSize = ctypes.sizeof(entry)
        has_entry = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while has_entry:
            entries.append(
                (
                    int(entry.th32ProcessID),
                    int(entry.th32ParentProcessID),
                    str(entry.szExeFile),
                )
            )
            has_entry = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return entries


def _terminate_windows_processes(process_ids: tuple[int, ...]) -> None:
    if not process_ids:
        return

    import ctypes
    from ctypes import wintypes

    process_terminate = 0x0001
    synchronize = 0x00100000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    for pid in process_ids:
        handle = kernel32.OpenProcess(process_terminate | synchronize, False, int(pid))
        if not handle:
            continue
        try:
            kernel32.TerminateProcess(handle, 1)
            kernel32.WaitForSingleObject(handle, 1000)
        finally:
            kernel32.CloseHandle(handle)


def stop_flet_descendants(parent_pid: int, *, timeout_s: float = 3.0) -> tuple[int, ...]:
    """Reap only this GUI's bundled Flet clients after its window closes."""
    if os.name != "nt":
        return ()

    deadline = time.monotonic() + max(0.0, float(timeout_s))
    attempted: set[int] = set()
    while True:
        remaining = _flet_process_ids(_windows_process_snapshot(), parent_pid)
        if not remaining:
            return ()
        for pid in remaining:
            if pid in attempted:
                continue
            attempted.add(pid)
            try:
                subprocess.run(
                    ["taskkill", "/f", "/t", "/pid", str(pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=2.0,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except (OSError, subprocess.SubprocessError):
                logger.exception("Failed to stop Flet client process %s", pid)
        if time.monotonic() >= deadline:
            remaining = _flet_process_ids(_windows_process_snapshot(), parent_pid)
            _terminate_windows_processes(remaining)
            terminate_deadline = time.monotonic() + 1.0
            while remaining and time.monotonic() < terminate_deadline:
                time.sleep(0.05)
                remaining = _flet_process_ids(_windows_process_snapshot(), parent_pid)
            return remaining
        time.sleep(0.05)


def ensure_vendored_flet_view() -> str | None:
    """Ensure a bundled Flet client exists for this OS and set FLET_VIEW_PATH."""
    artifact = _select_artifact_name()
    if artifact is None:
        _print_missing_package_message()
        return None

    archive_path = PACKAGES_DIR / artifact
    extract_dir = CLIENTS_DIR / _archive_stem(artifact)
    view_path = _view_path_for_platform(extract_dir)
    archive_digest = _archive_digest(archive_path)
    if not _view_path_ready(view_path) or not _cache_matches_archive(extract_dir, archive_digest):
        _print_prepare_message(artifact)
        _extract_archive(archive_path, extract_dir, archive_digest)
        view_path = _view_path_for_platform(extract_dir)
        if not _view_path_ready(view_path):
            raise FileNotFoundError(
                f"Flet desktop client was extracted, but no runnable view was found in {extract_dir}"
            )

    os.environ["FLET_VIEW_PATH"] = str(view_path)
    return str(view_path)


def _select_artifact_name() -> str | None:
    for candidate in _artifact_candidates():
        if (PACKAGES_DIR / candidate).is_file():
            return candidate
    return None


def _artifact_candidates() -> list[str]:
    artifact = _current_artifact_name()
    candidates = [artifact] if artifact else []
    if _is_linux():
        candidates.extend(name for name in _LINUX_FALLBACK_ARTIFACTS if name not in candidates)
    return candidates


def _current_artifact_name() -> str | None:
    return get_flet_desktop_artifact_name()


def _is_linux() -> bool:
    return get_os_name() == "Linux"


def _system_label() -> str:
    os_name = get_os_name()
    if os_name == "Linux":
        return f"Linux {get_linux_distro_id()} {get_arch()}"
    return f"{os_name} {get_arch()}"


def _print_prepare_message(artifact: str) -> None:
    logger.info("[Flet GUI] Detected system: %s", _system_label())
    logger.info("[Flet GUI] Preparing Flet GUI package: %s", artifact)


def _print_missing_package_message() -> None:
    candidates = ", ".join(_artifact_candidates()) or "unknown"
    logger.error("[Flet GUI] Detected system: %s", _system_label())
    logger.error("[Flet GUI] Missing matching Flet GUI package in: %s", PACKAGES_DIR)
    logger.error("[Flet GUI] Expected package candidate(s): %s", candidates)
    logger.error("[Flet GUI] Please turn on VPN and run the program again.")


def _archive_stem(file_name: str) -> str:
    if file_name.endswith(".tar.gz"):
        return file_name[:-7]
    if file_name.endswith(".zip"):
        return file_name[:-4]
    return Path(file_name).stem


def _extract_archive(archive_path: Path, extract_dir: Path, archive_digest: str) -> None:
    tmp_dir = extract_dir.with_name(f"{extract_dir.name}.tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as archive:
                _safe_zip_extractall(archive, tmp_dir)
        else:
            with tarfile.open(archive_path, "r:gz") as archive:
                _safe_tar_extractall(archive, tmp_dir)

        (tmp_dir / _ARCHIVE_DIGEST_FILE).write_text(archive_digest, encoding="ascii")
        shutil.rmtree(extract_dir, ignore_errors=True)
        tmp_dir.rename(extract_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _archive_digest(archive_path: Path) -> str:
    digest = hashlib.sha256()
    with archive_path.open("rb") as archive_file:
        for block in iter(lambda: archive_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_matches_archive(extract_dir: Path, archive_digest: str) -> bool:
    try:
        return (extract_dir / _ARCHIVE_DIGEST_FILE).read_text(encoding="ascii").strip() == archive_digest
    except OSError:
        return False


def _safe_zip_extractall(archive: zipfile.ZipFile, target_dir: Path) -> None:
    target = target_dir.resolve()
    for member in archive.infolist():
        destination = (target_dir / member.filename).resolve()
        if not _is_relative_to(destination, target):
            raise ValueError(f"Unsafe zip member path: {member.filename}")
    archive.extractall(target_dir)


def _safe_tar_extractall(archive: tarfile.TarFile, target_dir: Path) -> None:
    target = target_dir.resolve()
    for member in archive.getmembers():
        destination = (target_dir / member.name).resolve()
        if not _is_relative_to(destination, target):
            raise ValueError(f"Unsafe tar member path: {member.name}")
    archive.extractall(target_dir)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False

def _view_path_for_platform(extract_dir: Path) -> Path:
    if get_os_name() == "Windows":
        parent = _find_file_parent(extract_dir, "flet.exe")
        return parent or (extract_dir / "flet")

    system_name = get_os_name()
    if system_name == "Darwin":
        return extract_dir

    parent = _find_file_parent(extract_dir, "flet")
    return parent or (extract_dir / "flet")


def _view_path_ready(view_path: Path) -> bool:
    if get_os_name() == "Windows":
        return (view_path / "flet.exe").is_file()

    system_name = get_os_name()
    if system_name == "Darwin":
        if not view_path.is_dir():
            return False
        return any(path.name.endswith(".app") and path.is_dir() for path in view_path.iterdir())

    return (view_path / "flet").is_file()


def _find_file_parent(root: Path, file_name: str) -> Path | None:
    if not root.exists():
        return None
    for path in root.rglob(file_name):
        if path.is_file():
            return path.parent
    return None
