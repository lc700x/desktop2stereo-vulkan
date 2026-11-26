"""OS-backed process locks used to prevent duplicate GPU consumers."""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path


class ProcessInstanceLock:
    """Hold an exclusive process lock for the lifetime of the owner."""

    def __init__(
        self,
        lock_path: str,
        pid_path: str,
        handle: object,
        mutex_scope: str,
    ):
        self.lock_path = Path(lock_path)
        self.pid_path = Path(pid_path)
        self._handle = handle
        self._mutex_scope = str(mutex_scope)

    @classmethod
    def acquire(
        cls,
        lock_path: str,
        pid_path: str,
        *,
        mutex_scope: str,
    ) -> ProcessInstanceLock | None:
        Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            return cls._acquire_windows(lock_path, pid_path, mutex_scope)
        return cls._acquire_posix(lock_path, pid_path, mutex_scope)

    @classmethod
    def _acquire_posix(
        cls,
        lock_path: str,
        pid_path: str,
        mutex_scope: str,
    ) -> ProcessInstanceLock | None:
        import fcntl

        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            return None
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        lock = cls(lock_path, pid_path, descriptor, mutex_scope)
        lock._write_pid()
        return lock

    @classmethod
    def _acquire_windows(
        cls,
        lock_path: str,
        pid_path: str,
        mutex_scope: str,
    ) -> ProcessInstanceLock | None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_wchar_p,
        ]
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        digest = hashlib.sha256(
            os.path.abspath(lock_path).encode("utf-8")
        ).hexdigest()[:24]
        name = f"Local\\Desktop2Stereo-{mutex_scope}-{digest}"
        handle = kernel32.CreateMutexW(None, 0, name)
        if not handle:
            raise OSError(ctypes.get_last_error(), "CreateMutexW failed")
        if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
            kernel32.CloseHandle(handle)
            return None
        lock = cls(lock_path, pid_path, (kernel32, handle), mutex_scope)
        lock._write_pid()
        return lock

    def _write_pid(self) -> None:
        temporary = self.pid_path.with_name(f"{self.pid_path.name}.{os.getpid()}.tmp")
        temporary.write_text(f"{os.getpid()}\n", encoding="ascii")
        os.replace(temporary, self.pid_path)

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            if isinstance(handle, int):
                os.close(handle)
            else:
                kernel32, mutex = handle
                kernel32.CloseHandle(mutex)
        finally:
            try:
                owner = self.pid_path.read_text(encoding="ascii").strip()
                if owner == str(os.getpid()):
                    self.pid_path.unlink(missing_ok=True)
            except OSError:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
