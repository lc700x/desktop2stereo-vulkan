"""Apply the portable triton-JIT patches to the bundled ROCm python env.

Windows-only (called from install-rocm7-standalone.bat). Idempotent: safe to
run after every requirements install. The patches let triton compile
hip_utils with the bundled clang + project-local msvcstub headers/libs
(src/desktop2stereo/msvcstub) instead of requiring the Windows SDK/MSVC that
this bundle does not ship.

Run:  python apply-triton-portable-patches.py <python_site_packages_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path


def patch_file(path: Path, old: str, new: str, what: str) -> bool:
    if not path.is_file():
        print(f"[error] {path} missing ({what})")
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    if new in text:
        print(f"[ok] {path} already patched ({what})")
        return True
    if old not in text:
        print(f"[error] {path} has an unsupported Triton source layout ({what})")
        return False
    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="")
    print(f"[patch] {path} ({what})")
    return True


def main() -> int:
    if sys.platform != "win32":
        print("triton portable patches are Windows-only; skipped")
        return 0
    site_packages = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    if site_packages is None or not site_packages.is_dir():
        print("usage: apply-triton-portable-patches.py <site-packages dir>")
        return 2
    valid = patch_file(
        site_packages / "triton" / "windows_utils.py",
        '''def find_msvc(env_only: bool) -> tuple[Optional[str], list[str], list[str]]:
    if env_only:''',
        '''def find_msvc(env_only: bool) -> tuple[Optional[str], list[str], list[str]]:
    if os.environ.get("TRITON_NO_WINSDK"):
        # The bundled ROCm clang is selected before this discovery path. Avoid
        # adding host MSVC headers and libraries to the portable link command.
        return None, [], []
    if env_only:''',
        "windows_utils NO_WINSDK MSVC discovery",
    )
    valid = patch_file(
        site_packages / "triton" / "windows_utils.py",
        '''def find_winsdk(env_only: bool) -> tuple[list[str], list[str]]:
    if env_only:''',
        '''def find_winsdk(env_only: bool) -> tuple[list[str], list[str]]:
    import os as _os

    if _os.environ.get("TRITON_NO_WINSDK"):
        # ROCm-bundle-only mode: skip the Windows SDK include/lib paths; CRT
        # import libs come from the project-local msvcstub through the LIB
        # env var.
        _lib = _os.environ.get("LIB", "")
        return [], ([_lib] if _lib else [])
    if env_only:''',
        "windows_utils NO_WINSDK lib passthrough",
    ) and valid
    valid = patch_file(
        site_packages / "triton" / "runtime" / "build.py",
        '''        libraries = libraries + [f"python{version}"]
    if is_msvc(cc) or is_clang_cl(cc):''',
        '''        libraries = libraries + [f"python{version}"]
        if os.environ.get("TRITON_NO_WINSDK"):
            # clang-cl does not add kernel32 to the implicit link set.
            libraries = libraries + ["kernel32"]
    if is_msvc(cc) or is_clang_cl(cc):''',
        "build.py kernel32 lib",
    ) and valid
    valid = patch_file(
        site_packages
        / "triton" / "backends" / "amd" / "include" / "hip" / "hip_runtime.h",
        """#else
#include <stdint.h>
#include <stdlib.h>
#endif  // __cplusplus""",
        """#else
#include <stdint.h>
#include <stddef.h>
#endif  // __cplusplus""",
        "hip_runtime.h stdlib->stddef",
    ) and valid
    if not valid:
        print("triton portable patches failed; refusing to leave an unverified toolchain")
        return 1
    print("triton portable patches verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
