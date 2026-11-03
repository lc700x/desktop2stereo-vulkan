from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_VARIANTS = (
    "rocm-gfx1200-gfx1201",
    "rocm-rdna1-3",
    "cuda",
)
MOONLIGHT_REQUIRED_FILES = {
    "sunshine.exe",
    "zlib1.dll",
    "d2s_amd_encoder.dll",
    "LICENSE-Sunshine.txt",
    "NOTICE-Sunshine.txt",
    "sunshine-d2s-2026.914.233613.patch",
    "assets/apps.json",
    "assets/web/index.html",
}
SUNSHINE_VERSION = "2026.914.233613"
REQUIRE_SEPARATE_DRIVES = os.name == "nt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_reparse_points(path: Path) -> None:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    for current in (path, *path.parents):
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if current.is_symlink() or getattr(metadata, "st_file_attributes", 0) & reparse_flag:
            raise ValueError(f"Refusing to clean a path that crosses a symlink or junction: {current}")


def _direct_child(
    path: str | Path,
    parent: Path,
    label: str,
    *,
    allow_missing: bool = False,
) -> Path:
    candidate = Path(path).expanduser().absolute()
    _reject_reparse_points(candidate)
    resolved_parent = parent.resolve(strict=True)
    resolved = candidate.resolve(strict=not allow_missing)
    if resolved.parent != resolved_parent or resolved == resolved_parent:
        raise ValueError(f"{label} must be a direct child of {resolved_parent}: {resolved}")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    if not allow_missing and not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _validate_moonlight(validation: dict[str, Any], variant: str) -> tuple[int, str]:
    imports = validation.get("imports", {})
    if imports.get("pynvvideo_codec") != "2.2.0":
        raise ValueError(f"{variant} does not report PyNvVideoCodec 2.2.0")
    if imports.get("moonlight_bridge") != "streaming.moonlight_bridge":
        raise ValueError(f"{variant} failed the Moonlight bridge import check")
    if imports.get("moonlight_output") != "streaming.moonlight_output":
        raise ValueError(f"{variant} failed the Moonlight output import check")

    assets = validation.get("moonlight", {}).get("assets", {})
    files = assets.get("files", {})
    if not isinstance(files, dict) or not MOONLIGHT_REQUIRED_FILES.issubset(files):
        raise ValueError(f"{variant} has an incomplete Moonlight asset manifest")
    if assets.get("file_count") != len(files):
        raise ValueError(f"{variant} has an inconsistent Moonlight asset count")
    if any(name.startswith(("config/", "macos/")) for name in files):
        raise ValueError(f"{variant} Moonlight manifest includes excluded configuration or macOS files")
    if any(len(value) != 64 for value in files.values()):
        raise ValueError(f"{variant} Moonlight manifest has an invalid asset hash")
    digest_input = "".join(f"{name}  {digest}\n" for name, digest in sorted(files.items()))
    asset_hash = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
    if assets.get("sha256") != asset_hash:
        raise ValueError(f"{variant} Moonlight asset hash does not match its file list")

    runtime = validation.get("moonlight", {}).get("runtime", {})
    if runtime.get("sunshine_version") != SUNSHINE_VERSION or runtime.get("sunshine_exit_code") != 0:
        raise ValueError(f"{variant} Sunshine version check failed")
    if runtime.get("amd_encoder_abi", 0) < 2 or not runtime.get("zlib_loaded"):
        raise ValueError(f"{variant} Moonlight native dependency check failed")
    if runtime.get("moonlight_bridge_imported") != "streaming.moonlight_bridge":
        raise ValueError(f"{variant} Moonlight bridge runtime check failed")
    if runtime.get("moonlight_output_imported") != "streaming.moonlight_output":
        raise ValueError(f"{variant} Moonlight output runtime check failed")
    return assets["file_count"], asset_hash


def validate_cleanup_targets(
    release_root: str | Path,
    staging_root: str | Path,
    verification_root: str | Path,
    *,
    archives_moved: bool = False,
) -> dict[str, Any]:
    project_root = PROJECT_ROOT.resolve(strict=True)
    release = _direct_child(
        release_root,
        project_root / "dist" / "windows-portable",
        "Release directory",
    )
    staging = _direct_child(
        staging_root,
        project_root / ".portable-build",
        "Portable staging directory",
        allow_missing=True,
    )
    verification_candidate = Path(verification_root).expanduser().absolute()
    verification_parent = verification_candidate.parent
    if verification_parent.name != "Desktop2Stereo Portable 验证":
        raise ValueError(
            "Verification directory must be inside 'Desktop2Stereo Portable 验证'"
        )
    verification = _direct_child(
        verification_candidate,
        verification_parent,
        "Relocation verification directory",
        allow_missing=True,
    )

    source_manifest_path = release / "SOURCE_MANIFEST.json"
    validation_report_path = release / "validation-report.json"
    checksums_path = release / "SHA256SUMS.txt"
    for path in (source_manifest_path, validation_report_path, checksums_path):
        if not path.is_file():
            raise FileNotFoundError(f"Completed release metadata is missing: {path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    report = json.loads(validation_report_path.read_text(encoding="utf-8"))
    build_id = source_manifest.get("build_id")
    source_commit = source_manifest.get("source_commit")
    if not isinstance(build_id, str) or not build_id or not source_commit:
        raise ValueError("Release source manifest has no build identity")
    if report.get("build_id") != build_id or report.get("source_commit") != source_commit:
        raise ValueError("Release manifest and validation report refer to different builds")
    if not release.name.startswith(build_id):
        raise ValueError("Release directory does not match its build identity")
    if not staging.name.startswith(build_id) or not verification.name.startswith(build_id):
        raise ValueError("Cache directories do not match the release build identity")
    if len(set(source_manifest.get("packages", []))) != len(EXPECTED_VARIANTS):
        raise ValueError("Release does not contain exactly three distinct package variants")
    if set(source_manifest["packages"]) != set(EXPECTED_VARIANTS):
        raise ValueError("Release package variants do not match the Windows bundle set")
    if REQUIRE_SEPARATE_DRIVES and staging.drive.casefold() == verification.drive.casefold():
        raise ValueError("Staging and relocation verification must be on separate drives")
    if _is_within(release, staging) or _is_within(staging, release):
        raise ValueError("Release and staging directories overlap")
    if _is_within(release, verification) or _is_within(verification, release):
        raise ValueError("Release and verification directories overlap")
    if _is_within(staging, verification) or _is_within(verification, staging):
        raise ValueError("Staging and verification directories overlap")

    packages = report.get("packages", [])
    by_variant = {package.get("variant"): package for package in packages}
    if len(packages) != len(EXPECTED_VARIANTS) or set(by_variant) != set(EXPECTED_VARIANTS):
        raise ValueError("Validation report is missing a Windows package variant")

    expected_checksum_rows: list[str] = []
    local_archive_mismatches: list[Path] = []
    for variant in EXPECTED_VARIANTS:
        package = by_variant[variant]
        expected_name = f"Desktop2Stereo-{build_id}-{variant}.7z"
        if package.get("file") != expected_name:
            raise ValueError(f"Unexpected archive name for {variant}")
        archive = release / expected_name
        recorded_digest = package.get("sha256")
        recorded_size = package.get("size_bytes")
        if not isinstance(recorded_digest, str) or len(recorded_digest) != 64:
            raise ValueError(f"Validation report has an invalid SHA256 for {variant}")
        if archive.is_file() and archive.stat().st_size == recorded_size:
            digest = sha256_file(archive)
            if digest != recorded_digest:
                raise ValueError(f"Archive SHA256 does not match its validation report: {archive}")
        elif archives_moved:
            if archive.exists():
                local_archive_mismatches.append(archive)
            digest = recorded_digest
        else:
            raise ValueError(f"Archive is missing or has an unexpected size: {archive}")
        expected_checksum_rows.append(f"{digest}  {expected_name}")

        validation = package.get("validation", {})
        if validation.get("dependency_check") != "No broken requirements found.":
            raise ValueError(f"{variant} dependency check did not pass")
        if not validation.get("app_probe", {}).get("completed"):
            raise ValueError(f"{variant} application probe did not pass")
        if validation.get("torch_compile_smoke", {}).get("status") not in {"passed", "skipped"}:
            raise ValueError(f"{variant} Torch compile check is incomplete")
        asset_count, asset_hash = _validate_moonlight(validation, variant)

        portable_manifest_path = release / f"PORTABLE_MANIFEST-{variant}.json"
        if not portable_manifest_path.is_file():
            raise FileNotFoundError(f"Per-package manifest is missing: {portable_manifest_path}")
        portable_manifest = json.loads(portable_manifest_path.read_text(encoding="utf-8"))
        manifest_assets = portable_manifest.get("moonlight", {}).get("assets", {})
        if (
            portable_manifest.get("variant") != variant
            or portable_manifest.get("build_id") != build_id
            or portable_manifest.get("source_commit") != source_commit
            or manifest_assets.get("file_count") != asset_count
            or manifest_assets.get("sha256") != asset_hash
        ):
            raise ValueError(f"{variant} portable manifest does not match its validation report")

        for relative_path in (
            Path("dependencies") / variant / "requirements-resolved.txt",
            Path("dependencies") / variant / "wheel-sources-sha256.tsv",
            Path(f"SOURCE_SHA256SUMS-{variant}.txt"),
        ):
            if not (release / relative_path).is_file():
                raise FileNotFoundError(f"Release sidecar is missing: {release / relative_path}")

    checksum_rows = checksums_path.read_text(encoding="utf-8").splitlines()
    if checksum_rows != expected_checksum_rows:
        raise ValueError("SHA256SUMS.txt does not match the validated package archives")
    if staging.exists():
        for variant in EXPECTED_VARIANTS:
            if not (staging / variant).is_dir():
                raise FileNotFoundError(f"Staging cache is missing: {staging / variant}")
    if verification.exists():
        for variant in EXPECTED_VARIANTS:
            verification_caches = (
                verification / f"{build_id}-{variant}",
                verification / f"{build_id}-{variant}-verified",
            )
            if not any(path.is_dir() for path in verification_caches):
                raise FileNotFoundError(
                    f"Verification cache is missing for {variant}: {verification_caches[0]}"
                )

    return {
        "build_id": build_id,
        "release_root": release,
        "staging_root": staging,
        "verification_root": verification,
        "package_count": len(packages),
        "archives_moved": archives_moved,
        "local_archive_mismatches": local_archive_mismatches,
    }


def cleanup_cache_directories(plan: dict[str, Any], *, apply: bool = False) -> None:
    staging = Path(plan["staging_root"])
    verification = Path(plan["verification_root"])
    print(
        f"Verified {plan['package_count']} archives for build {plan['build_id']} "
        f"in {plan['release_root']}"
    )
    print(
        f"Staging cache: {staging}"
        + (" (present)" if staging.exists() else " (already absent)")
    )
    print(
        f"Relocation cache: {verification}"
        + (" (present)" if verification.exists() else " (already absent)")
    )
    if plan["archives_moved"]:
        print("Archive files were moved; release hashes and prior validation records will be used.")
        for archive in plan["local_archive_mismatches"]:
            print(f"Leaving the local archive file untouched: {archive}")
    if not apply:
        print("Dry run only. Pass --apply to remove these two cache directories.")
        return

    for cache in (staging, verification):
        if cache.exists():
            _reject_reparse_points(cache)
            shutil.rmtree(cache)
    print("Staging and relocation caches are absent; release files were left untouched.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove verified Windows portable build staging and relocation caches."
    )
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--verification-root", type=Path, required=True)
    parser.add_argument(
        "--archives-moved",
        action="store_true",
        help="Use recorded archive hashes and validation reports when archive files were moved.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete the validated staging and relocation directories. Default is a dry run.",
    )
    args = parser.parse_args()
    plan = validate_cleanup_targets(
        args.release_root,
        args.staging_root,
        args.verification_root,
        archives_moved=args.archives_moved,
    )
    cleanup_cache_directories(plan, apply=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
