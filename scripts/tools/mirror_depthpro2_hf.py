from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = ROOT / "src" / "desktop2stereo"
WEIGHTS = (
    ("depthpro2-s-bgcc6f8bg3.pt", "S", "https://ml-site.cdn-apple.com/models/depth-pro-2/depthpro2-s-bgcc6f8bg3.pt"),
    ("depthpro2-b-xb7m9egktw.pt", "B", "https://ml-site.cdn-apple.com/models/depth-pro-2/depthpro2-b-xb7m9egktw.pt"),
    ("depthpro2-l-tegstu7fim.pt", "L", "https://ml-site.cdn-apple.com/models/depth-pro-2/depthpro2-l-tegstu7fim.pt"),
    ("depthpro2-h-eivtm5w7m5.pt", "H", "https://ml-site.cdn-apple.com/models/depth-pro-2/depthpro2-h-eivtm5w7m5.pt"),
    ("depthpro2-fov-ew55zg4kq7.pt", "FOV", "https://ml-site.cdn-apple.com/models/depth-pro-2/depthpro2-fov-ew55zg4kq7.pt"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, target: Path) -> None:
    if target.is_file() and target.stat().st_size > 0:
        print(f"[skip] {target.name} ({target.stat().st_size} bytes)")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    print(f"[download] {url}")
    request = Request(url, headers={"User-Agent": "Desktop2Stereo/DepthPro2 mirror"})
    with urlopen(request, timeout=60) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    if not tmp.is_file() or tmp.stat().st_size <= 0:
        raise RuntimeError(f"download produced an empty file: {url}")
    tmp.replace(target)


def _write_mirror_metadata(folder: Path, manifest: dict[str, object]) -> None:
    (folder / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    license_source = APP_ROOT / "stereo_runtime" / "model_impl" / "depth_pro_2" / "LICENSE_MODEL"
    if license_source.is_file():
        shutil.copyfile(license_source, folder / "LICENSE_MODEL")
    (folder / "LICENSE").write_text(
        "DepthPro2 weights are distributed by Apple under the Apple Machine Learning "
        "Research Model License. See LICENSE_MODEL in this repository. This mirror "
        "does not change or relax the upstream license. The weights are for "
        "non-commercial research use only.\n",
        encoding="utf-8",
    )
    (folder / "README.md").write_text(
        "# Desktop2Stereo DepthPro2 mirror\n\n"
        "Gated public mirror of the official Apple DepthPro2 checkpoints used by "
        "Desktop2Stereo.\n\n"
        "## Files\n\n"
        "- `depthpro2-s-bgcc6f8bg3.pt` — DepthPro2-S\n"
        "- `depthpro2-b-xb7m9egktw.pt` — DepthPro2-B\n"
        "- `depthpro2-l-tegstu7fim.pt` — DepthPro2-L\n"
        "- `depthpro2-h-eivtm5w7m5.pt` — DepthPro2-H\n"
        "- `depthpro2-fov-ew55zg4kq7.pt` — focal-length estimator\n\n"
        "The official source is https://github.com/apple-aiml-research/ml-depth-pro-2 "
        "and the official checkpoint host is https://ml-site.cdn-apple.com/models/depth-pro-2. "
        "The mirror exists to provide a stable Hugging Face download path; license "
        "and attribution remain with Apple.\n\n"
        "Access is gated. Review `LICENSE_MODEL` before use. These weights are "
        "research-only and are not licensed for commercial use.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and upload the gated lc700x/DepthPro2 mirror.")
    parser.add_argument("--repo-id", default="lc700x/DepthPro2")
    parser.add_argument("--source-dir", default=None, help="Existing directory containing the five .pt files.")
    parser.add_argument("--prepare-only", action="store_true", help="Build the local mirror without uploading.")
    args = parser.parse_args()

    folder = Path(args.source_dir) if args.source_dir else ROOT / "outputs" / "hf-depthpro2-mirror"
    folder.mkdir(parents=True, exist_ok=True)
    manifest_files: dict[str, object] = {}
    for filename, _label, url in WEIGHTS:
        target = folder / filename
        _download(url, target)
        manifest_files[filename] = {
            "size": target.stat().st_size,
            "sha256": _sha256(target),
            "source_url": url,
        }
    manifest = {
        "repo_id": args.repo_id,
        "license": "Apple Machine Learning Research Model License (research-only)",
        "files": manifest_files,
    }
    _write_mirror_metadata(folder, manifest)
    print(f"[ready] {folder}")
    if args.prepare_only:
        return

    from huggingface_hub import HfApi, whoami

    try:
        identity = whoami()
    except Exception as exc:
        raise SystemExit(
            "Hugging Face authentication is required. Run `hf auth login` in the "
            "project environment, then rerun this command. Do not pass a token "
            f"through chat. ({type(exc).__name__})"
        ) from exc
    if str(identity.get("name", "")).casefold() != args.repo_id.split("/", 1)[0].casefold():
        raise SystemExit(
            f"Logged-in user {identity.get('name')!r} is not the owner of {args.repo_id!r}."
        )

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=False, exist_ok=True)
    api.update_repo_settings(repo_id=args.repo_id, repo_type="model", gated="manual")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(folder),
        commit_message="Add gated DepthPro2 weights and SHA-256 manifest",
    )
    print(f"[uploaded] https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
