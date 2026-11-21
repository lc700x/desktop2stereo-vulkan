# Repository Guidelines

## Project Structure & Module Organization

- `src/desktop2stereo/` contains the Python application. Boundaries are
  `capture/`, `stereo_runtime/`, `app_runtime/`, `gui/`, `streaming/`, `viewer/`,
  and `xr_viewer/`; Vulkan shaders and runtime assets live beside their loaders.
- `tests/` contains the pytest suite. `scripts/` is grouped into `benchmark/`,
  `smoke/`, `tools/`, `dev/`, and `windows/` for performance, integration, and
  diagnostic entry points.
- `native/` contains the CMake and Cargo bridges. Keep platform-specific code in
  its corresponding native or Python backend. Documentation is in `docs/`,
  sample media is in `samples/`, and GPU installers are in `src/env_install/`.

## Build, Test, and Development Commands

Use a provisioned project Python environment. Common commands:

```text
src/python3/bin/python -m pytest
python -m ruff check src tests scripts
src/python3/bin/python src/desktop2stereo/main.py --probe
src/python3/bin/python src/tools/openxr_vulkan_smoke.py --frames 300
```

Launch with `src/run_mac`, `src/run_linux.bash`, or `src/run_windows.bat`.
Build native bridges from their individual `native/*` directories with the
documented CMake or Cargo toolchain.

## Coding Style & Naming Conventions

Use four-space indentation, type hints, focused modules, and Ruff's configured
120-character limit. Python modules, functions, and variables use `snake_case`,
classes use `PascalCase`, and constants use `UPPER_SNAKE_CASE`. Resolve paths
through `project_paths.py` or runtime helpers; do not commit machine-specific
absolute paths. Preserve GPU zero-copy behavior and keep platform differences
behind backend boundaries.

## Testing Guidelines

Add tests under `tests/` as `test_*.py` files with `test_*` functions. Run the
nearest targeted test first, then the full suite. GPU, OpenXR, and OS-specific
tests may skip when hardware or libraries are unavailable; record the
environment for hardware validation. No repository-wide coverage threshold is
configured.

## Commit & Pull Request Guidelines

Use Conventional Commits such as `fix(capture): handle lost monitor` or
`feat(ui): add spatial menu control`. Keep subjects imperative and scoped.
Pull requests should explain behavior changes, link an issue or design record,
list tests and platform/GPU coverage, and include screenshots or logs for GUI,
OpenXR, or streaming changes. Do not include generated models, logs, build
output, credentials, or local runtime configuration.

## Configuration & Generated Files

Treat `src/desktop2stereo/settings.yaml`, runtime logs, downloaded models, and
build directories as machine-specific. Use `project_paths.env` or the
`D2S_PATHS_FILE` override for local layouts, and keep secrets and device paths
out of commits.
