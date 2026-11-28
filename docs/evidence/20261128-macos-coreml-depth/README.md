# macOS Core ML Depth Evidence

Validation date: 2026-11-28

Environment:

- Apple M1 iMac, 8-core CPU, 16 GB unified memory.
- macOS 15.2 arm64.
- Apple Silicon Metal Local Viewer.
- Strict Core ML GPU compute plan.

Artifacts:

- `report.md`: readable offline depth/Half-SBS and Local Viewer summary.
- `report.json` and `report.csv`: machine-readable measurements.
- `contact_sheets/`: Distill-Small@336 and Distill-Base@518 comparisons
  against every new model at both evidence resolutions.

The raw Local Viewer logs remain under the ignored
`outputs/macos_coreml_matrix/local_viewer/` directory. All nine runs show:

- `depth_backend=coreml`
- `fallback=false`
- `coreml_input_shared=1`
- `coreml_output_zero_copy=1`

The live model throughput should be read from `content_fps`, not
`present_fps`. The latter includes display reuse while the next depth result
is still running.
