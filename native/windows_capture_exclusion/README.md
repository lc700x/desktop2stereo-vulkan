# Windows Graphics Capture session filtering

This directory vendors the MIT-licensed `windows-capture` 2.0.0 crate. The
small local additions expose monitor-session window exclusions and each frame's
configuration iteration. `native/wgc_capture_exclusion.cpp` supplies the WGC
interfaces missing from the Windows SDK projection used by the GPU bindings.

The GPU wrappers are based on these upstream revisions:

- `wc_rocm` 0.1.3 at `ed9dfcc5d9236bb9a8bc41a7f1ebc796ce964be7`
- `wc_cuda` 0.1.3 at `62235d41851bf5aa2c3e182e6e05ed24253ff450`

The checked-in Cargo lockfiles pin the Rust dependency graph. The bundled
install patch scripts build the matching extension before replacing the
runtime's existing `.pyd` file. If Cargo is absent, the app keeps its
`WDA_EXCLUDEFROMCAPTURE` fallback.
