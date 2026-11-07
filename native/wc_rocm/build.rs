fn main() {
    let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("wgc_capture_exclusion.cpp");
    println!("cargo:rerun-if-changed={}", source.display());
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file(source)
        .compile("d2s_wgc_capture_exclusion");
    println!("cargo:rustc-link-lib=WindowsApp");
}
