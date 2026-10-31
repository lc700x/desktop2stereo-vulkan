#!/bin/bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Build the bundled Moonlight host on an Apple Silicon Mac." >&2
  exit 2
fi

root="$(cd "$(dirname "$0")/../.." && pwd)"
server_dir="$root/src/desktop2stereo/streaming/moonlight_server"
patch_file="$server_dir/sunshine-d2s-2026.914.233613.patch"
macos_compat_patch="$server_dir/sunshine-macos13-audio-availability.patch"
source_url="https://github.com/LizardByte/Sunshine.git"
source_tag="v2026.914.233613"
minimum_macos_version="13.0"
temp_root="${TMPDIR:-/tmp}"
temp_root="${temp_root%/}"
temp_dir="$(mktemp -d "$temp_root/d2s-sunshine.XXXXXX")"
trap 'rm -rf "$temp_dir"' EXIT
dependency_prefix="$temp_dir/dependencies"
ffmpeg_prefix="$temp_dir/ffmpeg"
build_jobs="$(sysctl -n hw.ncpu)"

download_source() {
  local url="$1" expected_sha256="$2" archive="$3" actual_sha256
  curl --fail --location --silent --show-error --retry 3 "$url" --output "$archive"
  actual_sha256="$(LC_ALL=C shasum -a 256 "$archive" | awk '{ print $1 }')"
  if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    echo "SHA-256 mismatch for $archive (got $actual_sha256)." >&2
    exit 1
  fi
}

build_macos13_dependencies() {
  local icu_archive="$temp_dir/icu4c-78.3-sources.tgz"
  local miniupnpc_archive="$temp_dir/miniupnpc-2.3.3.tar.gz"
  local openssl_archive="$temp_dir/openssl-3.5.9.tar.gz"
  local opus_archive="$temp_dir/opus-1.6.1.tar.gz"
  local target_flags="-arch arm64 -mmacosx-version-min=$minimum_macos_version"
  local library local_id dependency

  mkdir -p "$dependency_prefix"
  download_source \
    "https://github.com/unicode-org/icu/releases/download/release-78.3/icu4c-78.3-sources.tgz" \
    "3a2e7a47604ba702f345878308e6fefeca612ee895cf4a5f222e7955fabfe0c0" \
    "$icu_archive"
  download_source \
    "https://github.com/miniupnp/miniupnp/releases/download/miniupnpc_2_3_3/miniupnpc-2.3.3.tar.gz" \
    "d52a0afa614ad6c088cc9ddff1ae7d29c8c595ac5fdd321170a05f41e634bd1a" \
    "$miniupnpc_archive"
  download_source \
    "https://github.com/openssl/openssl/releases/download/openssl-3.5.9/openssl-3.5.9.tar.gz" \
    "603f5602e2eef00d77fbd429d34dcd5822bb301757a1bc9cdb24c670f1eb859a" \
    "$openssl_archive"
  download_source \
    "https://ftp.osuosl.org/pub/xiph/releases/opus/opus-1.6.1.tar.gz" \
    "6ffcb593207be92584df15b32466ed64bbec99109f007c82205f0194572411a1" \
    "$opus_archive"

  LC_ALL=C tar -xzf "$icu_archive" -C "$temp_dir"
  LC_ALL=C tar -xzf "$miniupnpc_archive" -C "$temp_dir"
  LC_ALL=C tar -xzf "$openssl_archive" -C "$temp_dir"
  LC_ALL=C tar -xzf "$opus_archive" -C "$temp_dir"

  (
    cd "$temp_dir/icu/source"
    env LC_ALL=C LANG=C MACOSX_DEPLOYMENT_TARGET="$minimum_macos_version" \
      CFLAGS="-O2 $target_flags" CXXFLAGS="-O2 $target_flags" \
      LDFLAGS="$target_flags" CC=clang CXX=clang++ \
      ./configure --prefix="$dependency_prefix" --enable-shared --disable-static \
        --with-data-packaging=library --disable-tests --disable-samples
    if ! gnumake -j"$build_jobs" > "$temp_dir/icu-build.log" 2>&1; then
      tail -n 80 "$temp_dir/icu-build.log" >&2
      exit 1
    fi
    if ! gnumake install > "$temp_dir/icu-install.log" 2>&1; then
      tail -n 80 "$temp_dir/icu-install.log" >&2
      exit 1
    fi
  )

  while IFS= read -r -d '' library; do
    local_id="$(otool -D "$library" | sed -n '2s/^[[:space:]]*//p')"
    if [[ "$local_id" == libicu*.dylib ]]; then
      install_name_tool -id "$dependency_prefix/lib/$local_id" "$library"
    fi
    while IFS= read -r dependency; do
      if [[ "$dependency" == libicu*.dylib ]]; then
        install_name_tool -change "$dependency" \
          "$dependency_prefix/lib/$dependency" "$library"
      fi
    done < <(otool -L "$library" | awk 'NR > 1 { print $1 }')
  done < <(find "$dependency_prefix/lib" -type f -name 'libicu*.78.3.dylib' -print0)

  env LC_ALL=C LANG=C MACOSX_DEPLOYMENT_TARGET="$minimum_macos_version" \
    cmake -S "$temp_dir/miniupnpc-2.3.3" -B "$temp_dir/miniupnpc-build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$dependency_prefix" \
      -DCMAKE_OSX_ARCHITECTURES=arm64 \
      -DCMAKE_OSX_DEPLOYMENT_TARGET="$minimum_macos_version" \
      -DUPNPC_BUILD_STATIC=OFF \
      -DUPNPC_BUILD_TESTS=OFF \
      -DUPNPC_BUILD_SAMPLE=OFF
  cmake --build "$temp_dir/miniupnpc-build" --parallel "$build_jobs"
  cmake --install "$temp_dir/miniupnpc-build"
  install_name_tool -id "$dependency_prefix/lib/libminiupnpc.21.dylib" \
    "$dependency_prefix/lib/libminiupnpc.2.3.3.dylib"

  (
    cd "$temp_dir/opus-1.6.1"
    env LC_ALL=C LANG=C MACOSX_DEPLOYMENT_TARGET="$minimum_macos_version" \
      CC=clang CFLAGS="-O3 $target_flags" LDFLAGS="$target_flags" \
      ./configure --prefix="$dependency_prefix" --disable-shared --enable-static \
        --disable-doc --disable-extra-programs
    if ! make -j"$build_jobs" > "$temp_dir/opus-build.log" 2>&1; then
      tail -n 80 "$temp_dir/opus-build.log" >&2
      exit 1
    fi
    if ! make install > "$temp_dir/opus-install.log" 2>&1; then
      tail -n 80 "$temp_dir/opus-install.log" >&2
      exit 1
    fi
  )

  (
    cd "$temp_dir/openssl-3.5.9"
    env LC_ALL=C LANG=C MACOSX_DEPLOYMENT_TARGET="$minimum_macos_version" \
      CC=clang CFLAGS="-O3 $target_flags" LDFLAGS="$target_flags" \
      ./Configure darwin64-arm64-cc --prefix="$dependency_prefix" \
        --openssldir="$dependency_prefix/etc/ssl" shared no-tests
    if ! make -j"$build_jobs" > "$temp_dir/openssl-build.log" 2>&1; then
      tail -n 80 "$temp_dir/openssl-build.log" >&2
      exit 1
    fi
    if ! make install_sw > "$temp_dir/openssl-install.log" 2>&1; then
      tail -n 80 "$temp_dir/openssl-install.log" >&2
      exit 1
    fi
  )
}

build_macos13_dependencies

git clone --depth 1 --branch "$source_tag" "$source_url" "$temp_dir/sunshine"
git -C "$temp_dir/sunshine" submodule update --init --depth 1 -- \
  third-party/Simple-Web-Server \
  third-party/TPCircularBuffer \
  third-party/build-deps \
  third-party/glad \
  third-party/libdisplaydevice \
  third-party/libvirtualhid \
  third-party/lizardbyte-common \
  third-party/moonlight-common-c
git -C "$temp_dir/sunshine/third-party/moonlight-common-c" \
  submodule update --init --depth 1 -- enet nanors
git -C "$temp_dir/sunshine/third-party/build-deps" \
  submodule update --init --depth 1 -- third-party/FFmpeg/FFmpeg
git -C "$temp_dir/sunshine" apply --include=src/video.cpp --include=src/nvhttp.cpp "$patch_file"
git -C "$temp_dir/sunshine" apply "$macos_compat_patch"
source_commit="$(git -C "$temp_dir/sunshine" rev-parse HEAD)"

cmake -S "$temp_dir/sunshine/third-party/build-deps" \
  -B "$temp_dir/ffmpeg-build" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$temp_dir/ffmpeg-stage" \
  -DFFMPEG_INSTALL_PREFIX="$ffmpeg_prefix" \
  -DCMAKE_C_COMPILER="$(brew --prefix llvm)/bin/clang" \
  -DCMAKE_CXX_COMPILER="$(brew --prefix llvm)/bin/clang++" \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$minimum_macos_version" \
  -DCMAKE_C_FLAGS="-arch arm64 -mmacosx-version-min=$minimum_macos_version" \
  -DCMAKE_CXX_FLAGS="-arch arm64 -mmacosx-version-min=$minimum_macos_version" \
  -DBUILD_ALL=OFF \
  -DBUILD_ALL_SUNSHINE=OFF \
  -DBUILD_FFMPEG=ON \
  -DBUILD_FFMPEG_SVT_AV1=OFF \
  -DBUILD_FFMPEG_X264=OFF \
  -DBUILD_FFMPEG_X265=OFF
if ! cmake --build "$temp_dir/ffmpeg-build" --parallel "$build_jobs" \
  > "$temp_dir/ffmpeg-build.log" 2>&1; then
  tail -n 100 "$temp_dir/ffmpeg-build.log" >&2
  exit 1
fi
if ! cmake --install "$temp_dir/ffmpeg-build" \
  > "$temp_dir/ffmpeg-install.log" 2>&1; then
  tail -n 100 "$temp_dir/ffmpeg-install.log" >&2
  exit 1
fi

MACOSX_DEPLOYMENT_TARGET="$minimum_macos_version" \
PKG_CONFIG_PATH="$dependency_prefix/lib/pkgconfig" \
BRANCH=main BUILD_VERSION="$source_tag" COMMIT="$source_commit" \
TAG="$source_tag" CLONE_URL="$source_url" \
cmake -S "$temp_dir/sunshine" -B "$temp_dir/build" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$temp_dir/stage" \
  -DCMAKE_C_COMPILER="$(brew --prefix llvm)/bin/clang" \
  -DCMAKE_CXX_COMPILER="$(brew --prefix llvm)/bin/clang++" \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$minimum_macos_version" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$dependency_prefix/lib" \
  -DICU_ROOT="$dependency_prefix" \
  -DOPENSSL_ROOT_DIR="$dependency_prefix" \
  -DCMAKE_PREFIX_PATH="$dependency_prefix" \
  -DOpus_ROOT_DIR="$dependency_prefix" \
  -DFFMPEG_PREPARED_BINARIES="$ffmpeg_prefix" \
  -DBUILD_DOCS=OFF \
  -DBUILD_TESTS=OFF \
  -DSUNSHINE_ENABLE_TRAY=OFF \
  -DSUNSHINE_BUILD_HOMEBREW=OFF \
  -DCUDA_FAIL_ON_MISSING=OFF
cmake --build "$temp_dir/build" --parallel
cmake --install "$temp_dir/build"

app="$temp_dir/stage/Sunshine.app"
if [[ ! -d "$app/Contents/MacOS" || ! -x "$app/Contents/MacOS/Sunshine" ]]; then
  echo "The Sunshine build did not produce the expected Sunshine.app bundle." >&2
  exit 1
fi

MACOSX_DEPLOYMENT_TARGET="$minimum_macos_version" clang++ \
  -std=c++17 -fobjc-arc -O2 -dynamiclib \
  -mmacosx-version-min="$minimum_macos_version" \
  "$root/src/desktop2stereo/streaming/videotoolbox_encoder.mm" \
  -framework Foundation -framework CoreMedia -framework CoreVideo \
  -framework Metal -framework VideoToolbox \
  -o "$temp_dir/d2s_videotoolbox_encoder.dylib"

check_macos_deployment_target() {
  local candidate="$1"
  local minos major minor
  minos="$(xcrun vtool -show-build "$candidate" | awk '$1 == "minos" { print $2; exit }')"
  if [[ -z "$minos" ]]; then
    echo "Could not verify the macOS deployment target for $candidate." >&2
    exit 1
  fi
  major="${minos%%.*}"
  minor="${minos#*.}"
  minor="${minor%%.*}"
  if (( major > 13 || (major == 13 && minor > 0) )); then
    echo "$candidate requires macOS $minos; rebuild bundled runtime dependencies for macOS $minimum_macos_version." >&2
    exit 1
  fi
}

while IFS= read -r -d '' candidate; do
  if /usr/bin/file -b "$candidate" | /usr/bin/grep -q 'Mach-O'; then
    check_macos_deployment_target "$candidate"
  fi
done < <(find "$app" -type f -print0)
check_macos_deployment_target "$temp_dir/d2s_videotoolbox_encoder.dylib"

mkdir -p "$server_dir/macos"
rm -rf "$server_dir/macos/Sunshine.app"
ditto --rsrc --extattr "$app" "$server_dir/macos/Sunshine.app"
cp "$temp_dir/d2s_videotoolbox_encoder.dylib" \
  "$server_dir/macos/d2s_videotoolbox_encoder.dylib"
codesign --force --deep --sign - "$server_dir/macos/Sunshine.app"
codesign --force --sign - "$server_dir/macos/d2s_videotoolbox_encoder.dylib"
echo "Bundled Sunshine and the VideoToolbox bridge in $server_dir/macos"
