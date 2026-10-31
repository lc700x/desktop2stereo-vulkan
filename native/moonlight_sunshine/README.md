# Bundled Sunshine host for Desktop2Stereo Moonlight mode

This folder contains a standalone Sunshine-derived GameStream host component. Desktop2Stereo remains a separate application; Moonlight clients connect to the host through the standard GameStream pairing and streaming protocol.

## Upstream and license

- Upstream: [LizardByte/Sunshine](https://github.com/LizardByte/Sunshine)
- Release tag: `v2026.914.233613`
- Source commit: `63d35f702ee9e362e43263742981836ec0710384`
- Sunshine is licensed under GNU GPL v3. The corresponding upstream license and notice are included beside this file.
- `sunshine-d2s-2026.914.233613.patch` contains the D2S encoded-frame bridge and the Windows build adjustments used for this bundled binary.

The bridge is enabled only in the child Sunshine process started by Desktop2Stereo. D2S supplies encoded GPU frames over an authenticated loopback socket; Sunshine retains pairing, audio capture, remote input, GameStream packetization, and network transport. When the bridge environment variables are absent, Sunshine keeps its ordinary capture path.

## Bundled files

- `sunshine.exe`: upstream Sunshine plus the D2S bridge, version `2026.914.233613`.
- `zlib1.dll` and `assets/`: Sunshine runtime dependencies and web UI assets.
- `d2s_amd_encoder.dll`: ABI 2 Windows AMF bridge used only by Moonlight mode on ROCm. The ordinary Desktop2Stereo AMD encoder DLL is left untouched.
- The AMF runtime and GPU driver come from the installed AMD driver package.

The bundled host runs as a child process in the current user session. It does not install a Sunshine service, Windows driver, firewall rule, or UPnP mapping. Audio uses the current default Windows output device.

## Ports

Sunshine uses its standard base port `47989` when the full port family is free. If another process occupies one of its ports, Desktop2Stereo scans for a free base port, saves it in the per-user `sunshine.conf`, and opens the pairing page on the corresponding web port. The service advertises the host name `D2S Sunshine` (mDNS label `D2S-Sunshine`).

The default family uses TCP offsets `-5, 0, +1, +21` and UDP offsets `+9, +10, +11, +13` from the base port. Sunshine documents that some Moonlight clients do not support custom port families; if a client cannot discover or connect to the alternate port, stop the other host to free the standard ports.

Pairing remains PIN-protected. When a client request is detected during Moonlight streaming, D2S opens the PIN dialog automatically; “Pair device” can also open it manually. Enter the four-digit PIN shown by the Moonlight client, and D2S submits it to Sunshine for the cryptographic pairing handshake. Pair one new client at a time. Sunshine's Web UI account is created automatically as `Desktop2Stereo`; its generated password is stored in the current user's `%LOCALAPPDATA%/Desktop2Stereo/Moonlight/web_account.json`.

The “Sunshine Settings” button opens a native Flet configuration panel. It reads and saves settings through the bundled Sunshine API using the generated per-user account and the D2S-managed certificate, so the panel does not show a browser login or certificate page. The panel also offers an option to open Sunshine's web UI in the system browser; the username and password are shown in the panel, and the browser may still show its self-signed certificate warning.

## Rebuilding from source

Check out the exact upstream tag with its pinned submodules, then apply the included patch from the Sunshine repository root:

```powershell
git clone --branch v2026.914.233613 --recurse-submodules https://github.com/LizardByte/Sunshine.git sunshine
Set-Location sunshine
git apply /path/to/sunshine-d2s-2026.914.233613.patch
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_DOCS=OFF -DBUILD_TESTS=OFF -DSUNSHINE_ENABLE_TRAY=OFF -DSUNSHINE_BUILD_WIX_INSTALLER=OFF -DSUNSHINE_USE_STATIC_QT=OFF -DCUDA_FAIL_ON_MISSING=OFF
cmake --build build --config Release
cmake --install build --prefix dist
```

The patch pins the Boost dependency to the UCRT64 toolchain version used for this build and disables the optional WiX installer target. Rebuilds need the Windows build dependencies documented by Sunshine; the application bundle does not contain a compiler.

## Build identity

- Sunshine SHA-256: `32134E57B1373D94343B6C794BAC7DA4B54235D6511BA0546DEF10B9E86A2FD0`
- AMF bridge SHA-256: `5510DD74DEDE5C20814DC75650FED2A0DB2878D3992813038CCBDF25A3C599A8`
