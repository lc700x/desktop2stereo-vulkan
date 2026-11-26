Desktop2Stereo for macOS (Apple Silicon)

Requirements
- macOS 15.0 or newer on Apple Silicon
- Homebrew
- Internet access for Python dependencies and the Vulkan runtime

Install
1. Extract this folder where you want to keep the app. Install it in its final
   location before running the installer; the Python environment is created
   inside this folder.
2. Open Terminal, change to this folder, and run:

       ./install_mac.command

   The installer reuses an existing Python 3.11 or 3.12 installation, or
   installs Python 3.12 with Homebrew, then installs the Python dependencies
   and the MoltenVK/Vulkan loader formulas.

Run
After installation, start the app from this folder with:

       ./run_mac

The first launch may download the selected depth model. macOS may ask you to
allow screen recording or accessibility access when those features are used.

Security and macOS permissions

If macOS blocks Sunshine on its first launch, try starting Moonlight Streaming once. Then open System Settings > Privacy & Security > Security and choose Open Anyway beside Sunshine. Confirm the prompt. macOS saves this choice for later launches of that app. Only approve software from a source you trust.

Gatekeeper approval does not grant privacy permissions. For screen streaming, enable both Desktop2Stereo and Sunshine under System Settings > Privacy & Security > Screen Recording (called Screen & System Audio Recording on newer macOS releases). If remote keyboard or mouse input is blocked, grant Sunshine access under Accessibility or use Sunshine Settings > Troubleshooting > Permissions. Restart Moonlight Streaming after changing permissions. macOS requires these privacy permissions to be approved by the user.
