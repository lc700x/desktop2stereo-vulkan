# UI Localization and Controller Guide Audit

**Audit date:** 2026-11-20
**Scope:** English/Chinese GUI catalogs and the OpenXR controller operation guide.

## Findings and corrections

- The GUI catalog contained `XR Preview Window` in English but not Chinese. Added `XR 预览窗口` and made the Flet label use the catalog entry directly.
- EN and CN catalogs now have matching key sets, including theme color names (English names remain unchanged in EN; CN has localized color names).
- Controller help incorrectly documented keyboard orbit as `Left Grip + left stick press`. The actual input handler orbits with left-stick axes while the left grip is held and the laser is latched to the keyboard; stick-click is not the orbit control. Corrected the Chinese and English guide rows.
- The non-room short-press `Left Y` action restores the initial screen position and dimensions but preserves rotation. Reworded both guides from “reset to default centered display” to “reset screen position and size.” The room-mode guide already describes its separate recenter behavior.
- Other reviewed face-button, trigger-chord, and A+B calibration instructions are represented by the existing controller-guide input tests.

## Translation notes

The seven Chinese catalog entries that intentionally retain their source spelling are technical/product terms: `Gamma`, `FP16`, `AMD AMF`, `torch.compile`, `TensorRT`, `CoreML`, and `OpenVINO`.

## Verification

- Localization and OpenXR/Flet settings suites plus controller-guide tests: **207 passed, 1 deselected**.
- The deselected existing layout test currently expects a non-fixed group bottom at 784 px; the current layout reports 768 px. This is unrelated to translation/controller-copy changes and remains an outstanding layout-test mismatch.
- Ruff was run on touched files but reports existing repository-wide `E501`/`F401` findings; no unrelated lint cleanup was performed.

## ABXY mapping cross-check (2026-11-21)

Compared the OpenXR action sampling hand paths, shortcut state machine, room-mode branch, developer A+B chord handler, and both English/Chinese guide rows. No further ABXY copy mismatch was found after the earlier reset wording correction.

- Right A: short press toggles curved/flat; 1-second hold cycles the status panel.
- Right B: short press toggles the VDXR backdrop; 1-second hold shows/hides the operation guide. During developer calibration, B saves and exits instead.
- Left X: short press toggles the virtual keyboard; 1-second hold toggles skybox/black background.
- Left Y: short press resets screen position/size (room mode recenters); 1-second hold cycles screen presets or, in room mode, room environments.
- Right A+B: 0.5-second hold switches controller model when available; 5-second hold enters/exits calibration.

Controller guide test module: **13 passed**, including short/long press action simulations and bilingual guide-row assertions.
