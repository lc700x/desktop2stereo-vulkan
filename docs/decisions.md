# UI Decision Notes

## 2026-11-20: Spatial controls in desktop and OpenXR menus

- Adapt the spatial UI reference's semantic grouping and explicit control-state hierarchy to the existing Flet desktop menu and rasterized OpenXR menu.
- Use theme semantic colors for primary start, destructive stop, and secondary reset actions so the desktop theme selector remains authoritative.
- Keep existing callbacks, dimensions, hit regions, OpenXR rendering, and toggle behavior unchanged; do not add the reference project's Compose runtime or dependencies.
- Preserve selected OpenXR choices while hovered, with pressed feedback taking precedence, so pointer/controller movement cannot visually erase the active selection.
