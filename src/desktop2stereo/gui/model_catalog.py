"""Lightweight depth-model metadata for GUI startup."""

V25_GUI_MODEL_NAMES = (
    "Distill-Any-Depth-Small",
    "Distill-Any-Depth-Base",
    "Distill-Any-Depth-Large",
    "InfiniDepth-Small",
    "InfiniDepth-SmallPlus",
    "InfiniDepth-Base",
    "InfiniDepth-Large",
    "ZipDepth-Base",
    "ZipDepth-Small",
    "Depth-Anything-V2-Small",
    "Depth-Anything-V2-Base",
    "Depth-Anything-V2-Large",
    "Video-Depth-Anything-Small",
    "Video-Depth-Anything-Base",
    "Video-Depth-Anything-Large",
    "DA3-SMALL",
    "DA3-BASE",
    "DA3-LARGE",
    "DA3-GIANT",
    "DA3NESTED-GIANT-LARGE",
    "DA3MONO-LARGE",
    "DA3METRIC-LARGE",
    "Depth-Anything-V2-Metric-Outdoor-Small",
    "Depth-Anything-V2-Metric-Outdoor-Base",
    "Depth-Anything-V2-Metric-Outdoor-Large",
    "Depth-Anything-V2-Metric-Indoor-Small",
    "Depth-Anything-V2-Metric-Indoor-Base",
    "Depth-Anything-V2-Metric-Indoor-Large",
    "Metric-Video-Depth-Anything-Small",
    "Metric-Video-Depth-Anything-Base",
    "Metric-Video-Depth-Anything-Large",
    "depth-anything-small",
    "depth-anything-base",
    "depth-anything-large",
    "depth-anything-indoor-large",
    "depth-anything-outdoor-large",
    "DepthPro-Large",
)


def _resolutions_for_model(name):
    if name in {"ZipDepth-Base", "ZipDepth-Small"}:
        return [192, 224, 288, 352, 384, 448, 512]
    if name.startswith("InfiniDepth-"):
        return [192, 240, 304, 336, 384, 448, 512]
    if name.startswith("DA3"):
        return [182, 224, 280, 322, 378, 434, 504]
    if name == "DepthPro-Large":
        return [1536]
    return [196, 238, 294, 336, 392, 448, 518]


GUI_MODEL_CATALOG = {
    name: {"resolutions": _resolutions_for_model(name)}
    for name in V25_GUI_MODEL_NAMES
}
