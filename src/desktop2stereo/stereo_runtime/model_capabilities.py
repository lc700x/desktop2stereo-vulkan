"""Model/backend compatibility hints shared by GUI and runtime code."""

DISABLE_TRT_KEYWORDS = [
    "dpt-hybrid-midas",
    "depthpro",
    "da3-giant",
    "da3nested-giant",
    "video-depth-anything",
]

TRT_FIX_KEYWORDS = [
    "depth-anything/DA3-SMALL",
    "depth-anything/DA3-BASE",
    "depth-anything/DA3-LARGE",
    "depth-anything/DA3-LARGE-1.1",
    "depth-anything/DA3METRIC-LARGE",
    "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    "depth-anything/DA3MONO-LARGE",
    "depth-anything/Video-Depth-Anything-Small",
    "depth-anything/Video-Depth-Anything-Base",
    "depth-anything/Video-Depth-Anything-Large",
    "depth-anything/Metric-Video-Depth-Anything-Small",
    "depth-anything/Metric-Video-Depth-Anything-Base",
    "depth-anything/Metric-Video-Depth-Anything-Large",
    "Intel/zoedepth-nyu-kitti",
    "lc700x/InfiniDepth-Small",
    "lc700x/InfiniDepth-SmallPlus",
    "lc700x/InfiniDepth-Base",
    "lc700x/InfiniDepth-Large",
]

FORCE_FP32_KEYWORDS = [
    "Intel/zoedepth-nyu",
    "Intel/zoedepth-kitti",
    "depthpro",
    "zoedepth",
    "infinidepth-large",
]

COMPILE_FIX_KEYWORDS = [
    "depth-anything/Video-Depth-Anything-Small",
    "depth-anything/Video-Depth-Anything-Base",
    "depth-anything/Video-Depth-Anything-Large",
    "depth-anything/Metric-Video-Depth-Anything-Small",
    "depth-anything/Metric-Video-Depth-Anything-Base",
    "depth-anything/Metric-Video-Depth-Anything-Large",
]

DISABLE_COREML_KEYWORDS = [
    "video-depth-anything",
    "da3-",
    "da3nested",
    "dpt-beit",
    "zoedepth",
    "depthpro",
    "infinidepth",
    "zipdepth-base",
]

DISABLE_OPENVINO_KEYWORDS = [
    "da3-",
    "dpt-hybrid-midas-hf",
]

# MIGraphX is selected from the compiled ONNX graph. This variant is hidden
# because its Unfold upsampler is not the supported mobile/NPU graph path.
DISABLE_MIGRAPHX_KEYWORDS = ["zipdepth-base"]

DISABLE_CUDNN_KEYWORDS = [
    "6950",
    "6900",
    "6850",
    "6800",
    "6750",
    "6700",
    "6650",
    "6600",
    "6550",
    "6500",
    "6400",
    "6300",
    "680",
    "6100",
    "5700",
    "5600",
    "5500",
    "5400",
    "5300",
    "520",
    "160",
    "AMD Radeon(TM) Graphics",
]

DISABLE_TRITON_KEYWORDS = ["520", "160"]


def model_name_mapping():
    from .model_registry import ModelRegistry

    return {spec.name: spec.model_id for spec in ModelRegistry.default().list()}
