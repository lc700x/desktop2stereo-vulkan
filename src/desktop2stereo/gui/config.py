import json
import os

import yaml

from utils import DEFAULT_PORT
from utils.xr_headset_presets import DEFAULT_XR_HEADSET_MODEL

from .model_catalog import GUI_MODEL_CATALOG
from .paths import BASE_DIR


_MODEL_SIZES = ["Small", "SmallPlus", "Base", "Large", "Giant"]
_SIZE_ORDER = {s: i for i, s in enumerate(_MODEL_SIZES)}
_ENV_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".hdr")
_ENV_IMAGE_NAMES = ("background", "panorama", "equirectangular", "360", "sky", "skybox")


def _find_env_image_for_gui(room_dir):
    if not room_dir or not os.path.isdir(room_dir):
        return None
    for stem in _ENV_IMAGE_NAMES:
        for ext in _ENV_IMAGE_EXTS:
            path = os.path.join(room_dir, stem + ext)
            if os.path.isfile(path):
                return path
    for name in sorted(os.listdir(room_dir), key=lambda value: value.lower()):
        path = os.path.join(room_dir, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in _ENV_IMAGE_EXTS:
            return path
    return None


def parse_model_name(name):
    """Split model name into (family, size). DepthPro treated as Large."""
    parts = name.split("-")
    size_parts = []
    i = len(parts) - 1
    while i >= 0:
        matched = None
        for sz in _MODEL_SIZES:
            if parts[i].upper() == sz.upper():
                matched = sz
                break
        if matched:
            size_parts.insert(0, matched)
            i -= 1
        else:
            break
    if size_parts:
        family = "-".join(parts[:i + 1])
        size = "-".join(size_parts)
        return (family, size)
    return (name, "")


def build_family_size_map(model_list):
    """Returns (families_ordered, family_to_sizes) from list of full model names."""
    families = []
    family_to_sizes = {}
    for name in model_list:
        family, size = parse_model_name(name)
        if family not in family_to_sizes:
            family_to_sizes[family] = []
            families.append(family)
        if size and size not in family_to_sizes[family]:
            family_to_sizes[family].append(size)
    for family in family_to_sizes:
        family_to_sizes[family].sort(key=lambda s: _SIZE_ORDER.get(s, 99))
    return families, family_to_sizes


DEFAULT_MODEL_LIST = list(GUI_MODEL_CATALOG.keys())
DEFAULT_FAMILIES, FAMILY_TO_SIZES = build_family_size_map(DEFAULT_MODEL_LIST)
FAMILY_SIZE_TO_MODEL = {}
for name in DEFAULT_MODEL_LIST:
    f, s = parse_model_name(name)
    FAMILY_SIZE_TO_MODEL[(f, s)] = name


def default_base_depth_model():
    """Return the reset-time Base model, preferring the default model family."""
    if not DEFAULT_MODEL_LIST:
        return ""
    default_family, _ = parse_model_name(DEFAULT_MODEL_LIST[0])
    same_family_base = FAMILY_SIZE_TO_MODEL.get((default_family, "Base"))
    if same_family_base:
        return same_family_base
    if "Distill-Any-Depth-Base" in DEFAULT_MODEL_LIST:
        return "Distill-Any-Depth-Base"
    for name in DEFAULT_MODEL_LIST:
        _, size = parse_model_name(name)
        if size == "Base":
            return name
    return DEFAULT_MODEL_LIST[0]


DEFAULTS = {
    "Capture Mode": "Monitor",
    "Monitor Index": 1,
    "Monitor Identity": None,
    "Window Title": "",
    "Show FPS": False,
    "Model List": DEFAULT_MODEL_LIST,
    "Depth Model": (
        "Distill-Any-Depth-Base"
        if "Distill-Any-Depth-Base" in DEFAULT_MODEL_LIST
        else default_base_depth_model()
    ),
    "Depth Strength": 0.25,
    "Depth Quick": "Standard",
    "Depth Resolution": 518,
    "Anti-aliasing": 0,
    "Depth Antialias Strength": 0.0,
    "Depth Pop": 0.0,
    "Foreground Pop": 1.15,
    "Midground Pop": 1.05,
    "Background Pop": 1.05,
    "Depth Separation Preset": "standard",
    "Convergence": 0.0,
    "Dynamic Convergence": False,
    "Dynamic Convergence Strength": 0.0,
    "Stereo Preset": "cinema",
    "Stereo Quality": "quality_4k",
    "Stereo Compute Backend": "auto",
    "Parallax Budget Preset": "standard",
    "Temporal": False,
    "Temporal Strength": 0.0,
    "Auto Scene Reset": False,
    "Scene Reset Threshold": 0.22,
    "Edge Dilation": 1,
    "Mask Feather Radius": 1,
    "Hole Fill Mode": "none",
    "Hole Fill Radius": 0,
    "Hole Fill Strength": 0.0,
    "Edge Threshold": 0.04,
    "Cross Eyed": False,
    "Anaglyph Method": "red_cyan",
    "Color Brightness": 1.0,
    "Color Contrast": 1.0,
    "Color Saturation": 1.0,
    "Color Gamma": 1.0,
    "Color Temperature": 0.0,
    "Color Tint": 0.0,
    "Vulkan Projection Min LOD": 0.0,
    "Vulkan Projection Max LOD": 0.35,
    "Vulkan Projection MIP LOD Bias": -0.35,
    "Vulkan Projection RCAS Sharpness": 0.5,
    "Display Mode": "Half-SBS",
    "FP16": True,
    "torch.compile": False,
    "TensorRT": False,
    "Parallel Inference": False,
    "Parallel Inference Workers": 1,
    "Recompile TensorRT": False,
    "CoreML": False,
    "Recompile CoreML": False,
    "MIGraphX": False,
    "Recompile MIGraphX": False,
    "Recompile OpenVINO": False,
    "Computing Device": 0,
    "Language": "EN",
    "Run Mode": "OpenXR Link",
    "XR Headset Model": DEFAULT_XR_HEADSET_MODEL,
    "XR Preview Window": True,
    "VSync": False,
    "Window Preview": False,
    "Target FPS": 0,
    "Processing Resolution": "Auto",
    "Render Size Policy": "scaled",
    "Render Scale": "4K / 100%",
    "Render Fixed Width": 1920,
    "Render Fixed Height": 1080,
    "Render Max Pixels": 3840 * 2160,
    "Render Min Dimension": 480,
    "Render Align": 1,
    "Upscaler": "Off",
    "Upscaler Sharpness": 0.0,
    "Stream Protocol": "WebRTC",
    "Streamer Port": DEFAULT_PORT,
    "Stream Quality": 100,
    "Use Stream Calibration": True,
    "Stream Target Bitrate Mbps": 0,
    "Stream Peak Bitrate Mbps": 0,
    "Stream Calibration Port": DEFAULT_PORT + 1,
    "Stream Key": "live",
    "Stereo Mix": None,
    "Audio Capture Backend": "auto",
    "Video Encoder Backend": "auto",
    "CRF": 23,
    "Audio Delay": 0.0,
    "Controller Model": "PICO",
    "Environment Model": "Default",
    "NVIDIA Frame Generation": False,
    # Legacy key is read only for migration from pre-NvFRUC settings.
    "Lossless Scaling Support": False,
    "Capture Tool": "none",
    "Display Fit Mode": "contain",
    "Fill 16:9": True,
    "Fix Viewer Aspect": False,
    "Stereo Output": None,
    "Stereo Output Identity": None,
    "Show Log Panel": True,
}


def discover_environment_keys():
    """Return canonical environment keys saved to settings.yaml."""
    env_base = os.path.join(BASE_DIR, "xr_viewer", "environments")
    options = []
    if not os.path.isdir(env_base):
        return ["Default"]
    room_dirs = []
    for name in os.listdir(env_base):
        room_dir = os.path.join(env_base, name)
        if not os.path.isdir(room_dir) or name.startswith("."):
            continue
        if (
            os.path.isfile(os.path.join(room_dir, "profile.json"))
            or os.path.isfile(os.path.join(room_dir, "environment.glb"))
            or _find_env_image_for_gui(room_dir)
        ):
            room_dirs.append(name)
    if os.path.exists(os.path.join(env_base, "environment.glb")) and "Default" not in room_dirs:
        room_dirs.append("Default")
    options.extend(sorted(room_dirs, key=lambda key: (key.lower() != "default", key.lower())))
    return options or ["Default"]


def load_environment_display_names(keys=None):
    """Return per-folder display_name values from profile.json."""
    env_base = os.path.join(BASE_DIR, "xr_viewer", "environments")
    names_by_key = {}
    for key in keys or discover_environment_keys():
        names = {}
        profile_path = os.path.join(env_base, key, "profile.json")
        try:
            if os.path.isfile(profile_path):
                with open(profile_path, "r", encoding="utf-8-sig") as f:
                    raw = f.read().strip()
                if raw:
                    profile = json.loads(raw) or {}
                    display_name = profile.get("display_name") if isinstance(profile, dict) else None
                    if isinstance(display_name, dict):
                        names = {str(k): str(v) for k, v in display_name.items() if v}
                    elif isinstance(display_name, str) and display_name:
                        names = {"EN": display_name, "CN": display_name}
        except (OSError, ValueError):
            names = {}
        names_by_key[key] = names
    return names_by_key


def environment_display_label(key, lang="EN", names_by_key=None):
    """Map canonical environment key to the localized GUI label."""
    if key == "None":
        key = "Default"
    if key == "Default":
        return "默认" if lang == "CN" else "Default"
    names = (names_by_key or {}).get(key) or {}
    return names.get(lang) or names.get("EN") or key


def environment_key_from_label(label, lang="EN", keys=None, names_by_key=None):
    """Map a localized environment label back to its canonical key."""
    text = str(label or "").strip()
    keys = keys or discover_environment_keys()
    for key in keys:
        if environment_display_label(key, lang, names_by_key) == text:
            return key
    for key in keys:
        if key.lower() == text.lower():
            return key
    if text.lower() in ("默认", "none"):
        return "Default"
    return "Default" if "Default" in keys else (keys[0] if keys else "Default")


def get_environment_model_options(lang="EN", return_keys=False):
    """Return selectable room environment names for the GUI."""
    keys = discover_environment_keys()
    if return_keys:
        return keys
    names_by_key = load_environment_display_names(keys)
    return [environment_display_label(key, lang, names_by_key) for key in keys]


HAVE_YAML = True


def save_yaml(path, cfg):
    if not HAVE_YAML:
        return False, "PyYAML not installed"
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        os.replace(tmp, path)
        return True, ""
    except Exception as e:
        return False, str(e)
