from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal


DepthOutputSemantics = Literal[
    "relative_inverse_depth",
    "raw_disparity",
    "relative_z_depth",
    "metric_z_depth",
]
DEFAULT_DEPTH_OUTPUT_SEMANTICS: DepthOutputSemantics = "relative_inverse_depth"


@dataclass(frozen=True)
class DepthModelSpec:
    name: str
    model_id: str
    family: str = "generic"
    output_semantics: DepthOutputSemantics = DEFAULT_DEPTH_OUTPUT_SEMANTICS

    def model_dir(self, cache_dir: str | Path = "./models") -> Path:
        return resolve_model_dir(self.model_id, cache_dir)


def resolve_model_dir(model_id: str, cache_dir: str | Path = "./models") -> Path:
    return Path(cache_dir) / ("models--" + str(model_id).replace("/", "--"))


class ModelRegistry:
    def __init__(self, specs: Iterable[DepthModelSpec]) -> None:
        self._by_name: dict[str, DepthModelSpec] = {}
        self._by_id: dict[str, DepthModelSpec] = {}
        for spec in specs:
            self._by_name[_key(spec.name)] = spec
            self._by_id.setdefault(_key(spec.model_id), spec)

    @classmethod
    def default(cls) -> "ModelRegistry":
        return cls(DEFAULT_MODEL_SPECS)

    def get(self, name_or_model_id: str) -> DepthModelSpec:
        key = _key(name_or_model_id)
        if key in self._by_name:
            return self._by_name[key]
        if key in self._by_id:
            return self._by_id[key]
        raise KeyError(f"unknown depth model: {name_or_model_id!r}")

    def resolve_model_id(self, name_or_model_id: str) -> str:
        return self.get(name_or_model_id).model_id

    def resolve_output_semantics(self, name_or_model_id: str) -> DepthOutputSemantics:
        """Return the declared raw output convention for a model."""
        try:
            return self.get(name_or_model_id).output_semantics
        except KeyError:
            # Preserve the historical behavior for custom models, but do not
            # infer metric behavior from arbitrary identifier substrings.
            return DEFAULT_DEPTH_OUTPUT_SEMANTICS

    def list(self) -> list[DepthModelSpec]:
        return list(self._by_name.values())

    def names(self) -> list[str]:
        return [spec.name for spec in self.list()]


def _key(value: str) -> str:
    return str(value).strip().lower()


def resolve_output_semantics(name_or_model_id: str) -> DepthOutputSemantics:
    return ModelRegistry.default().resolve_output_semantics(name_or_model_id)


DEFAULT_MODEL_SPECS: tuple[DepthModelSpec, ...] = (
    DepthModelSpec("Distill-Any-Depth-Small", "xingyang1/Distill-Any-Depth-Small-hf", "distill-any-depth"),
    DepthModelSpec("Distill-Any-Depth-Base", "lc700x/Distill-Any-Depth-Base-hf", "distill-any-depth"),
    DepthModelSpec("Distill-Any-Depth-Large", "xingyang1/Distill-Any-Depth-Large-hf", "distill-any-depth"),
    DepthModelSpec("Depth-Anything-V2-Small", "depth-anything/Depth-Anything-V2-Small-hf", "depth-anything-v2"),
    DepthModelSpec("Depth-Anything-V2-Base", "depth-anything/Depth-Anything-V2-Base-hf", "depth-anything-v2"),
    DepthModelSpec("Depth-Anything-V2-Large", "depth-anything/Depth-Anything-V2-Large-hf", "depth-anything-v2"),
    DepthModelSpec("InfiniDepth-Small", "lc700x/InfiniDepth-Small", "infinidepth", "raw_disparity"),
    DepthModelSpec("InfiniDepth-SmallPlus", "lc700x/InfiniDepth-SmallPlus", "infinidepth", "raw_disparity"),
    DepthModelSpec("InfiniDepth-Base", "lc700x/InfiniDepth-Base", "infinidepth", "raw_disparity"),
    DepthModelSpec("InfiniDepth-Large", "lc700x/InfiniDepth-Large", "infinidepth", "raw_disparity"),
    DepthModelSpec("ZipDepth-Base", "pablovela5620/zipdepth", "zipdepth"),
    DepthModelSpec("ZipDepth-Small", "pablovela5620/zipdepth", "zipdepth-npu"),
    DepthModelSpec("Video-Depth-Anything-Small", "depth-anything/Video-Depth-Anything-Small", "video-depth-anything"),
    DepthModelSpec("Video-Depth-Anything-Base", "depth-anything/Video-Depth-Anything-Base", "video-depth-anything"),
    DepthModelSpec("Video-Depth-Anything-Large", "depth-anything/Video-Depth-Anything-Large", "video-depth-anything"),
    DepthModelSpec("DA3-SMALL", "depth-anything/DA3-SMALL", "da3", "relative_z_depth"),
    DepthModelSpec("DA3-BASE", "depth-anything/DA3-BASE", "da3", "relative_z_depth"),
    DepthModelSpec("DA3-LARGE", "depth-anything/DA3-LARGE-1.1", "da3", "relative_z_depth"),
    DepthModelSpec("DA3-GIANT", "depth-anything/DA3-GIANT-1.1", "da3", "relative_z_depth"),
    DepthModelSpec("DA3METRIC-LARGE", "depth-anything/DA3METRIC-LARGE", "da3", "metric_z_depth"),
    DepthModelSpec("DA3NESTED-GIANT-LARGE", "depth-anything/DA3NESTED-GIANT-LARGE-1.1", "da3", "metric_z_depth"),
    DepthModelSpec("DA3MONO-LARGE", "depth-anything/DA3MONO-LARGE", "da3", "relative_z_depth"),
    DepthModelSpec("Depth-Anything-V2-Metric-Outdoor-Small", "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf", "depth-anything-v2-metric", "metric_z_depth"),
    DepthModelSpec("Depth-Anything-V2-Metric-Outdoor-Base", "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf", "depth-anything-v2-metric", "metric_z_depth"),
    DepthModelSpec("Depth-Anything-V2-Metric-Outdoor-Large", "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf", "depth-anything-v2-metric", "metric_z_depth"),
    DepthModelSpec("Depth-Anything-V2-Metric-Indoor-Small", "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", "depth-anything-v2-metric", "metric_z_depth"),
    DepthModelSpec("Depth-Anything-V2-Metric-Indoor-Base", "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf", "depth-anything-v2-metric", "metric_z_depth"),
    DepthModelSpec("Depth-Anything-V2-Metric-Indoor-Large", "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf", "depth-anything-v2-metric", "metric_z_depth"),
    DepthModelSpec("Metric-Video-Depth-Anything-Small", "depth-anything/Metric-Video-Depth-Anything-Small", "metric-video-depth-anything", "metric_z_depth"),
    DepthModelSpec("Metric-Video-Depth-Anything-Base", "depth-anything/Metric-Video-Depth-Anything-Base", "metric-video-depth-anything", "metric_z_depth"),
    DepthModelSpec("Metric-Video-Depth-Anything-Large", "depth-anything/Metric-Video-Depth-Anything-Large", "metric-video-depth-anything", "metric_z_depth"),
    DepthModelSpec("depth-anything-small", "LiheYoung/depth-anything-small-hf", "depth-anything-v1"),
    DepthModelSpec("depth-anything-base", "LiheYoung/depth-anything-base-hf", "depth-anything-v1"),
    DepthModelSpec("depth-anything-large", "LiheYoung/depth-anything-large-hf", "depth-anything-v1"),
    DepthModelSpec("depth-anything-indoor-large", "lc700x/depth-anything-indoor-large-hf", "depth-anything-v1"),
    DepthModelSpec("depth-anything-outdoor-large", "lc700x/depth-anything-outdoor-large-hf", "depth-anything-v1"),
    DepthModelSpec("dpt-dinov2-small-kitti", "facebook/dpt-dinov2-small-kitti", "dpt-dinov2", "metric_z_depth"),
    DepthModelSpec("dpt-dinov2-base-kitti", "lc700x/dpt-dinov2-base-kitti-hf", "dpt-dinov2", "metric_z_depth"),
    DepthModelSpec("dpt-dinov2-large-kitti", "lc700x/dpt-dinov2-large-kitti-hf", "dpt-dinov2", "metric_z_depth"),
    DepthModelSpec("dpt-dinov2-giant-kitti", "lc700x/dpt-dinov2-giant-kitti-hf", "dpt-dinov2", "metric_z_depth"),
    DepthModelSpec("dpt-dinov2-small-nyu", "dpt-dinov2-small-nyu-hf", "dpt-dinov2", "metric_z_depth"),
    DepthModelSpec("dpt-dinov2-base-nyu", "lc700x/dpt-dinov2-base-nyu-hf", "dpt-dinov2", "metric_z_depth"),
    DepthModelSpec("dpt-dinov2-large-nyu", "dpt-dinov2-large-nyu-hf", "dpt-dinov2", "metric_z_depth"),
    DepthModelSpec("dpt-dinov2-giant-nyu", "facebook/dpt-dinov2-giant-nyu", "dpt-dinov2", "metric_z_depth"),
    DepthModelSpec("depth-ai", "lc700x/depth-ai-hf", "other", "metric_z_depth"),
    DepthModelSpec("dpt-hybrid-midas", "lc700x/dpt-hybrid-midas-hf", "dpt"),
    DepthModelSpec("dpt-beit-base-384", "Intel/dpt-beit-base-384", "dpt"),
    DepthModelSpec("dpt-beit-large-512", "Intel/dpt-beit-large-512", "dpt"),
    DepthModelSpec("dpt-large", "Intel/dpt-large", "dpt"),
    DepthModelSpec("dpt-large-redesign", "lc700x/dpt-large-redesign-hf", "dpt"),
    DepthModelSpec("zoedepth-nyu-kitti", "Intel/zoedepth-nyu-kitti", "zoedepth", "metric_z_depth"),
    DepthModelSpec("zoedepth-nyu", "Intel/zoedepth-nyu", "zoedepth", "metric_z_depth"),
    DepthModelSpec("zoedepth-kitti", "Intel/zoedepth-kitti", "zoedepth", "metric_z_depth"),
    DepthModelSpec("DepthPro-Large", "apple/DepthPro-hf", "depthpro", "metric_z_depth"),
)
