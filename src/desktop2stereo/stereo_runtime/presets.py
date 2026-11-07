from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Literal

from .openxr_render import OpenXRRenderConfig
from .output import OutputFormat
from .parallax import PARALLAX_BUDGET_TABLE
from .synthesis import StereoConfig

StereoModePreset = Literal["auto", "traditional_fastest", "cinema", "game_low_latency", "still_image_hq", "debug_export"]

PARALLAX_BUDGET_PRESETS = PARALLAX_BUDGET_TABLE

PRESET_CHOICES: tuple[StereoModePreset, ...] = (
    "auto",
    "traditional_fastest",
    "cinema",
    "game_low_latency",
    "still_image_hq",
    "debug_export",
)

_PRESET_ALIASES: dict[str, StereoModePreset] = {
    "auto": "auto",
    "traditional": "traditional_fastest",
    "traditional_fastest": "traditional_fastest",
    "traditional / fastest": "traditional_fastest",
    "fastest": "traditional_fastest",
    "cinema": "cinema",
    "movie": "cinema",
    "film": "cinema",
    "game": "game_low_latency",
    "game_low_latency": "game_low_latency",
    "game / low latency": "game_low_latency",
    "low_latency": "game_low_latency",
    "still": "still_image_hq",
    "still_image": "still_image_hq",
    "still_image_hq": "still_image_hq",
    "still image / hq": "still_image_hq",
    "hq": "still_image_hq",
    "debug": "debug_export",
    "export": "debug_export",
    "debug_export": "debug_export",
    "debug / export": "debug_export",
}


@dataclass(frozen=True)
class AutoModeSignals:
    """Pre-aggregated host/runtime signals.

    Hosts should collect GPU/input/audio/window metrics asynchronously and pass
    a smoothed snapshot here. This library never polls OS metrics in the
    capture or inference hot path.
    """

    frame_motion_score: float = 0.0
    scene_cut_score: float = 0.0
    still_duration_s: float = 0.0
    gpu_3d_util: float = 0.0
    gpu_video_decode_util: float = 0.0
    input_activity: float = 0.0
    idle_seconds: float = 0.0
    audio_active: bool = False
    maximized: bool = False
    foreground_process: str = ""
    fullscreen: bool = False
    openxr_active: bool = False
    user_export_action: bool = False
    latency_pressure: float = 0.0
    target_fps: float = 60.0


@dataclass(frozen=True)
class AutoModeDecision:
    preset: StereoModePreset
    reason: str
    hold_seconds: float = 3.0
    blend_seconds: float = 0.35
    require_consecutive_frames: int = 8
    scores: dict[str, float] | None = None


@dataclass(frozen=True)
class AutoModeRuntimeState:
    active_preset: StereoModePreset
    candidate_preset: StereoModePreset | None = None
    candidate_count: int = 0
    hold_remaining_s: float = 0.0
    last_decision: AutoModeDecision | None = None


class AutoModeRuntime:
    """Debounced auto-mode state machine for host runtimes.

    The update call is intentionally synchronous and cheap: it consumes a
    ready-made AutoModeSignals snapshot and never performs OS/GPU polling.
    """

    def __init__(self, initial_preset: StereoModePreset = "cinema") -> None:
        self.state = AutoModeRuntimeState(active_preset=normalize_preset(initial_preset))

    def reset(self, preset: StereoModePreset = "cinema") -> None:
        self.state = AutoModeRuntimeState(active_preset=normalize_preset(preset))

    def update(self, signals: AutoModeSignals, *, dt_s: float = 1.0) -> AutoModeDecision:
        proposed = classify_auto_mode(signals)
        hold_remaining = max(0.0, self.state.hold_remaining_s - max(0.0, float(dt_s)))

        if proposed.preset == self.state.active_preset:
            self.state = AutoModeRuntimeState(
                active_preset=self.state.active_preset,
                candidate_preset=None,
                candidate_count=0,
                hold_remaining_s=hold_remaining,
                last_decision=proposed,
            )
            return proposed

        if hold_remaining > 0.0 and proposed.preset not in _FAST_UPGRADE_PRESETS:
            decision = AutoModeDecision(
                self.state.active_preset,
                f"holding {self.state.active_preset}; candidate {proposed.preset}",
                hold_seconds=hold_remaining,
                blend_seconds=proposed.blend_seconds,
                require_consecutive_frames=proposed.require_consecutive_frames,
                scores=proposed.scores,
            )
            self.state = AutoModeRuntimeState(
                active_preset=self.state.active_preset,
                candidate_preset=proposed.preset,
                candidate_count=0,
                hold_remaining_s=hold_remaining,
                last_decision=decision,
            )
            return decision

        if proposed.preset == self.state.candidate_preset:
            candidate_count = self.state.candidate_count + 1
        else:
            candidate_count = 1

        required = max(1, proposed.require_consecutive_frames)
        if candidate_count >= required:
            self.state = AutoModeRuntimeState(
                active_preset=proposed.preset,
                candidate_preset=None,
                candidate_count=0,
                hold_remaining_s=proposed.hold_seconds,
                last_decision=proposed,
            )
            return proposed

        decision = AutoModeDecision(
            self.state.active_preset,
            f"confirming {proposed.preset} {candidate_count}/{required}",
            hold_seconds=hold_remaining,
            blend_seconds=proposed.blend_seconds,
            require_consecutive_frames=required,
            scores=proposed.scores,
        )
        self.state = AutoModeRuntimeState(
            active_preset=self.state.active_preset,
            candidate_preset=proposed.preset,
            candidate_count=candidate_count,
            hold_remaining_s=hold_remaining,
            last_decision=decision,
        )
        return decision


def normalize_preset(preset: str | StereoModePreset) -> StereoModePreset:
    key = str(preset).strip().lower().replace("-", "_")
    try:
        return _PRESET_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"unknown stereo mode preset: {preset!r}") from exc


def auto_detection_required(preset: str | StereoModePreset) -> bool:
    return normalize_preset(preset) == "auto"


def stereo_config_for_preset(
    preset: str | StereoModePreset,
    *,
    output_format: OutputFormat = "half_sbs",
    overrides: dict[str, Any] | None = None,
) -> StereoConfig:
    normalized = normalize_preset(preset)
    if normalized == "auto":
        normalized = "cinema"

    config = _STEREO_PRESETS[normalized]
    config = replace(config, output_format=output_format)
    if overrides:
        config = _replace_checked(config, overrides)
    return config


def openxr_config_for_preset(
    preset: str | StereoModePreset,
    *,
    screen_roll: float = 0.0,
    overrides: dict[str, Any] | None = None,
) -> OpenXRRenderConfig:
    normalized = normalize_preset(preset)
    if normalized == "auto":
        normalized = "cinema"

    config = _OPENXR_PRESETS[normalized]
    config = replace(config, screen_roll=screen_roll)
    if overrides:
        config = _replace_checked(config, overrides)
    return config


def classify_auto_mode(signals: AutoModeSignals) -> AutoModeDecision:
    if signals.user_export_action:
        return AutoModeDecision("debug_export", "user export/debug action", hold_seconds=2.0, blend_seconds=0.2, require_consecutive_frames=1)

    scores = auto_mode_scores(signals)
    game_score = scores["game"]
    video_score = scores["video"]
    still_score = scores["still"]

    if game_score >= 3.0 and game_score >= video_score + 1.0:
        return AutoModeDecision(
            "game_low_latency",
            "high 3d/input/latency behavior",
            hold_seconds=2.0,
            blend_seconds=0.2,
            require_consecutive_frames=4,
            scores=scores,
        )

    if video_score >= 3.0 and video_score >= game_score + 0.75:
        return AutoModeDecision(
            "cinema",
            "video decode dominated behavior",
            hold_seconds=3.0,
            blend_seconds=0.35,
            require_consecutive_frames=6,
            scores=scores,
        )

    if still_score >= 4.0 and game_score < 2.0 and video_score < 2.5:
        return AutoModeDecision(
            "still_image_hq",
            "idle low-motion behavior",
            hold_seconds=4.0,
            blend_seconds=0.5,
            require_consecutive_frames=12,
            scores=scores,
        )

    if abs(game_score - video_score) < 0.75 and max(game_score, video_score) >= 2.5:
        return AutoModeDecision("cinema", "mixed behavior fallback", hold_seconds=2.0, blend_seconds=0.35, require_consecutive_frames=8, scores=scores)

    if signals.openxr_active:
        return AutoModeDecision("cinema", "openxr active conservative cinema defaults", hold_seconds=3.0, blend_seconds=0.35, require_consecutive_frames=8, scores=scores)

    return AutoModeDecision("cinema", "default stable mode", hold_seconds=3.0, blend_seconds=0.35, require_consecutive_frames=8, scores=scores)


def auto_mode_scores(signals: AutoModeSignals) -> dict[str, float]:
    process = signals.foreground_process.lower()
    fullscreen_or_maximized = signals.fullscreen or signals.maximized
    gpu_3d = _clamp01(signals.gpu_3d_util)
    video_decode = _clamp01(signals.gpu_video_decode_util)
    input_activity = _clamp01(signals.input_activity)
    latency_pressure = _clamp01(signals.latency_pressure)
    frame_motion = _clamp01(signals.frame_motion_score)
    scene_cut = _clamp01(signals.scene_cut_score)

    game_hint = _contains_any(process, ("steam", "unity", "unreal", "ue4", "ue5", "dx", "vulkan"))
    video_hint = _contains_any(process, ("vlc", "mpv", "potplayer", "player", "video", "chrome", "edge", "firefox"))

    game = 0.0
    if gpu_3d >= 0.60:
        game += 3.0
    elif gpu_3d >= 0.30:
        game += 1.6
    if input_activity >= 0.65:
        game += 2.0
    elif input_activity >= 0.35:
        game += 0.8
    if fullscreen_or_maximized:
        game += 0.7
    if latency_pressure >= 0.65 or signals.target_fps >= 90.0:
        game += 1.0
    if frame_motion >= 0.35 or scene_cut >= 0.35:
        game += 0.9
    if game_hint:
        game += 0.5

    video = 0.0
    if video_decode >= 0.25 and gpu_3d < 0.20:
        video += 3.0
    elif video_decode >= 0.15:
        video += 1.4
    if fullscreen_or_maximized:
        video += 0.7
    if input_activity <= 0.15 and signals.idle_seconds >= 5.0:
        video += 0.8
    if signals.audio_active:
        video += 0.5
    if video_hint:
        video += 0.5

    still = 0.0
    if gpu_3d < 0.08:
        still += 1.0
    if video_decode < 0.05:
        still += 1.0
    if signals.idle_seconds >= 30.0:
        still += 3.0
    elif signals.idle_seconds >= 10.0:
        still += 1.0
    if frame_motion <= 0.03 and signals.still_duration_s >= 1.5:
        still += 2.0
    if not signals.audio_active:
        still += 0.4

    return {"game": game, "video": video, "still": still}


def stereo_config_for_auto_mode(
    signals: AutoModeSignals,
    *,
    output_format: OutputFormat = "half_sbs",
    overrides: dict[str, Any] | None = None,
) -> tuple[AutoModeDecision, StereoConfig]:
    decision = classify_auto_mode(signals)
    return decision, stereo_config_for_preset(decision.preset, output_format=output_format, overrides=overrides)


def openxr_config_for_auto_mode(
    signals: AutoModeSignals,
    *,
    screen_roll: float = 0.0,
    overrides: dict[str, Any] | None = None,
) -> tuple[AutoModeDecision, OpenXRRenderConfig]:
    decision = classify_auto_mode(signals)
    return decision, openxr_config_for_preset(decision.preset, screen_roll=screen_roll, overrides=overrides)


def preset_summary() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "stereo": asdict(_STEREO_PRESETS[name]),
            "openxr": asdict(_OPENXR_PRESETS[name]),
            "parallax_budget_presets": PARALLAX_BUDGET_PRESETS,
        }
        for name in ("traditional_fastest", "cinema", "game_low_latency", "still_image_hq", "debug_export")
    }


def _replace_checked(config: StereoConfig | OpenXRRenderConfig, overrides: dict[str, Any]):
    allowed = {field.name for field in fields(config)}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"unknown config override fields: {unknown}")
    return replace(config, **overrides)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


_FAST_UPGRADE_PRESETS: set[StereoModePreset] = {"game_low_latency", "debug_export"}


_STEREO_PRESETS: dict[StereoModePreset, StereoConfig] = {
    "auto": StereoConfig(),
    "traditional_fastest": StereoConfig(
        backend="fast",
        layers=2,
        temporal=False,
        temporal_strength=0.0,
        auto_reset_temporal=False,
        scene_reset_threshold=0.0,
        depth_strength=0.48,
        convergence=0.5,
        parallax_preset="standard",
        edge_dilation=0,
        edge_threshold=0.04,
        depth_antialias_strength=0.0,
        depth_pop=0.0,
        hole_fill="fast",
        hole_fill_mode="balanced",
        mask_feather_radius=0,
        fused=True,
    ),
    "cinema": StereoConfig(
        backend="quality_4k",
        layers=2,
        temporal=False,
        temporal_strength=0.0,
        auto_reset_temporal=False,
        scene_reset_threshold=0.22,
        depth_strength=0.48,
        convergence=0.5,
        parallax_preset="standard",
        edge_dilation=2,
        edge_threshold=0.04,
        depth_antialias_strength=1.0,
        depth_pop=0.0,
        hole_fill="none",
        hole_fill_mode="none",
        hole_fill_radius=0,
        hole_fill_strength=0.0,
        fused=True,
    ),
    "game_low_latency": StereoConfig(
        backend="fast_plus",
        layers=2,
        temporal=True,
        temporal_strength=0.25,
        auto_reset_temporal=True,
        scene_reset_threshold=0.18,
        depth_strength=0.48,
        convergence=0.5,
        parallax_preset="standard",
        edge_dilation=1,
        edge_threshold=0.04,
        depth_antialias_strength=0.0,
        depth_pop=0.0,
        hole_fill="none",
        hole_fill_mode="none",
        hole_fill_radius=0,
        hole_fill_strength=0.0,
        fused=True,
    ),
    "still_image_hq": StereoConfig(
        backend="hq_4k",
        layers=3,
        depth_strength=0.48,
        convergence=0.5,
        parallax_preset="standard",
        temporal=False,
        auto_reset_temporal=False,
        edge_threshold=0.04,
        depth_antialias_strength=1.5,
        depth_pop=0.0,
        hole_fill="edge_aware",
        hole_fill_mode="quality",
        hole_fill_radius=3,
        hole_fill_strength=1.0,
        fused=True,
    ),
    "debug_export": StereoConfig(
        backend="quality_4k",
        layers=2,
        temporal=True,
        temporal_strength=0.75,
        auto_reset_temporal=True,
        scene_reset_threshold=0.22,
        depth_strength=0.48,
        convergence=0.5,
        parallax_preset="standard",
        edge_dilation=2,
        edge_threshold=0.04,
        depth_antialias_strength=0.0,
        depth_pop=0.0,
        hole_fill="edge_aware",
        hole_fill_mode="quality",
        hole_fill_radius=3,
        hole_fill_strength=1.0,
        debug_output=True,
        fused=True,
    ),
}

_OPENXR_PRESETS: dict[StereoModePreset, OpenXRRenderConfig] = {
    "auto": OpenXRRenderConfig(),
    "traditional_fastest": OpenXRRenderConfig(depth_strength=0.48, convergence=0.5, parallax_preset="standard", output_mode="rgb_depth"),
    "cinema": OpenXRRenderConfig(depth_strength=0.48, convergence=0.5, parallax_preset="standard", output_mode="full_synthesis_eyes"),
    "game_low_latency": OpenXRRenderConfig(depth_strength=0.48, convergence=0.5, parallax_preset="standard"),
    "still_image_hq": OpenXRRenderConfig(depth_strength=0.48, convergence=0.5, parallax_preset="standard"),
    "debug_export": OpenXRRenderConfig(depth_strength=0.48, convergence=0.5, parallax_preset="standard"),
}
