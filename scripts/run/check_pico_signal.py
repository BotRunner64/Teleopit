"""Pico mocap/video signal diagnostic entry point."""

from __future__ import annotations

from collections import Counter
import logging
import time
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig

from teleopit.inputs.human_frame_validation import HumanFrameValidationResult, validate_human_frame
from teleopit.inputs.pico4_provider import Pico4InputProvider
from teleopit.inputs.pico_video import PicoVideoRuntime, bridge_video_source, parse_pico_video_config
from teleopit.runtime.common import cfg_get


logger = logging.getLogger("teleopit.tools.check_pico_signal")


def _fmt_vec(values: tuple[float, ...] | None) -> str:
    if values is None:
        return "None"
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"


def _frame_stats(frame: dict[str, Any]) -> dict[str, Any]:
    positions = []
    quat_norms = []
    pelvis_pos = None
    for name, value in frame.items():
        try:
            pos, quat = value
        except Exception:
            continue
        try:
            pos_arr = np.asarray(pos, dtype=np.float64).reshape(-1)
            quat_arr = np.asarray(quat, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if pos_arr.shape[0] >= 3 and np.all(np.isfinite(pos_arr[:3])):
            positions.append(pos_arr[:3])
            if str(name) == "Pelvis":
                pelvis_pos = pos_arr[:3].copy()
        if quat_arr.size > 0 and np.all(np.isfinite(quat_arr)):
            quat_norms.append(float(np.linalg.norm(quat_arr)))

    if not positions:
        return {}

    pos = np.asarray(positions, dtype=np.float64)
    return {
        "pelvis_pos": pelvis_pos,
        "min_pos": np.min(pos, axis=0),
        "max_pos": np.max(pos, axis=0),
        "extent": np.ptp(pos, axis=0),
        "max_abs_pos": float(np.max(np.abs(pos))),
        "quat_norm_min": min(quat_norms) if quat_norms else None,
        "quat_norm_max": max(quat_norms) if quat_norms else None,
    }


def _log_invalid(seq: int, age_ms: float, result: HumanFrameValidationResult) -> None:
    logger.warning(
        "Invalid Pico body frame | seq=%s age_ms=%.1f reason=%s joint=%s "
        "max_abs_pos=%s threshold=%s pos=%s quat=%s detail=%s",
        seq,
        age_ms,
        result.reason,
        result.joint_name,
        f"{result.max_abs_pos:.4f}" if result.max_abs_pos is not None else "None",
        f"{result.max_pos_value:.4f}" if result.max_pos_value is not None else "None",
        _fmt_vec(result.pos),
        _fmt_vec(result.quat),
        result.detail,
    )


def _log_summary(
    *,
    window_s: float,
    total: int,
    valid: int,
    invalid_reasons: Counter[str],
    provider_fps: float,
    last_seq: int | None,
    last_age_ms: float | None,
    last_stats: dict[str, Any],
    pushed_video_frames: int,
) -> None:
    if total <= 0:
        logger.info(
            "Pico signal summary | window=%.1fs samples=0 provider_fps=%.1f "
            "last_seq=%s video_frames=%d",
            window_s,
            provider_fps,
            last_seq,
            pushed_video_frames,
        )
        return

    invalid = total - valid
    reason_text = ",".join(f"{reason}:{count}" for reason, count in invalid_reasons.most_common()) or "none"
    pelvis = last_stats.get("pelvis_pos")
    extent = last_stats.get("extent")
    min_pos = last_stats.get("min_pos")
    max_pos = last_stats.get("max_pos")
    logger.info(
        "Pico signal summary | window=%.1fs samples=%d valid=%d invalid=%d reasons=%s "
        "provider_fps=%.1f last_seq=%s last_age_ms=%s video_frames=%d "
        "max_abs_pos=%s pelvis=%s extent=%s min=%s max=%s quat_norm=[%s,%s]",
        window_s,
        total,
        valid,
        invalid,
        reason_text,
        provider_fps,
        last_seq,
        f"{last_age_ms:.1f}" if last_age_ms is not None else "None",
        pushed_video_frames,
        f"{last_stats.get('max_abs_pos'):.4f}" if "max_abs_pos" in last_stats else "None",
        _fmt_np_vec(pelvis),
        _fmt_np_vec(extent),
        _fmt_np_vec(min_pos),
        _fmt_np_vec(max_pos),
        _fmt_float(last_stats.get("quat_norm_min")),
        _fmt_float(last_stats.get("quat_norm_max")),
    )


def _fmt_np_vec(values: Any) -> str:
    if values is None:
        return "None"
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{float(value):.4f}" for value in arr) + "]"


def _fmt_float(value: Any) -> str:
    if value is None:
        return "None"
    return f"{float(value):.4f}"


def _build_provider(cfg: DictConfig, video_enabled: bool) -> Pico4InputProvider:
    input_cfg = cfg_get(cfg, "input", {}) or {}
    video_cfg = parse_pico_video_config(input_cfg)
    return Pico4InputProvider(
        human_format=str(cfg_get(input_cfg, "human_format", "pico_bridge")),
        timeout=float(cfg_get(input_cfg, "pico4_timeout", 60.0)),
        buffer_size=int(cfg_get(input_cfg, "pico4_buffer_size", 60)),
        timestamp_gap_reset_s=float(cfg_get(input_cfg, "pico4_timestamp_gap_reset_s", 0.15)),
        pause_button=cfg_get(input_cfg, "pause_button", "A"),
        pause_debounce_s=float(cfg_get(input_cfg, "pause_debounce_s", 0.25)),
        bridge_host=str(cfg_get(input_cfg, "bridge_host", "0.0.0.0")),
        bridge_port=int(cfg_get(input_cfg, "bridge_port", 63901)),
        bridge_discovery=bool(cfg_get(input_cfg, "bridge_discovery", True)),
        bridge_advertise_ip=cfg_get(input_cfg, "bridge_advertise_ip", None),
        bridge_video=bridge_video_source(video_cfg),
        bridge_video_enabled=video_enabled,
        bridge_start_timeout=float(cfg_get(input_cfg, "bridge_start_timeout", 10.0)),
        bridge_history_size=int(cfg_get(input_cfg, "bridge_history_size", 120)),
    )


@hydra.main(version_base=None, config_path="../../teleopit/configs", config_name="pico4_sim2real")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    input_cfg = cfg_get(cfg, "input", {}) or {}
    video_cfg = parse_pico_video_config(input_cfg)
    mocap_switch = cfg_get(cfg, "mocap_switch", {}) or {}
    max_pos_value = float(cfg_get(mocap_switch, "max_position_value", 5.0))
    diag_cfg = cfg_get(cfg, "diagnostic", {}) or {}
    poll_hz = float(cfg_get(diag_cfg, "poll_hz", cfg_get(cfg_get(cfg, "multiprocess", {}) or {}, "pico_io_hz", 120.0)))
    summary_interval_s = float(cfg_get(diag_cfg, "summary_interval_s", 1.0))
    duration_s = float(cfg_get(diag_cfg, "duration_s", 0.0))

    logger.info("Starting Pico signal diagnostic")
    logger.info(
        "Pico bridge | host=%s port=%s discovery=%s advertise_ip=%s",
        cfg_get(input_cfg, "bridge_host", "0.0.0.0"),
        cfg_get(input_cfg, "bridge_port", 63901),
        cfg_get(input_cfg, "bridge_discovery", True),
        cfg_get(input_cfg, "bridge_advertise_ip", None),
    )
    logger.info(
        "Signal check | max_position_value=%.3fm poll_hz=%.1f summary_interval_s=%.1f "
        "duration_s=%s video_enabled=%s video_source=%s",
        max_pos_value,
        poll_hz,
        summary_interval_s,
        f"{duration_s:.1f}" if duration_s > 0.0 else "until Ctrl-C",
        video_cfg.enabled,
        video_cfg.source,
    )

    provider = _build_provider(cfg, video_cfg.enabled)
    video_runtime = PicoVideoRuntime(provider=provider, config=video_cfg, mode="sim2real")
    total = 0
    valid = 0
    invalid_reasons: Counter[str] = Counter()
    last_seq: int | None = None
    last_age_ms: float | None = None
    last_stats: dict[str, Any] = {}
    window_start_s = time.monotonic()
    start_s = window_start_s
    sleep_s = 1.0 / max(poll_hz, 1.0)

    try:
        video_runtime.start()
        while True:
            now = time.monotonic()
            if duration_s > 0.0 and now - start_s >= duration_s:
                break

            video_runtime.tick()
            if provider.has_frame():
                try:
                    frame, timestamp_s, seq = provider.get_frame_packet()
                except Exception:
                    logger.exception("Failed to read Pico body frame packet")
                else:
                    seq = int(seq)
                    if seq != last_seq:
                        last_seq = seq
                        last_age_ms = max((time.monotonic() - float(timestamp_s)) * 1000.0, 0.0)
                        last_stats = _frame_stats(frame)
                        result = validate_human_frame(frame, max_pos_value=max_pos_value)
                        total += 1
                        if result.valid:
                            valid += 1
                        else:
                            invalid_reasons[result.reason] += 1
                            _log_invalid(seq, last_age_ms, result)

            now = time.monotonic()
            if now - window_start_s >= summary_interval_s:
                _log_summary(
                    window_s=now - window_start_s,
                    total=total,
                    valid=valid,
                    invalid_reasons=invalid_reasons,
                    provider_fps=float(provider.fps),
                    last_seq=last_seq,
                    last_age_ms=last_age_ms,
                    last_stats=last_stats,
                    pushed_video_frames=video_runtime.pushed_frames,
                )
                total = 0
                valid = 0
                invalid_reasons.clear()
                window_start_s = now

            time.sleep(sleep_s)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt -- stopping Pico signal diagnostic")
    finally:
        video_runtime.stop()
        provider.close()


if __name__ == "__main__":
    main()
