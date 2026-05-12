"""Optional camera-to-Pico video streaming for pico-bridge 0.2.0."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Any

import numpy as np

from teleopit.runtime.common import cfg_get

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PicoVideoConfig:
    enabled: bool = False
    source: str | None = None
    width: int = 1280
    height: int = 720
    fps: int = 30
    device: str | None = None
    fail_on_error: bool = False


def parse_pico_video_config(input_cfg: Any) -> PicoVideoConfig:
    video_cfg = cfg_get(input_cfg, "video", {}) or {}
    source = cfg_get(video_cfg, "source", None)
    source_str = None if source in (None, "", "null") else str(source).lower()
    enabled = bool(cfg_get(video_cfg, "enabled", False))
    if enabled and source_str not in ("realsense", "mujoco", "test-pattern"):
        raise ValueError(
            "input.video.source must be one of realsense, mujoco, or test-pattern when input.video.enabled=true"
        )
    width = int(cfg_get(video_cfg, "width", 1280))
    height = int(cfg_get(video_cfg, "height", 720))
    fps = int(cfg_get(video_cfg, "fps", 30))
    if enabled and (width <= 0 or height <= 0 or fps <= 0):
        raise ValueError("input.video.width, height, and fps must be positive when video is enabled")
    device = cfg_get(video_cfg, "device", None)
    return PicoVideoConfig(
        enabled=enabled,
        source=source_str,
        width=width,
        height=height,
        fps=fps,
        device=None if device in (None, "", "null") else str(device),
        fail_on_error=bool(cfg_get(video_cfg, "fail_on_error", False)),
    )


def bridge_video_source(config: PicoVideoConfig) -> str | None:
    if not config.enabled:
        return None
    if config.source == "test-pattern":
        return "test-pattern"
    return "frames"


class PicoVideoRuntime:
    """Lifecycle wrapper for optional video streaming into a Pico4InputProvider."""

    def __init__(
        self,
        *,
        provider: Any,
        config: PicoVideoConfig,
        mode: str,
        robot: Any | None = None,
    ) -> None:
        self._provider = provider
        self._config = config
        self._mode = mode
        self._robot = robot
        self._producer: _VideoProducer | None = None
        self._stopped = False

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def start(self) -> None:
        if not self._config.enabled:
            return
        self._stopped = False
        if self._config.source == "test-pattern":
            logger.info("Pico video enabled via pico-bridge test-pattern source")
            return
        if not callable(getattr(self._provider, "push_video_frame", None)):
            self._handle_error(RuntimeError("Pico input provider does not support push_video_frame"))
            return

        producer: _VideoProducer | None = None
        try:
            if self._config.source == "realsense":
                producer = _RealSenseVideoProducer(self._provider, self._config)
            elif self._config.source == "mujoco":
                producer = _MujocoCameraVideoProducer(self._provider, self._config, self._robot)
            else:
                raise ValueError(f"Unsupported Pico video source: {self._config.source!r}")
            self._producer = producer
            producer.start()
            logger.info("Pico video producer started | source=%s", self._config.source)
        except Exception as exc:
            if producer is not None:
                try:
                    producer.stop()
                except Exception:
                    logger.exception("Failed to stop Pico video producer after startup error")
            self._producer = None
            self._handle_error(exc)

    def tick(self) -> None:
        if self._producer is None or self._stopped:
            return
        try:
            self._producer.tick()
        except Exception as exc:
            try:
                self.stop()
            finally:
                self._handle_error(exc)

    def stop(self) -> None:
        self._stopped = True
        if self._producer is not None:
            self._producer.stop()
            self._producer = None

    def _handle_error(self, exc: Exception) -> None:
        if self._config.fail_on_error:
            raise RuntimeError("Pico video pipeline failed") from exc
        logger.warning("Pico video disabled after error: %s", exc)


class _VideoProducer:
    def start(self) -> None: ...

    def tick(self) -> None: ...

    def stop(self) -> None: ...


class _RealSenseVideoProducer(_VideoProducer):
    def __init__(self, provider: Any, config: PicoVideoConfig) -> None:
        self._provider = provider
        self._config = config
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pico_realsense_video", daemon=True)
        self._error: BaseException | None = None

    def start(self) -> None:
        self._thread.start()
        self._ready_event.wait(timeout=5.0)
        if self._error is not None:
            raise RuntimeError("failed to start RealSense video producer") from self._error
        if not self._ready_event.is_set():
            raise TimeoutError("RealSense video producer did not become ready within 5s")

    def tick(self) -> None:
        if self._error is not None:
            raise RuntimeError("RealSense video producer failed") from self._error

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            import pyrealsense2 as rs

            pipeline = rs.pipeline()
            config = rs.config()
            if self._config.device is not None:
                config.enable_device(self._config.device)
            config.enable_stream(
                rs.stream.color,
                self._config.width,
                self._config.height,
                rs.format.rgb8,
                self._config.fps,
            )
            pipeline.start(config)
            self._ready_event.set()
            try:
                while not self._stop_event.is_set():
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    rgb = np.ascontiguousarray(np.asanyarray(color_frame.get_data()), dtype=np.uint8)
                    self._provider.push_video_frame(rgb)
            finally:
                pipeline.stop()
        except BaseException as exc:
            self._error = exc
            self._ready_event.set()
            logger.exception("RealSense Pico video producer failed")


class _MujocoCameraVideoProducer(_VideoProducer):
    def __init__(self, provider: Any, config: PicoVideoConfig, robot: Any | None) -> None:
        self._provider = provider
        self._config = config
        self._robot = robot
        self._renderer: Any | None = None
        self._next_frame_time = 0.0
        self._camera_name = "d435i_rgb"

    def start(self) -> None:
        if self._robot is None:
            raise RuntimeError("MuJoCo Pico video requires a robot instance")
        model = getattr(self._robot, "model", None)
        if model is None:
            raise RuntimeError("MuJoCo Pico video requires robot.model")
        import mujoco

        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self._camera_name)
        if camera_id < 0:
            raise ValueError(f"MuJoCo camera '{self._camera_name}' not found")
        self._renderer = mujoco.Renderer(model, height=self._config.height, width=self._config.width)

    def tick(self) -> None:
        if self._renderer is None:
            return
        now = time.monotonic()
        if now < self._next_frame_time:
            return
        data = getattr(self._robot, "data", None)
        if data is None:
            raise RuntimeError("MuJoCo Pico video requires robot.data")
        self._renderer.update_scene(data, camera=self._camera_name)
        frame = np.ascontiguousarray(self._renderer.render(), dtype=np.uint8)
        self._provider.push_video_frame(frame)
        self._next_frame_time = now + 1.0 / float(self._config.fps)

    def stop(self) -> None:
        renderer = self._renderer
        self._renderer = None
        if renderer is not None:
            renderer.close()
