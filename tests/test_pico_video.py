from __future__ import annotations

import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import teleopit.inputs.pico_video as pico_video
from teleopit.inputs.pico_video import PicoVideoRuntime, bridge_video_source, parse_pico_video_config


class _FrameSink:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def push_video_frame(self, frame: np.ndarray) -> int:
        self.frames.append(frame)
        return len(self.frames)


def test_pico_video_config_maps_camera_sources_to_bridge_frames() -> None:
    cfg = parse_pico_video_config(
        {"video": {"enabled": True, "source": "realsense", "width": 320, "height": 240, "fps": 15}}
    )

    assert cfg.enabled is True
    assert cfg.source == "realsense"
    assert cfg.width == 320
    assert bridge_video_source(cfg) == "frames"

    test_pattern_cfg = parse_pico_video_config({"video": {"enabled": True, "source": "test-pattern"}})
    assert bridge_video_source(test_pattern_cfg) == "test-pattern"


def test_pico_video_config_rejects_enabled_unknown_source() -> None:
    with pytest.raises(ValueError, match="input.video.source"):
        parse_pico_video_config({"video": {"enabled": True, "source": "webcam"}})


def test_realsense_video_runtime_pushes_rgb_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_rs = ModuleType("pyrealsense2")
    fake_rs.stream = SimpleNamespace(color="color")
    fake_rs.format = SimpleNamespace(rgb8="rgb8")

    class FakeConfig:
        def enable_device(self, _device: str) -> None:
            pass

        def enable_stream(self, *_args: object) -> None:
            pass

    class FakeColorFrame:
        def get_data(self) -> np.ndarray:
            return np.ones((2, 3, 3), dtype=np.uint8)

    class FakeFrames:
        def get_color_frame(self) -> FakeColorFrame:
            return FakeColorFrame()

    class FakePipeline:
        def start(self, _config: object) -> None:
            pass

        def wait_for_frames(self) -> FakeFrames:
            time.sleep(0.005)
            return FakeFrames()

        def stop(self) -> None:
            pass

    fake_rs.config = FakeConfig
    fake_rs.pipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)

    sink = _FrameSink()
    config = parse_pico_video_config({"video": {"enabled": True, "source": "realsense", "width": 3, "height": 2}})
    runtime = PicoVideoRuntime(provider=sink, config=config, mode="sim2real")

    runtime.start()
    time.sleep(0.03)
    runtime.stop()

    assert sink.frames
    assert sink.frames[-1].shape == (2, 3, 3)
    assert sink.frames[-1].dtype == np.uint8


def test_video_runtime_stops_producer_after_startup_error(monkeypatch: pytest.MonkeyPatch) -> None:
    stopped = False

    class FailingProducer:
        def __init__(self, _provider: object, _config: object) -> None:
            pass

        def start(self) -> None:
            raise TimeoutError("camera did not become ready")

        def tick(self) -> None:
            pass

        def stop(self) -> None:
            nonlocal stopped
            stopped = True

    monkeypatch.setattr(pico_video, "_RealSenseVideoProducer", FailingProducer)

    sink = _FrameSink()
    config = parse_pico_video_config({"video": {"enabled": True, "source": "realsense"}})
    runtime = PicoVideoRuntime(provider=sink, config=config, mode="sim2real")

    runtime.start()

    assert stopped is True
    assert sink.frames == []


def test_video_runtime_stops_producer_before_reraising_tick_error(monkeypatch: pytest.MonkeyPatch) -> None:
    stopped = False

    class FailingProducer:
        def __init__(self, _provider: object, _config: object) -> None:
            pass

        def start(self) -> None:
            pass

        def tick(self) -> None:
            raise RuntimeError("camera lost")

        def stop(self) -> None:
            nonlocal stopped
            stopped = True

    monkeypatch.setattr(pico_video, "_RealSenseVideoProducer", FailingProducer)

    sink = _FrameSink()
    config = parse_pico_video_config(
        {"video": {"enabled": True, "source": "realsense", "fail_on_error": True}}
    )
    runtime = PicoVideoRuntime(provider=sink, config=config, mode="sim2real")

    runtime.start()
    with pytest.raises(RuntimeError, match="Pico video pipeline failed"):
        runtime.tick()

    assert stopped is True


def test_mujoco_video_runtime_renders_camera_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mujoco = ModuleType("mujoco")
    fake_mujoco.mjtObj = SimpleNamespace(mjOBJ_CAMERA="camera")
    fake_mujoco.mj_name2id = lambda _model, _obj, _name: 0

    class FakeRenderer:
        def __init__(self, _model: object, *, height: int, width: int) -> None:
            self.frame = np.zeros((height, width, 3), dtype=np.uint8)

        def update_scene(self, _data: object, *, camera: str) -> None:
            assert camera == "d435i_rgb"

        def render(self) -> np.ndarray:
            return self.frame

        def close(self) -> None:
            pass

    fake_mujoco.Renderer = FakeRenderer
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)

    sink = _FrameSink()
    robot = SimpleNamespace(model=object(), data=object())
    config = parse_pico_video_config({"video": {"enabled": True, "source": "mujoco", "width": 4, "height": 3}})
    runtime = PicoVideoRuntime(provider=sink, config=config, mode="sim2sim", robot=robot)

    runtime.start()
    runtime.tick()
    runtime.stop()

    assert len(sink.frames) == 1
    assert sink.frames[0].shape == (3, 4, 3)
