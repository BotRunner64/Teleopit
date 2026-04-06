from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class _FakeBridge:
    def __init__(self, check_mode_responses, *, release_codes=None) -> None:
        self._check_mode_responses = list(check_mode_responses)
        self._release_codes = list(release_codes or [])
        self.release_calls = 0

    def wait_for_state(self, _timeout: float) -> bool:
        return True

    def check_mode(self):
        if not self._check_mode_responses:
            raise AssertionError("check_mode called unexpectedly")
        return self._check_mode_responses.pop(0)

    def release_mode(self) -> int:
        self.release_calls += 1
        if self._release_codes:
            return self._release_codes.pop(0)
        return 0

    def stop_publish(self) -> None:
        pass


def _install_bridge_module(monkeypatch: pytest.MonkeyPatch, bridge: _FakeBridge) -> None:
    fake_module = ModuleType("g1_bridge_sdk")
    fake_module.G1Bridge = lambda *_args, **_kwargs: bridge
    monkeypatch.setitem(sys.modules, "g1_bridge_sdk", fake_module)


def test_enter_debug_mode_fails_on_check_mode_rpc_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from teleopit.sim2real import unitree_g1

    bridge = _FakeBridge([(1, "")])
    _install_bridge_module(monkeypatch, bridge)
    monkeypatch.setattr(unitree_g1.time, "sleep", lambda _seconds: None)

    robot = unitree_g1.UnitreeG1Robot({"network_interface": "eth0"})

    assert robot.enter_debug_mode() is False
    assert bridge.release_calls == 0


def test_enter_debug_mode_fails_on_release_mode_rpc_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from teleopit.sim2real import unitree_g1

    bridge = _FakeBridge([(0, "ai")], release_codes=[7])
    _install_bridge_module(monkeypatch, bridge)
    monkeypatch.setattr(unitree_g1.time, "sleep", lambda _seconds: None)

    robot = unitree_g1.UnitreeG1Robot({"network_interface": "eth0"})

    assert robot.enter_debug_mode() is False
    assert bridge.release_calls == 1
