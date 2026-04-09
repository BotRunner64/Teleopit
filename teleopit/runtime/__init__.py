from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "InferenceComponents",
    "MocapComponents",
    "build_inference_components",
    "build_sim2real_mocap_components",
    "build_simulation_cfg",
    "cfg_get",
    "cfg_set",
    "parse_viewers",
    "validate_policy_path",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "InferenceComponents": (".factory", "InferenceComponents"),
    "MocapComponents": (".factory", "MocapComponents"),
    "build_inference_components": (".factory", "build_inference_components"),
    "build_sim2real_mocap_components": (".factory", "build_sim2real_mocap_components"),
    "build_simulation_cfg": (".factory", "build_simulation_cfg"),
    "cfg_get": (".common", "cfg_get"),
    "cfg_set": (".common", "cfg_set"),
    "parse_viewers": (".common", "parse_viewers"),
    "validate_policy_path": (".cli", "validate_policy_path"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
