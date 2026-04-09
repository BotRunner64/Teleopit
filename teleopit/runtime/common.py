from __future__ import annotations

from pathlib import Path
from typing import Any

_MISSING = object()

_VALID_VIEWERS = frozenset({"mocap", "retarget", "sim2sim"})


def cfg_get(cfg: Any, key: str, default: Any = _MISSING) -> Any:
    """Read *key* from *cfg* (dict, OmegaConf, or plain object).

    When *default* is omitted the key is **required** — a `KeyError` is raised
    if it cannot be found.
    """
    if isinstance(cfg, dict):
        if default is _MISSING:
            return cfg[key]
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        value = cfg.get(key)
        if value is not None:
            return value
        if default is _MISSING:
            raise KeyError(key)
        return default
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if default is _MISSING:
        raise KeyError(key)
    return default


def cfg_set(cfg: Any, key: str, value: Any) -> None:
    if isinstance(cfg, dict):
        cfg[key] = value
        return
    if hasattr(cfg, "__setitem__"):
        try:
            cfg[key] = value
            return
        except Exception:
            pass
    setattr(cfg, key, value)


def parse_viewers(cfg: Any) -> set[str]:
    if cfg_get(cfg, "viewer", None) is not None:
        raise ValueError("Legacy config key 'viewer' has been removed. Use 'viewers' instead.")

    viewers_raw = cfg_get(cfg, "viewers", None)
    if viewers_raw is None:
        return set()

    if hasattr(viewers_raw, "__iter__") and not isinstance(viewers_raw, str):
        viewers = {str(v).strip().lower() for v in viewers_raw if str(v).strip()}
    else:
        raw = str(viewers_raw).strip().lower()
        if raw == "all":
            return set(_VALID_VIEWERS)
        if raw in ("none", "false", ""):
            return set()
        raw = raw.strip("'\"[]")
        viewers = {token.strip() for token in raw.split(",") if token.strip()}

    invalid = viewers.difference(_VALID_VIEWERS)
    if invalid:
        valid = ", ".join(sorted(_VALID_VIEWERS))
        raise ValueError(
            f"Unsupported viewers {sorted(invalid)}. Use a subset of [{valid}] or 'all'/'none'."
        )
    return viewers


def require_section(cfg: Any, key: str) -> Any:
    section = cfg_get(cfg, key, None)
    if section is None:
        raise ValueError(f"cfg must include a '{key}' section")
    return section


def normalize_path_in_cfg(
    cfg: Any,
    key: str,
    *,
    base_dir: Path,
    required: bool = False,
    must_exist: bool = True,
    missing_message: str | None = None,
) -> Path | None:
    raw = str(cfg_get(cfg, key, "")).strip()
    if not raw or raw == "None":
        if required:
            raise ValueError(missing_message or f"{key} must be set")
        return None

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{key} resolved to {path} which does not exist.")

    cfg_set(cfg, key, str(path))
    return path


def parse_nonnegative_int(value: object, *, field_name: str, default: Any = _MISSING) -> int:
    """Parse *value* as a non-negative integer.

    If *value* is ``None``/``""``/``"null"`` and *default* is provided, return *default*.
    Otherwise raise ``ValueError`` for invalid values.
    """
    if value in (None, "", "null"):
        if default is not _MISSING:
            return int(default)
        raise ValueError(f"{field_name} must be a non-negative integer, got {value!r}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a non-negative integer, got {value!r}")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{field_name} must be >= 0, got {value!r}")
    return parsed


def parse_alpha(value: object, *, field_name: str, default: Any = _MISSING) -> float:
    """Parse *value* as a smoothing alpha in ``(0, 1]``.

    If *value* is ``None``/``""``/``"null"`` and *default* is provided, return *default*.
    """
    if value in (None, "", "null"):
        if default is not _MISSING:
            return float(default)
        raise ValueError(f"{field_name} must be in (0, 1], got {value!r}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be in (0, 1], got {value!r}")
    parsed = float(value)
    if parsed <= 0.0 or parsed > 1.0:
        raise ValueError(f"{field_name} must be in (0, 1], got {value!r}")
    return parsed


def parse_optional_nonnegative_int(value: object, *, field_name: str) -> int | None:
    """Parse *value* as an optional non-negative integer.

    Returns ``None`` for ``None``/``""``/``"null"`` inputs.
    """
    if value in (None, "", "null"):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a non-negative integer, got {value!r}")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{field_name} must be >= 0, got {value!r}")
    return parsed
