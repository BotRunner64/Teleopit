from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class RolloutTraceWriter:
    def __init__(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        self.path = Path(path).expanduser()
        self.metadata = dict(metadata or {})
        self._records: list[dict[str, np.ndarray]] = []

    def add_step(self, **fields: Any) -> None:
        record: dict[str, np.ndarray] = {}
        for key, value in fields.items():
            if value is None:
                continue
            if isinstance(value, np.ndarray):
                record[key] = value.copy()
                continue
            if isinstance(value, (list, tuple, int, float, bool, np.generic)):
                record[key] = np.asarray(value)
                continue
            record[key] = np.asarray(value)
        self._records.append(record)

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, np.ndarray] = {}
        all_keys = sorted({key for record in self._records for key in record})
        for key in all_keys:
            values = [record.get(key) for record in self._records]
            if any(value is None for value in values):
                continue
            arrays = [np.asarray(value) for value in values if value is not None]
            try:
                payload[key] = np.stack(arrays, axis=0)
            except ValueError:
                payload[key] = np.asarray(arrays, dtype=object)

        payload["metadata_json"] = np.asarray(
            json.dumps(self.metadata, sort_keys=True),
            dtype=np.str_,
        )
        np.savez_compressed(self.path, **payload)
        return self.path
