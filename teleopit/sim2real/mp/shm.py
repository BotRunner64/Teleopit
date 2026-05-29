"""Shared-memory ring buffer for large sim2real video frames."""

from __future__ import annotations

from multiprocessing import shared_memory
from typing import Any

import numpy as np
from numpy.typing import NDArray

from teleopit.sim2real.mp.messages import SharedFrameDescriptor


class SharedFrameRingWriter:
    """Owns a fixed-size ring of frame slots in shared memory."""

    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        dtype: str | np.dtype[Any] = np.uint8,
        slots: int = 3,
        name: str | None = None,
    ) -> None:
        if slots <= 0:
            raise ValueError(f"slots must be positive, got {slots}")
        dtype_np = np.dtype(dtype)
        shape_tuple = tuple(int(dim) for dim in shape)
        if not shape_tuple or any(dim <= 0 for dim in shape_tuple):
            raise ValueError(f"shape must contain positive dimensions, got {shape}")

        self.shape = shape_tuple
        self.dtype = dtype_np
        self.slots = int(slots)
        self._slot_size = int(np.prod(shape_tuple)) * dtype_np.itemsize
        self._shm = shared_memory.SharedMemory(
            name=name,
            create=True,
            size=self._slot_size * self.slots,
        )
        self._seq = 0

    @property
    def name(self) -> str:
        return self._shm.name

    def write(self, frame: NDArray[np.generic], *, timestamp_s: float) -> SharedFrameDescriptor:
        frame_arr = np.asarray(frame, dtype=self.dtype)
        if tuple(frame_arr.shape) != self.shape:
            raise ValueError(f"frame shape {frame_arr.shape} does not match shared ring shape {self.shape}")
        slot = self._seq % self.slots
        view = self._slot_view(slot)
        np.copyto(view, np.ascontiguousarray(frame_arr))
        descriptor = SharedFrameDescriptor(
            shm_name=self._shm.name,
            slot=slot,
            seq=self._seq,
            timestamp_s=float(timestamp_s),
            shape=self.shape,
            dtype=str(self.dtype),
            slots=self.slots,
        )
        self._seq += 1
        return descriptor

    def close(self, *, unlink: bool = True) -> None:
        self._shm.close()
        if unlink:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass

    def _slot_view(self, slot: int) -> NDArray[np.generic]:
        if slot < 0 or slot >= self.slots:
            raise ValueError(f"slot must be in [0, {self.slots}), got {slot}")
        start = slot * self._slot_size
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self._shm.buf, offset=start)


class SharedFrameRingReader:
    """Attaches to shared frame rings lazily and reads descriptor-selected slots."""

    def __init__(self) -> None:
        self._rings: dict[str, shared_memory.SharedMemory] = {}

    def read(self, descriptor: SharedFrameDescriptor, *, copy: bool = False) -> NDArray[np.generic]:
        shm = self._rings.get(descriptor.shm_name)
        if shm is None:
            shm = shared_memory.SharedMemory(name=descriptor.shm_name)
            self._rings[descriptor.shm_name] = shm

        dtype = np.dtype(descriptor.dtype)
        shape = tuple(int(dim) for dim in descriptor.shape)
        slot_size = int(np.prod(shape)) * dtype.itemsize
        if descriptor.slot < 0 or descriptor.slot >= descriptor.slots:
            raise ValueError(f"descriptor slot {descriptor.slot} out of range for {descriptor.slots} slots")
        view = np.ndarray(shape, dtype=dtype, buffer=shm.buf, offset=descriptor.slot * slot_size)
        if copy:
            return view.copy()
        return view

    def close(self) -> None:
        for shm in self._rings.values():
            shm.close()
        self._rings.clear()
