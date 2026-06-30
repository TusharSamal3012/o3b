"""HuggingFace-sharding helpers for ConfigurableDataset.

A "sharded" dataset is a materialised cache of the items a dataset would
otherwise build lazily from raw files.  Each item (FrameObject, Object,
ObjectPair, …) is serialised into a flat, Arrow-friendly record, stored via
``datasets.Dataset.save_to_disk`` (sharded), and read back with
``load_from_disk``.

Serialisation is generic and recursive:
  - ``torch.Tensor``        → {"__t__": 1, "shape": [...], "dtype": "...", "data": [...]}
  - dataclass instance      → {"__dc__": "ClassName", "fields": {name: encoded, ...}}
  - list / tuple            → {"__l__": [encoded, ...]}
  - str / int / float / None → stored as-is

This keeps nested structures (e.g. an Object's Mesh) fully self-contained in
the shards.  ``datasets`` is imported lazily so the dependency is only required
when sharding is actually used.
"""
from __future__ import annotations

import math
import os
import shutil
import time
from dataclasses import fields, is_dataclass
from pathlib import Path

import torch


# ── dataclass registry (for reconstructing nested values) ──────────────────────

def _dataclass_registry() -> dict:
    from o3b.data.modalities import FrameObject, SceneObject, Object, ObjectPair
    from o3b.data.datatypes.mesh import Mesh
    return {c.__name__: c for c in (FrameObject, SceneObject, Object, ObjectPair, Mesh)}


# ── (de)serialisation ──────────────────────────────────────────────────────────

def _encode(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        t = value.detach().cpu()
        np_array = t.numpy()
        return {
            "__t__": 2,                              # v2: bytes encoding
            "shape": list(t.shape),
            "dtype": str(t.dtype).replace("torch.", ""),
            "data":  np_array.tobytes(),             # raw bytes, O(1) per element
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__dc__":  type(value).__name__,
            "fields":  {f.name: _encode(getattr(value, f.name)) for f in fields(value)},
        }
    if isinstance(value, (list, tuple)):
        return {"__l__": [_encode(v) for v in value]}
    return value


def _decode(value):
    if value is None:
        return None
    if isinstance(value, dict):
        if "__t__" in value:
            import numpy as np
            dtype_str = value["dtype"]
            shape     = value["shape"]
            data      = value["data"]
            if value["__t__"] == 2 and isinstance(data, (bytes, bytearray, memoryview)):
                # v2: raw bytes → numpy → torch (fast path)
                arr = np.frombuffer(data, dtype=np.dtype(dtype_str)).copy()
                return torch.from_numpy(arr.reshape(shape))
            else:
                # v1 legacy: Python list (slow path, kept for backward compatibility)
                torch_dtype = getattr(torch, dtype_str)
                if len(data) == 0:
                    return torch.zeros(shape, dtype=torch_dtype)
                return torch.tensor(data, dtype=torch_dtype).reshape(shape)
        if "__dc__" in value:
            cls    = _dataclass_registry()[value["__dc__"]]
            kwargs = {k: _decode(v) for k, v in value["fields"].items()}
            return cls(**kwargs)
        if "__l__" in value:
            return [_decode(v) for v in value["__l__"]]
    return value


def item_to_record(item) -> dict:
    """Serialise a dataclass item into a flat Arrow-friendly record dict."""
    return {f.name: _encode(getattr(item, f.name)) for f in fields(item)}


def record_to_item(record: dict, item_cls):
    """Reconstruct a dataclass item of type ``item_cls`` from a stored record."""
    kwargs = {k: _decode(v) for k, v in record.items()}
    return item_cls(**kwargs)


# ── disk I/O ────────────────────────────────────────────────────────────────────

_WRITE_BATCH_SIZE = 1048


def build_sharded_dataset(records: list[dict]):
    """Build an in-memory HuggingFace Dataset from a list of records."""
    from datasets import Dataset as HFDataset
    return HFDataset.from_list(records)


def build_sharded_dataset_from_generator(gen_fn, writer_batch_size: int = 1000):
    """Build a HuggingFace Dataset by streaming records from a generator.

    Processes ``writer_batch_size`` records at a time so peak memory is
    bounded to that many items rather than the full dataset.  ``gen_fn``
    is a zero-argument callable that returns an iterator of record dicts.
    """
    from datasets import Dataset as HFDataset
    return HFDataset.from_generator(gen_fn, num_proc=1, writer_batch_size=writer_batch_size)


def _remove_dir(path: Path) -> None:
    """Remove a directory robustly, tolerating NFS ``.nfsXXXX`` leftovers.

    On NFS, deleting a file that is still held open (e.g. memory-mapped by a
    prior ``load_from_disk``, or a stale handle from an interrupted run) leaves
    a ``.nfsXXXX`` placeholder, so a plain ``rmtree`` fails the final ``rmdir``
    with ``OSError: [Errno 39] Directory not empty``.

    To avoid blocking on the live path, the directory is first *renamed* out of
    the way (an atomic metadata op that works even with open files), then the
    renamed copy is deleted best-effort; any ``.nfs`` leftovers there are
    harmless and get cleaned up once the holding process exits.
    """
    if not path.exists():
        return
    trash = path.with_name(f"{path.name}.trash-{os.getpid()}-{time.time_ns()}")
    try:
        os.rename(str(path), str(trash))
    except OSError:
        # rename failed (e.g. cross-device); fall back to in-place rmtree
        shutil.rmtree(str(path), ignore_errors=True)
        return
    shutil.rmtree(str(trash), ignore_errors=True)


def write_sharded_dataset(hf_dataset, path) -> None:
    """Save a HuggingFace Dataset to ``path`` as Arrow shards (overwrites)."""
    path = Path(path)
    _remove_dir(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    num_shards = max(1, int(math.ceil(len(hf_dataset) / _WRITE_BATCH_SIZE)))
    hf_dataset.save_to_disk(str(path), num_shards=num_shards)


def read_sharded_dataset(path):
    """Load a sharded HuggingFace Dataset from ``path``."""
    from datasets import load_from_disk
    return load_from_disk(str(path))
