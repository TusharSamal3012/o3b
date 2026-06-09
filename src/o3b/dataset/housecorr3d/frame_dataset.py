"""HouseCorr3DFrame — per-frame, per-object dataset built on top of the
raw Omni6DPose directory structure (ROPE for real, SOPE for synthetic).

Each item is one valid object instance visible in one scene frame, returned as
a FrameObject with rgb, depth, mask, cam_intr4x4, cam_tform4x4_obj, category.

Index layout (frames.db):
    frame_id         TEXT PRIMARY KEY  -- e.g. "real/scene001/0000/2"
    scene_name       TEXT              -- e.g. "scene001" or "patch01_scene002"
    object_idx       INTEGER           -- 0-based index within the frame's objects list
    split            TEXT              -- "train" | "test"
    data_type        TEXT              -- "real" | "synthetic"
    category         TEXT
    object_id        TEXT              -- oid, e.g. "omniobject3d-mug_001"
    rgb_path         TEXT              -- relative to path_raw
    depth_path       TEXT              -- relative to path_raw
    mask_path        TEXT              -- relative to path_raw (exr with per-instance channels)
    cam_intr4x4      TEXT              -- JSON 4x4
    cam_tform4x4_obj TEXT              -- JSON 4x4  (camera-from-object SE3)
    is_valid         INTEGER           -- 1 = valid

Filtering via DatasetConfig:
    split:           "train" | "test" | "all"   (default "train")
    filter_is_real:  True  → real only  |  False → synthetic only (default False)
    categories:      list[str] | None
    filter_count_max: int | None
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import torch

from o3b.dataset.dataset import register_dataset, ConfigurableDataset, DatasetConfig, ItemType
from o3b.data.datatypes import FrameObject


@register_dataset("HouseCorr3DFrame")
class HouseCorr3DFrame(ConfigurableDataset):

    categories: tuple = ()   # populated from enum at module import

    @classmethod
    def _path_raw(cls, cfg: DatasetConfig) -> Path:
        return cfg.path_raw or cfg.root

    @classmethod
    def _path_preprocess(cls, cfg: DatasetConfig) -> Path:
        return cfg.path_preprocess or cfg.root

    # ── setup ─────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._frame_rows_id)

    def _setup(self) -> None:
        self._frame_rows: list[dict] = []
        self._frame_rows_id: list[int] = []

        db_path = self._path_preprocess(self.cfg) / "frames.db"
        if not db_path.exists():
            return

        con = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            cur = con.cursor()
            self._frame_rows = [dict(r) for r in cur.execute("SELECT * FROM frames").fetchall()]

            split     = self.cfg.split or "train"
            is_real   = self.cfg.filter_is_real
            cats      = self.cfg.categories
            limit     = self.cfg.filter_count_max

            split_clause = "" if split == "all" else f" AND split = '{split}'"
            real_clause  = (
                " AND data_type = 'real'" if is_real
                else " AND data_type = 'synthetic'"
            )
            limit_clause = f" LIMIT {limit}" if limit else ""

            if cats:
                placeholders = ", ".join("?" * len(cats))
                cat_clause   = f" AND category IN ({placeholders})"
                params: list = list(cats)
            else:
                cat_clause, params = "", []

            rows = cur.execute(
                f"SELECT rowid FROM frames"
                f" WHERE is_valid = 1{split_clause}{real_clause}{cat_clause}{limit_clause}",
                params,
            ).fetchall()
            # rowid is 1-based; our list is 0-based
            self._frame_rows_id = [r[0] - 1 for r in rows]
        finally:
            con.close()

    # ── item loading ──────────────────────────────────────────────────────────

    def _load_frame_object(self, idx: int) -> FrameObject:
        row = self._frame_rows[self._frame_rows_id[idx]]

        path_raw = self._path_raw(self.cfg)
        mods     = self.cfg.modalities  # None = all

        def _want(name: str) -> bool:
            return mods is None or name in mods

        rgb = None
        if _want("rgb") and row.get("rgb_path"):
            rgb = _load_image_tensor(path_raw / row["rgb_path"])

        depth = None
        if _want("depth") and row.get("depth_path"):
            depth = _load_depth_tensor(path_raw / row["depth_path"])

        mask = None
        if _want("mask") and row.get("mask_path"):
            mask = _load_mask_tensor(path_raw / row["mask_path"], row["object_idx"])

        cam_intr4x4 = None
        if _want("cam_intr4x4") and row.get("cam_intr4x4"):
            cam_intr4x4 = torch.tensor(json.loads(row["cam_intr4x4"]), dtype=torch.float32)

        cam_tform4x4_obj = None
        if _want("cam_tform4x4_obj") and row.get("cam_tform4x4_obj"):
            cam_tform4x4_obj = torch.tensor(
                json.loads(row["cam_tform4x4_obj"]), dtype=torch.float32
            )

        category    = row.get("category") if _want("category") else None
        object_id   = row.get("object_id", "")
        frame_id    = row.get("frame_id", "")

        return FrameObject(
            frame_id         = frame_id,
            frame_object_id  = frame_id,
            object_id        = object_id,
            rgb              = rgb,
            depth            = depth,
            fo_mask          = mask,
            cam_intr4x4      = cam_intr4x4,
            cam_tform4x4_obj = cam_tform4x4_obj,
            category         = category,
        )

    # ── CLI hooks ─────────────────────────────────────────────────────────────

    @classmethod
    def fetch(cls, cfg: DatasetConfig, *, url: Optional[str] = None) -> None:
        from o3b.dataset.housecorr3d.dataset import HouseCorr3D
        HouseCorr3D.fetch(cfg, url=url)

    @classmethod
    def index(cls, cfg: DatasetConfig, *, db: Optional[Path] = None) -> None:
        path_raw        = cls._path_raw(cfg)
        path_preprocess = cls._path_preprocess(cfg)
        db_path         = db or path_preprocess / "frames.db"

        print(f"path_raw        : {path_raw}")
        print(f"path_preprocess : {path_preprocess}")
        print(f"db              : {db_path}")

        rope_root = path_raw / "ROPE"
        sope_root = path_raw / "SOPE"

        if not rope_root.exists() and not sope_root.exists():
            print(
                f"ERROR: neither {rope_root} nor {sope_root} found. "
                "Run 'fetch' first.",
                file=sys.stderr,
            )
            sys.exit(1)

        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Remove stale WAL/SHM files so we get a clean exclusive lock immediately.
        for suffix in ("-wal", "-shm"):
            stale = db_path.with_name(db_path.name + suffix)
            stale.unlink(missing_ok=True)
        con = sqlite3.connect(db_path, timeout=60)
        cur = con.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("DROP TABLE IF EXISTS frames")
        cur.execute("""
            CREATE TABLE frames (
                frame_id         TEXT PRIMARY KEY,
                scene_name       TEXT    NOT NULL,
                object_idx       INTEGER NOT NULL,
                split            TEXT    NOT NULL,
                data_type        TEXT    NOT NULL,
                category         TEXT,
                object_id        TEXT,
                rgb_path         TEXT,
                depth_path       TEXT,
                mask_path        TEXT,
                cam_intr4x4      TEXT,
                cam_tform4x4_obj TEXT,
                is_valid         INTEGER DEFAULT 1
            )
        """)

        from tqdm import tqdm

        total = 0

        # ── real data: ROPE (always test split) ───────────────────────────────
        if rope_root.exists():
            scene_dirs = sorted(d for d in rope_root.iterdir() if d.is_dir())
            print(f"\nIndexing ROPE ({len(scene_dirs)} scenes, split=test, data_type=real)")
            bar = tqdm(scene_dirs, unit="scene", desc="ROPE")
            for scene_dir in bar:
                n = _index_scene(
                    cur       = cur,
                    scene_dir = scene_dir,
                    scene_name= scene_dir.name,
                    split     = "test",
                    data_type = "real",
                    path_raw  = path_raw,
                )
                total += n
                bar.set_postfix(rows=total)

        # ── synthetic data: SOPE ─────────────────────────────────────────────
        if sope_root.exists():
            patch_dirs = sorted(d for d in sope_root.iterdir() if d.is_dir())
            # collect all scene dirs upfront so tqdm can show total count
            sope_tasks: list[tuple[Path, str, str]] = []
            for patch_dir in patch_dirs:
                for split_name in ("train", "test"):
                    split_dir = patch_dir / split_name
                    if not split_dir.exists():
                        continue
                    for kind_dir in (d for d in split_dir.iterdir() if d.is_dir()):
                        for scene_dir in sorted(d for d in kind_dir.iterdir() if d.is_dir()):
                            scene_name = f"{patch_dir.name}_{scene_dir.name}"
                            sope_tasks.append((scene_dir, scene_name, split_name))

            print(f"\nIndexing SOPE ({len(patch_dirs)} patches, {len(sope_tasks)} scenes, data_type=synthetic)")
            bar = tqdm(sope_tasks, unit="scene", desc="SOPE")
            for scene_dir, scene_name, split_name in bar:
                n = _index_scene(
                    cur        = cur,
                    scene_dir  = scene_dir,
                    scene_name = scene_name,
                    split      = split_name,
                    data_type  = "synthetic",
                    path_raw   = path_raw,
                )
                total += n
                bar.set_postfix(rows=total)

        con.commit()
        con.close()
        print(f"\nDone. {total} frame-object rows indexed → {db_path}")

    @classmethod
    def visualize(
        cls,
        cfg: DatasetConfig,
        *,
        db: Optional[Path] = None,
        limit: int = 20,
        object_id: Optional[str] = None,
        render: bool = False,
        render_frames: int = 0,
        renderer: str = "pyrender",
        debug: bool = False,
        **_,
    ) -> None:
        path_preprocess = cls._path_preprocess(cfg)
        db_path = db or path_preprocess / "frames.db"
        if not db_path.exists():
            print(f"No index found at {db_path}. Run 'index' first.", file=sys.stderr)
            sys.exit(1)

        dataset = cls(cfg)
        if not dataset._frame_rows_id:
            print("No frames found matching the current config filters.")
            return

        total = len(dataset._frame_rows_id)
        if object_id:
            dataset._frame_rows_id = [
                i for i in dataset._frame_rows_id
                if dataset._frame_rows[i].get("object_id") == object_id
            ]
        if limit < len(dataset._frame_rows_id):
            dataset._frame_rows_id = dataset._frame_rows_id[:limit]

        print(f"Showing {len(dataset._frame_rows_id)} / {total} frames  {db_path}\n")
        for i in dataset._frame_rows_id:
            row = dataset._frame_rows[i]
            print(
                f"  {row['frame_id']:<60}"
                f"  cat={row.get('category','?'):<20}"
                f"  split={row['split']}  {row['data_type']}"
            )


# ── indexing helpers ──────────────────────────────────────────────────────────

def _index_scene(
    cur,
    scene_dir: Path,
    scene_name: str,
    split: str,
    data_type: str,
    path_raw: Path,
) -> int:
    """Insert frame-object rows for one scene directory. Returns rows inserted."""
    frame_ids_color = [
        p.stem[: -len("_color")]
        for p in scene_dir.iterdir()
        if p.name.endswith("_color.png")
    ]
    n = 0
    for frame_id_raw in sorted(frame_ids_color):
        meta_path = scene_dir / f"{frame_id_raw}_meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue

        cam_meta   = meta.get("camera", {})
        intrinsics = cam_meta.get("intrinsics", {})

        try:
            from o3b.dataset.housecorr3d._frame_utils import (
                build_cam_intr4x4,
                build_cam_tform4x4_obj,
            )
            cam_intr4x4_list      = build_cam_intr4x4(intrinsics, scene_dir / f"{frame_id_raw}_color.png")
            cam_tform4x4_world    = build_cam_tform4x4_obj(cam_meta)
        except Exception:
            cam_intr4x4_list   = None
            cam_tform4x4_world = None

        rgb_path   = scene_dir / f"{frame_id_raw}_color.png"
        depth_path = scene_dir / f"{frame_id_raw}_depth.exr"
        mask_path  = scene_dir / f"{frame_id_raw}_mask.exr"

        rgb_rpath   = str(rgb_path.relative_to(path_raw))   if rgb_path.exists()   else None
        depth_rpath = str(depth_path.relative_to(path_raw)) if depth_path.exists() else None
        mask_rpath  = str(mask_path.relative_to(path_raw))  if mask_path.exists()  else None

        for obj_idx, (obj_name, obj) in enumerate(meta.get("objects", {}).items()):
            is_valid  = int(bool(obj.get("is_valid", True)))
            category  = obj.get("meta", {}).get("class_name")
            object_id = obj.get("meta", {}).get("oid") or obj_name

            # per-object cam_tform4x4_obj: uses the object's own quaternion/translation
            try:
                from o3b.dataset.housecorr3d._frame_utils import build_obj_cam_tform
                cam_tform4x4_obj_list = build_obj_cam_tform(obj)
            except Exception:
                cam_tform4x4_obj_list = cam_tform4x4_world

            frame_id = f"{data_type}/{scene_name}/{frame_id_raw}/{obj_idx}"

            cur.execute(
                """
                INSERT OR IGNORE INTO frames
                    (frame_id, scene_name, object_idx, split, data_type,
                     category, object_id,
                     rgb_path, depth_path, mask_path,
                     cam_intr4x4, cam_tform4x4_obj, is_valid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    frame_id, scene_name, obj_idx, split, data_type,
                    category, object_id,
                    rgb_rpath, depth_rpath, mask_rpath,
                    json.dumps(cam_intr4x4_list)   if cam_intr4x4_list   is not None else None,
                    json.dumps(cam_tform4x4_obj_list) if cam_tform4x4_obj_list is not None else None,
                    is_valid,
                ),
            )
            n += 1
    return n


# ── modality loaders ─────────────────────────────────────────────────────────

def _load_image_tensor(path: Path) -> Optional[torch.Tensor]:
    """Load PNG/JPEG → (3, H, W) float32 in [0, 1]."""
    if not path.exists():
        return None
    try:
        import torchvision.io as tio
        img = tio.read_image(str(path))          # (C, H, W) uint8
        return img.float() / 255.0
    except Exception:
        return None


def _load_depth_tensor(path: Path) -> Optional[torch.Tensor]:
    """Load depth EXR → (H, W) float32."""
    if not path.exists():
        return None
    try:
        import imageio.v3 as iio
        import numpy as np
        arr = iio.imread(str(path), plugin="EXR-FI")   # (H, W) or (H, W, C)
        arr = np.array(arr, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return torch.from_numpy(arr)
    except Exception:
        return None


def _load_mask_tensor(path: Path, object_idx: int) -> Optional[torch.Tensor]:
    """Load mask EXR for a specific object index → (H, W) bool."""
    if not path.exists():
        return None
    try:
        import imageio.v3 as iio
        import numpy as np
        arr = iio.imread(str(path), plugin="EXR-FI")   # (H, W, N_objs)
        arr = np.array(arr, dtype=np.float32)
        if arr.ndim == 2:
            return torch.from_numpy(arr > 0)
        if object_idx < arr.shape[-1]:
            return torch.from_numpy(arr[..., object_idx] > 0)
        return None
    except Exception:
        return None
