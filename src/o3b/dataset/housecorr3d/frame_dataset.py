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

            split      = self.cfg.split or "train"
            is_real    = self.cfg.filter_is_real
            cats       = self.cfg.categories
            limit      = self.cfg.filter_count_max
            has_kpts   = self.cfg.filter_has_kpts

            split_clause = "" if split == "all" else f" AND split = '{split}'"
            if is_real is None:
                real_clause = ""
            elif is_real:
                real_clause = " AND data_type = 'real'"
            else:
                real_clause = " AND data_type = 'synthetic'"
            kpts_clause  = " AND has_kpts = 1" if has_kpts else ""
            limit_clause = f" LIMIT {limit}" if limit else ""

            if cats:
                placeholders = ", ".join("?" * len(cats))
                cat_clause   = f" AND category IN ({placeholders})"
                params: list = list(cats)
            else:
                cat_clause, params = "", []

            rows = cur.execute(
                f"SELECT rowid FROM frames"
                f" WHERE is_valid = 1{split_clause}{real_clause}{kpts_clause}{cat_clause}{limit_clause}",
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

        depth_mask = None
        if _want("depth_mask") and depth is not None:
            depth_mask = (depth > 0)

        mask = None
        if _want("fo_mask") and row.get("mask_path") and row.get("mask_id") is not None:
            mask = _load_mask_tensor(path_raw / row["mask_path"], int(row["mask_id"]))

        cam_intr4x4 = None
        if _want("cam_intr4x4") and row.get("cam_intr4x4"):
            cam_intr4x4 = torch.tensor(json.loads(row["cam_intr4x4"]), dtype=torch.float32)

        cam_tform4x4_obj = None
        if _want("cam_tform4x4_obj") and row.get("cam_tform4x4_obj"):
            cam_tform4x4_obj = torch.tensor(
                json.loads(row["cam_tform4x4_obj"]), dtype=torch.float32
            )

        # cam_bbox2d: derive from fo_mask (xyxy pixel bbox)
        cam_bbox2d = None
        if _want("cam_bbox2d") and mask is not None:
            from o3b.cv.visual.draw import get_bboxs_from_masks
            cam_bbox2d = get_bboxs_from_masks(mask[None])[0].float()  # (4,) xyxy

        # cam_bbox3d: stored as (3,) obj-space side lengths for use with draw_bbox3d
        cam_bbox3d = None
        if _want("cam_bbox3d") and row.get("obj_size3d"):
            cam_bbox3d = torch.tensor(json.loads(row["obj_size3d"]), dtype=torch.float32)

        # obj_kpts3d / obj_kpts3d_mask: load from preprocess dir, scale mesh-units → metres
        obj_kpts3d      = None
        obj_kpts3d_mask = None
        if _want("obj_kpts3d"):
            oid        = row.get("object_id", "")
            kpts_path  = self._path_preprocess(self.cfg) / "obj_kpts3d" / oid / "kpts3d.pt"
            if kpts_path.exists():
                try:
                    data      = torch.load(kpts_path, weights_only=True)  # (K, 4)
                    obj_scale = float(row.get("obj_scale") or 1.0)
                    obj_kpts3d = data[:, :3] * obj_scale               # (K, 3) in metres
                    if _want("obj_kpts3d_mask"):
                        obj_kpts3d_mask = data[:, 3] > 0.5             # (K,) bool
                except Exception:
                    pass

        category    = row.get("category") if _want("category") else None
        object_id   = row.get("object_id", "")
        frame_id    = row.get("frame_id", "")

        return FrameObject(
            frame_id         = frame_id,
            frame_object_id  = frame_id,
            object_id        = object_id,
            rgb              = rgb,
            depth            = depth,
            depth_mask       = depth_mask,
            fo_mask          = mask,
            cam_intr4x4      = cam_intr4x4,
            cam_tform4x4_obj = cam_tform4x4_obj,
            cam_bbox2d       = cam_bbox2d,
            cam_bbox3d       = cam_bbox3d,
            obj_kpts3d       = obj_kpts3d,
            obj_kpts3d_mask  = obj_kpts3d_mask,
            category         = category,
        )

    # ── CLI hooks ─────────────────────────────────────────────────────────────

    @classmethod
    def fetch(cls, cfg: DatasetConfig, *, url: Optional[str] = None) -> None:
        from o3b.dataset.housecorr3d.dataset import HouseCorr3D
        HouseCorr3D.fetch(cfg, url=url)

    @classmethod
    def index(
        cls,
        cfg: DatasetConfig,
        *,
        db: Optional[Path] = None,
        remove: bool = False,
        max_index: Optional[int] = None,
    ) -> None:
        path_raw        = cls._path_raw(cfg)
        path_preprocess = cls._path_preprocess(cfg)
        db_path         = db or path_preprocess / "frames.db"
        # filter_count_max stops indexing once this many filter-matching rows are found.
        # --max (max_index) is an unconditional override (for quick testing without a yaml edit).
        limit       = max_index or cfg.filter_count_max  # None = index everything
        filter_kpts = cfg.filter_has_kpts                # only count kpts rows toward limit
        is_real     = cfg.filter_is_real                 # None = both, True = real only, False = synthetic
        kpts_preprocess = path_preprocess / "obj_kpts3d"

        print(f"path_raw        : {path_raw}")
        print(f"path_preprocess : {path_preprocess}")
        print(f"db              : {db_path}")
        if limit:
            kpts_note = " (kpts rows only)" if filter_kpts else ""
            print(f"filter_count_max: {limit}{kpts_note}  — indexing stops after this many matching rows")
        if is_real is not None:
            print(f"filter_is_real  : {is_real}  ({'real (ROPE) only' if is_real else 'synthetic (SOPE) only'})")

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
        if remove and db_path.exists():
            db_path.unlink()
            print(f"Removed existing index: {db_path}")
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
                mask_id          INTEGER,
                split            TEXT    NOT NULL,
                data_type        TEXT    NOT NULL,
                category         TEXT,
                object_id        TEXT,
                rgb_path         TEXT,
                depth_path       TEXT,
                mask_path        TEXT,
                cam_intr4x4      TEXT,
                cam_tform4x4_obj TEXT,
                obj_size3d       TEXT,
                obj_scale        REAL,
                has_kpts         INTEGER DEFAULT 0,
                is_valid         INTEGER DEFAULT 1
            )
        """)

        from tqdm import tqdm

        COMMIT_EVERY = 50  # flush to disk every N scenes
        total_rows   = 0   # all rows inserted
        matched      = 0   # rows matching the index-time filter (used for limit check)
        scenes_since_commit = 0
        done  = False

        def _remaining():
            """Remaining matching rows allowed in the current scene."""
            return (limit - matched) if limit else None

        def _update(n_total: int, n_match: int) -> bool:
            """Update counters; return True when the limit has been reached."""
            nonlocal total_rows, matched, scenes_since_commit
            total_rows += n_total
            matched    += n_match
            scenes_since_commit += 1
            if scenes_since_commit >= COMMIT_EVERY:
                con.commit()
                scenes_since_commit = 0
            return bool(limit and matched >= limit)

        # ── real data: ROPE (always test split) ───────────────────────────────
        if rope_root.exists() and not done and is_real is not False:
            scene_dirs = sorted(d for d in rope_root.iterdir() if d.is_dir())
            print(f"\nIndexing ROPE ({len(scene_dirs)} scenes, split=test, data_type=real)")
            bar = tqdm(scene_dirs, unit="scene", desc="ROPE")
            for scene_dir in bar:
                n_total, n_match = _index_scene(
                    cur             = cur,
                    scene_dir       = scene_dir,
                    scene_name      = scene_dir.name,
                    split           = "test",
                    data_type       = "real",
                    path_raw        = path_raw,
                    kpts_preprocess = kpts_preprocess,
                    limit           = _remaining(),
                    filter_kpts     = filter_kpts,
                )
                bar.set_postfix(rows=total_rows, matched=matched)
                if _update(n_total, n_match):
                    done = True
                    bar.close()
                    break

        # ── synthetic data: SOPE ─────────────────────────────────────────────
        if sope_root.exists() and not done and is_real is not True:
            patch_dirs = sorted(d for d in sope_root.iterdir() if d.is_dir())
            print(f"\nIndexing SOPE ({len(patch_dirs)} patches, data_type=synthetic)")

            def _sope_scenes():
                """Yield (scene_dir, scene_name, split_name) lazily — no upfront NFS scan."""
                for patch_dir in patch_dirs:
                    for split_name in ("train", "test"):
                        split_dir = patch_dir / split_name
                        if not split_dir.exists():
                            continue
                        for kind_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
                            for scene_dir in sorted(d for d in kind_dir.iterdir() if d.is_dir()):
                                yield scene_dir, f"{patch_dir.name}_{scene_dir.name}", split_name

            bar = tqdm(_sope_scenes(), unit="scene", desc="SOPE")
            for scene_dir, scene_name, split_name in bar:
                n_total, n_match = _index_scene(
                    cur             = cur,
                    scene_dir       = scene_dir,
                    scene_name      = scene_name,
                    split           = split_name,
                    data_type       = "synthetic",
                    path_raw        = path_raw,
                    kpts_preprocess = kpts_preprocess,
                    limit           = _remaining(),
                    filter_kpts     = filter_kpts,
                )
                bar.set_postfix(rows=total_rows, matched=matched)
                if _update(n_total, n_match):
                    bar.close()
                    break

        con.commit()
        con.close()
        print(f"\nDone. {total_rows} rows indexed ({matched} matching filter) → {db_path}")

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
        for seq_idx, i in enumerate(dataset._frame_rows_id):
            row = dataset._frame_rows[i]
            print(
                f"[{seq_idx + 1}/{len(dataset._frame_rows_id)}]"
                f"  {row['frame_id']:<60}"
                f"  cat={row.get('category','?'):<20}"
                f"  split={row['split']}  {row['data_type']}"
            )
            fo = dataset._load_frame_object(seq_idx)
            fo.viz(show=True)


# ── indexing helpers ──────────────────────────────────────────────────────────

def _index_scene(
    cur,
    scene_dir: Path,
    scene_name: str,
    split: str,
    data_type: str,
    path_raw: Path,
    kpts_preprocess: Path,
    limit: Optional[int] = None,
    filter_kpts: bool = False,
) -> tuple[int, int]:
    """Insert frame-object rows for one scene. Returns (n_total, n_matching).

    n_matching counts rows that satisfy the index-time filter:
    - filter_kpts=True  → only rows with has_kpts=1
    - filter_kpts=False → same as n_total

    Stops early once *limit* matching rows have been inserted (None = no limit).
    """
    from o3b.dataset.housecorr3d._frame_utils import (
        build_cam_intr4x4,
        build_cam_tform4x4_obj,
        build_obj_cam_tform,
        _png_size,
    )

    frame_ids_color = sorted(
        p.stem[: -len("_color")]
        for p in scene_dir.iterdir()
        if p.name.endswith("_color.png")
    )

    # Read image dimensions once for the scene (all frames share the same resolution).
    scene_img_size: tuple[int, int] | None = None
    for fid in frame_ids_color:
        p = scene_dir / f"{fid}_color.png"
        if p.exists():
            try:
                scene_img_size = _png_size(p)
            except Exception:
                pass
            break

    n       = 0  # total rows inserted
    n_match = 0  # rows matching the filter
    for frame_id_raw in frame_ids_color:
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
            cam_intr4x4_list   = build_cam_intr4x4(intrinsics, img_size=scene_img_size)
            cam_tform4x4_world = build_cam_tform4x4_obj(cam_meta)
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
            # Mask EXR pixel value = integer prefix of the object key (e.g. "5_mango_..." → 5),
            # NOT the sequential 'id' field (1, 2, 3...).
            try:
                mask_id = int(obj_name.split("_")[0])
            except (ValueError, IndexError):
                mask_id = None
            bbox_side_len = obj.get("meta", {}).get("bbox_side_len")  # [w, h, d] metres
            scale_raw = obj.get("meta", {}).get("scale", None)
            if isinstance(scale_raw, (list, tuple)):
                obj_scale = float(scale_raw[0])   # isotropic
            elif scale_raw is not None:
                obj_scale = float(scale_raw)
            else:
                obj_scale = 1.0
            has_kpts = 1 if (kpts_preprocess / object_id / "kpts3d.pt").exists() else 0

            # per-object cam_tform4x4_obj: uses the object's own quaternion/translation
            try:
                cam_tform4x4_obj_list = build_obj_cam_tform(obj)
            except Exception:
                cam_tform4x4_obj_list = cam_tform4x4_world

            frame_id = f"{data_type}/{scene_name}/{frame_id_raw}/{obj_idx}"

            cur.execute(
                """
                INSERT OR IGNORE INTO frames
                    (frame_id, scene_name, object_idx, mask_id, split, data_type,
                     category, object_id,
                     rgb_path, depth_path, mask_path,
                     cam_intr4x4, cam_tform4x4_obj, obj_size3d,
                     obj_scale, has_kpts, is_valid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    frame_id, scene_name, obj_idx, mask_id, split, data_type,
                    category, object_id,
                    rgb_rpath, depth_rpath, mask_rpath,
                    json.dumps(cam_intr4x4_list)      if cam_intr4x4_list      is not None else None,
                    json.dumps(cam_tform4x4_obj_list) if cam_tform4x4_obj_list is not None else None,
                    json.dumps(bbox_side_len)          if bbox_side_len         is not None else None,
                    obj_scale, has_kpts, is_valid,
                ),
            )
            n += 1
            if not filter_kpts or has_kpts:
                n_match += 1
            if limit is not None and n_match >= limit:
                return n, n_match
    return n, n_match


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
    """Load depth EXR → (H, W) float32 in metres."""
    if not path.exists():
        return None
    try:
        import os, cv2, numpy as np
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        arr = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
        if arr is None:
            return None
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return torch.from_numpy(arr.astype(np.float32))
    except Exception:
        return None


def _load_mask_tensor(path: Path, mask_id: int) -> Optional[torch.Tensor]:
    """Load scene mask EXR and return a bool (H, W) for the given object mask_id.

    Omni6DPose stores all objects in one EXR.  Channel 2 (BGR) scaled by 255
    gives an integer object-id per pixel matching the 'id' field in meta.json.
    """
    if not path.exists():
        return None
    try:
        import os, cv2, numpy as np
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        arr = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
        if arr is None:
            return None
        ids = np.array(arr[:, :, 2] * 255, dtype=np.uint8)
        ids[ids == 255] = 0  # bug fix for test_real subset (spurious 255 values)
        return torch.from_numpy(ids == mask_id)
    except Exception:
        return None
