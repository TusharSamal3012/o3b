from __future__ import annotations
import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

from od3d_basic.dataset.dataset import register_dataset, ConfigurableDataset, ItemType

import torch
from od3d_basic.data.modalities import (
    FrameObject, SceneObject, Object, ObjectPair,
)

from od3d_basic.dataset.housecorr3d.enum import OMNI6DPOSE_CATEGORIES

_MESH_EXTS = {".obj", ".ply", ".glb", ".gltf", ".stl", ".fbx"}


@register_dataset("HouseCorr3D")
class HouseCorr3D(ConfigurableDataset):

    categories: tuple = tuple(OMNI6DPOSE_CATEGORIES)

    @property
    def path_raw(self) -> Path:
        return self.cfg.path_raw

    @property
    def path_preprocess(self) -> Path:
        return self.cfg.path_preprocess

    @property
    def path_object_meshes(self) -> Path:
        return self.path_raw / "PAM" / "object_meshes"

    @property
    def path_raw_meta(self) -> Path:
        return self.path_raw / "Meta"

    def __len__(self):
        return len(self._object_rows_id)

    def _setup(self) -> None:
        self._object_rows: list[dict] = []
        db_path = self.path_preprocess / "index.db"
        if not db_path.exists():
            return
        # immutable=1: skip WAL/-shm entirely (safe here — _setup only reads).
        # mode=ro alone still needs the -shm file, which fails on read-only
        # Slurm filesystems even for plain SELECTs.
        con = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            cur = con.cursor()

            self._object_rows = [dict(r) for r in cur.execute("SELECT * FROM objects").fetchall()]
            _id_to_idx: dict[str, int] = {r["object_id"]: i for i, r in enumerate(self._object_rows)}

            cats = self.cfg.categories  # Optional[list[str]]

            if self.cfg.item_type == ItemType.OBJECT:
                kpts_clause = " AND obj_kpts3d IS NOT NULL" if self.cfg.filter_has_kpts else ""
                real_clause = (
                    " AND object_id LIKE '%real%'"
                    if self.cfg.filter_is_real else
                    " AND object_id NOT LIKE '%real%'"
                )
                limit_clause = f" LIMIT {self.cfg.max_samples}" if self.cfg.max_samples else ""
                if cats:
                    placeholders = ", ".join("?" * len(cats))
                    cat_clause   = f" AND category IN ({placeholders})"
                    params: list = list(cats)
                else:
                    cat_clause, params = "", []
                rows = cur.execute(
                    f"SELECT object_id FROM objects WHERE 1=1{kpts_clause}{real_clause}{cat_clause}{limit_clause}",
                    params,
                ).fetchall()
                self._object_rows_id = [_id_to_idx[r["object_id"]] for r in rows]

            elif self.cfg.item_type == ItemType.OBJECT_PAIR:
                kpts_clause = (
                    " AND src_o.obj_kpts3d IS NOT NULL AND trgt_o.obj_kpts3d IS NOT NULL"
                    if self.cfg.filter_has_kpts else ""
                )
                real_clause = (
                    " AND src_o.object_id LIKE '%real%' AND trgt_o.object_id LIKE '%real%'"
                    if self.cfg.filter_is_real else
                    " AND src_o.object_id NOT LIKE '%real%' AND trgt_o.object_id NOT LIKE '%real%'"
                )
                limit_clause = f" LIMIT {self.cfg.max_samples}" if self.cfg.max_samples else ""
                if cats:
                    placeholders = ", ".join("?" * len(cats))
                    cat_clause   = (f" AND src_o.category IN ({placeholders})"
                                    f" AND trgt_o.category IN ({placeholders})")
                    params = list(cats) + list(cats)
                else:
                    cat_clause, params = "", []
                rows = cur.execute(f"""
                    SELECT op.src_object_id, op.trgt_object_id
                    FROM object_pairs op
                    JOIN objects src_o  ON op.src_object_id  = src_o.object_id
                    JOIN objects trgt_o ON op.trgt_object_id = trgt_o.object_id
                    WHERE 1=1{kpts_clause}{real_clause}{cat_clause}{limit_clause}
                """, params).fetchall()
                self._object_rows_id = [
                    (_id_to_idx[r["src_object_id"]], _id_to_idx[r["trgt_object_id"]])
                    for r in rows
                ]
        finally:
            con.close()

    def _load_object(self, idx: int) -> Object:
        return self._load_object_from_row(self._object_rows[self._object_rows_id[idx]])

    def _load_object_from_row(self, row: dict) -> Object:
        oid = row["object_id"]
        mods = self.cfg.object_modalities

        need_mesh        = _want("mesh",                    mods)
        need_verts3d     = _want("verts3d",                 mods)
        need_verts3d_feats = _want("verts3d_feats",         mods)
        need_tform       = _want("obj_ncds0c_tform4x4_obj", mods)
        need_kpts        = _want("obj_kpts3d",              mods)
        need_category    = _want("category",                mods)

        mesh, tform = None, None
        if need_mesh or need_verts3d or need_verts3d_feats or need_tform or need_kpts:
            mesh_rel = row.get("mesh_path")
            mesh_entry = self.path_raw / mesh_rel if mesh_rel else self.path_object_meshes / oid
            from od3d_basic.io import _load_mesh
            mesh, tform = _load_mesh(mesh_entry)

            if (need_mesh or need_verts3d or need_verts3d_feats) and \
                    self.cfg.mesh_type != "default" and mesh is not None:
                from od3d_basic.data.datatypes.mesh import Mesh
                converted_path = (
                    self.path_preprocess / "mesh" / self.cfg.mesh_type / f"{oid}.glb"
                )
                mesh = Mesh.load_or_convert(converted_path, mesh_entry, self.cfg.mesh_type)

        kpts3d, kpts3d_mask = None, None
        if need_kpts:
            kpts3d, kpts3d_mask = _load_kpts3d_by_id(oid, self.path_preprocess, tform)

        category_id = None
        category = None
        if need_category:
            cat_str = row.get("category")
            if cat_str is not None:
                try:
                    category_id = list(self.categories).index(cat_str)
                except ValueError:
                    pass
            category = row.get("category")

        return Object(
            object_id               = oid,
            mesh                    = mesh if need_mesh else None,
            verts3d                 = mesh.verts if (need_verts3d and mesh is not None) else None,
            verts3d_feats           = mesh.vert_feats if (need_verts3d_feats and mesh is not None) else None,
            obj_ncds0c_tform4x4_obj = tform if (need_tform or need_mesh) else None,
            obj_kpts3d              = kpts3d,
            obj_kpts3d_mask         = kpts3d_mask,
            category                = category,
            category_id             = category_id,
        )

    def _load_object_pair(self, idx: int) -> ObjectPair:
        src_idx, trgt_idx = self._object_rows_id[idx]
        src_row  = self._object_rows[src_idx]
        trgt_row = self._object_rows[trgt_idx]
        return ObjectPair(
            src_object_id  = src_row["object_id"],
            trgt_object_id = trgt_row["object_id"],
            src_object     = self._load_object_from_row(src_row),
            trgt_object    = self._load_object_from_row(trgt_row),
        )

    # ── CLI hooks ─────────────────────────────────────────────────────────────

    @classmethod
    def _path_raw(cls, cfg) -> Path:
        return cfg.path_raw or cfg.root

    @classmethod
    def _path_preprocess(cls, cfg) -> Path:
        return cfg.path_preprocess or cfg.root

    @classmethod
    def fetch(cls, cfg, *, url: Optional[str] = None) -> None:
        path_raw = cls._path_raw(cfg)
        print(f"Target directory: {path_raw}")
        path_raw.mkdir(parents=True, exist_ok=True)
        if not url:
            print(
                "No --url provided.\n"
                "Expected on-disk layout after download:\n"
                f"  {path_raw}/PAM/object_meshes/<object_id>/mesh.obj\n"
                f"  {path_raw}/Meta/<object_id>.json\n"
                "Provide --url <zip-url> to download automatically."
            )
            return
        import tempfile, urllib.request, zipfile
        print(f"Downloading {url} …")
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            urllib.request.urlretrieve(url, tmp_path, _download_progress)
            print(f"\nExtracting to {path_raw} …")
            with zipfile.ZipFile(tmp_path) as zf:
                zf.extractall(path_raw)
            print("Done.")
        finally:
            tmp_path.unlink(missing_ok=True)

    @classmethod
    def index(cls, cfg, *, db: Optional[Path] = None) -> None:
        if cfg.item_type not in (ItemType.OBJECT, ItemType.OBJECT_PAIR):
            print(f"ERROR: item_type '{cfg.item_type}' indexing is not implemented.", file=sys.stderr)
            sys.exit(1)
        if cfg.item_type == ItemType.OBJECT_PAIR:
            cls._index_object_pairs(cfg, db=db)
            return
        path_raw = cls._path_raw(cfg)
        path_preprocess = cls._path_preprocess(cfg)

        db_path = db or path_preprocess / "index.db"
        mesh_root = path_raw / "PAM" / "object_meshes"
        meta_root = path_raw / "Meta"

        print(f"path_raw        : {path_raw}")
        print(f"path_preprocess : {path_preprocess}")
        print(f"mesh_root       : {mesh_root}")
        print(f"meta_root       : {meta_root}")
        print(f"db              : {db_path}")

        if not mesh_root.exists():
            print(f"ERROR: {mesh_root} does not exist. Run 'fetch' first.", file=sys.stderr)
            sys.exit(1)

        meta_by_id: dict[str, dict] = {}
        if meta_root.exists():
            meta_files = [f for f in sorted(meta_root.iterdir())
                          if f.suffix in (".json", ".yaml", ".yml")]
            print(f"\nReading meta   : {len(meta_files)} file(s) in {meta_root}")
            for f in meta_files:
                _load_meta_file(f, meta_by_id, fmt="json" if f.suffix == ".json" else "yaml")
            print(f"  loaded meta for {len(meta_by_id)} object(s)")
        else:
            print(f"\nNote: {meta_root} not found — no meta columns will be added.")

        print(f"\nScanning meshes : {mesh_root}")
        mesh_entries: list[dict] = []
        no_mesh: list[str] = []
        for entry in sorted(mesh_root.iterdir()):
            if entry.is_dir():
                mesh_file = _first_mesh_in_dir(entry)
                if mesh_file is None:
                    no_mesh.append(entry.name)
                mesh_entries.append({
                    "object_id": entry.name,
                    "mesh_path": str(mesh_file.relative_to(path_raw)) if mesh_file else None,
                })
            elif entry.suffix.lower() in _MESH_EXTS:
                mesh_entries.append({
                    "object_id": entry.stem,
                    "mesh_path": str(entry.relative_to(path_raw)),
                })
        print(f"  found {len(mesh_entries)} object(s)"
              + (f", {len(no_mesh)} without a mesh file" if no_mesh else ""))

        kpts_root = path_preprocess / "obj_kpts3d"
        print(f"\nScanning kpts   : {kpts_root}")
        kpts_by_id: dict[str, tuple] = {}
        if kpts_root.exists():
            for entry in sorted(kpts_root.iterdir()):
                if not entry.is_dir():
                    continue
                kpts_file = entry / "kpts3d.pt"
                if not kpts_file.exists():
                    continue
                try:
                    t = torch.load(kpts_file, map_location="cpu")  # (K, 4)
                    kpts3d = t[:, :3].tolist()
                    mask   = t[:, 3].bool().tolist()
                    kpts_by_id[entry.name] = (json.dumps(kpts3d), json.dumps(mask))
                except Exception as exc:
                    print(f"  WARN: could not load {kpts_file}: {exc}")
            print(f"  loaded kpts for {len(kpts_by_id)} object(s)")
        else:
            print(f"  Note: {kpts_root} not found — obj_kpts3d/obj_kpts3d_mask will be NULL.")

        _META_SKIP = {"object_id", "class_name"}
        all_meta_keys: list[str] = []
        for meta in meta_by_id.values():
            for k in meta:
                if k not in all_meta_keys and k not in _META_SKIP:
                    all_meta_keys.append(k)
        print(f"\nMeta columns    : {all_meta_keys if all_meta_keys else '(none)'}")

        extra_col_defs = (
            ", " + ", ".join(f'"{k}" TEXT' for k in all_meta_keys)
            if all_meta_keys else ""
        )

        print(f"\nWriting index   : {db_path}")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("DROP TABLE IF EXISTS objects")
        cur.execute(f"""
            CREATE TABLE objects (
                object_id       TEXT PRIMARY KEY,
                mesh_path       TEXT,
                obj_kpts3d      TEXT,
                obj_kpts3d_mask TEXT,
                category        TEXT{extra_col_defs}
            )
        """)
        for i, entry in enumerate(mesh_entries):
            oid = entry["object_id"]
            meta = meta_by_id.get(oid, {})
            kpts_json, mask_json = kpts_by_id.get(oid, (None, None))
            category = _to_text(meta.get("class_name"))
            cols = ["object_id", "mesh_path", "obj_kpts3d", "obj_kpts3d_mask", "category"] + all_meta_keys
            vals = [oid, entry["mesh_path"], kpts_json, mask_json, category] + [_to_text(meta.get(k)) for k in all_meta_keys]
            placeholders = ", ".join("?" * len(cols))
            col_names = ", ".join(f'"{c}"' for c in cols)
            cur.execute(f"INSERT INTO objects ({col_names}) VALUES ({placeholders})", vals)
            if (i + 1) % 100 == 0 or (i + 1) == len(mesh_entries):
                sys.stdout.write(f"\r  inserted {i + 1}/{len(mesh_entries)}")
                sys.stdout.flush()
        print()
        con.commit()

        con.row_factory = sqlite3.Row
        first = con.execute("SELECT * FROM objects LIMIT 1").fetchone()
        if first:
            print("\nFirst row:")
            for col in first.keys():
                val = first[col]
                display = val if val is None or len(str(val)) <= 80 else str(val)[:77] + "…"
                print(f"  {col:<30} = {display!r}")

        con.close()
        print(f"\nDone. {len(mesh_entries)} object(s) indexed -> {db_path}")

    @classmethod
    def _index_object_pairs(cls, cfg, *, db: Optional[Path] = None) -> None:
        path_preprocess = cls._path_preprocess(cfg)
        db_path = db or path_preprocess / "index.db"

        if not db_path.exists():
            print(f"ERROR: {db_path} not found. Run 'index' with item_type=object first.", file=sys.stderr)
            sys.exit(1)

        con = sqlite3.connect(db_path)
        cur = con.cursor()

        existing = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='objects'"
        ).fetchone()
        if not existing:
            print("ERROR: 'objects' table not found. Run 'index' with item_type=object first.", file=sys.stderr)
            con.close()
            sys.exit(1)

        cur.execute("DROP TABLE IF EXISTS object_pairs")
        cur.execute("""
            CREATE TABLE object_pairs (
                src_object_id  TEXT NOT NULL,
                trgt_object_id TEXT NOT NULL,
                category       TEXT,
                PRIMARY KEY (src_object_id, trgt_object_id)
            )
        """)

        cur.execute("""
            INSERT INTO object_pairs (src_object_id, trgt_object_id, category)
            SELECT a.object_id, b.object_id, a.category
            FROM objects a
            JOIN objects b ON a.category IS b.category AND a.object_id != b.object_id
            WHERE a.category IS NOT NULL
        """)

        n_pairs = cur.execute("SELECT COUNT(*) FROM object_pairs").fetchone()[0]
        con.commit()
        con.close()
        print(f"Done. {n_pairs} object pair(s) indexed -> {db_path}")

    @classmethod
    def visualize(
        cls,
        cfg,
        *,
        db: Optional[Path] = None,
        limit: int = 20,
        object_id: Optional[str] = None,
        render: bool = False,
        render_frames: int = 0,
        renderer: str = "pyrender",
    ) -> None:
        path_preprocess = cls._path_preprocess(cfg)
        db_path = db or path_preprocess / "index.db"
        if not db_path.exists():
            print(f"No index found at {db_path}. Run 'index' first.", file=sys.stderr)
            sys.exit(1)

        dataset = cls(cfg)
        if not dataset._object_rows_id:
            print("No objects found in index"
                  + (" with keypoints available." if cfg.filter_has_kpts else "."))
            return

        total = len(dataset._object_rows_id)

        if object_id:
            if cfg.item_type == ItemType.OBJECT_PAIR:
                dataset._object_rows_id = [
                    (si, ti) for si, ti in dataset._object_rows_id
                    if dataset._object_rows[si]["object_id"] == object_id
                    or dataset._object_rows[ti]["object_id"] == object_id
                ]
            else:
                dataset._object_rows_id = [
                    i for i in dataset._object_rows_id
                    if dataset._object_rows[i]["object_id"] == object_id
                ]
        if limit < len(dataset._object_rows_id):
            dataset._object_rows_id = dataset._object_rows_id[:limit]

        filter_note = "  [filtered: has kpts]" if cfg.filter_has_kpts else ""
        print(f"Showing {len(dataset._object_rows_id)} / {total} objects{filter_note}  {db_path}\n")
        for entry in dataset._object_rows_id:
            if cfg.item_type == ItemType.OBJECT_PAIR:
                src_row  = dataset._object_rows[entry[0]]
                trgt_row = dataset._object_rows[entry[1]]
                mask_json = src_row.get("obj_kpts3d_mask")
                if mask_json:
                    mask = json.loads(mask_json)
                    kpts_info = f"  kpts={sum(mask)}/{len(mask)}"
                else:
                    kpts_info = "  kpts=none"
                print(f"{src_row['object_id']}  <--->  {trgt_row['object_id']}{kpts_info}")
            else:
                row = dataset._object_rows[entry]
                mask_json = row.get("obj_kpts3d_mask")
                if mask_json:
                    mask = json.loads(mask_json)
                    kpts_info = f"  kpts={sum(mask)}/{len(mask)}"
                else:
                    kpts_info = "  kpts=none"
                print(f"{row['object_id']}{kpts_info}")

        from od3d_basic.data.viz import visualize_dataset
        visualize_dataset(dataset, render=render, render_frames=render_frames, renderer=renderer)

    # ── item loading ──────────────────────────────────────────────────────────

    def _load_frame_object(self, idx: int) -> FrameObject:
        row = self._object_rows[self._object_rows_id[idx]]
        obj_id = row["object_id"]
        frame_idx = 0
        mesh_rel = row.get("mesh_path")
        entry = self.path_raw / mesh_rel if mesh_rel else self.path_object_meshes / obj_id

        from od3d_basic.io import _load_mesh
        mesh, tform = (
            _load_mesh(entry) if _want("mesh", self.cfg.modalities) else (None, None)
        )
        kpts3d, kpts3d_mask = (
            _load_kpts3d_by_id(obj_id, self.path_preprocess, tform)
            if _want("obj_kpts3d", self.cfg.modalities) else (None, None)
        )
        return FrameObject(
            frame_object_id         = f"{obj_id}_{frame_idx}",
            frame_id                = f"{obj_id}_{frame_idx}",
            object_id               = obj_id,
            rgb                     = _load_rgb(entry, frame_idx)    if _want("rgb",              self.cfg.modalities) else None,
            depth                   = _load_depth(entry, frame_idx)  if _want("depth",            self.cfg.modalities) else None,
            cam_intr4x4             = _load_intr(entry)              if _want("cam_intr4x4",      self.cfg.modalities) else None,
            cam_bbox2d              = _load_bbox2d(entry, frame_idx) if _want("cam_bbox2d",       self.cfg.modalities) else None,
            cam_tform4x4_obj        = _load_pose(entry, frame_idx)   if _want("cam_tform4x4_obj", self.cfg.modalities) else None,
            mesh                    = mesh,
            obj_ncds0c_tform4x4_obj = tform,
            obj_kpts3d              = kpts3d,
            obj_kpts3d_mask         = kpts3d_mask,
            category                = _load_category(entry) if _want("category", self.cfg.modalities) else None,
        )

    def _load_scene_object(self, idx: int) -> SceneObject:
        row = self._object_rows[self._object_rows_id[idx]]
        obj_id = row["object_id"]
        mesh_rel = row.get("mesh_path")
        entry = self.path_raw / mesh_rel if mesh_rel else self.path_object_meshes / obj_id
        T = self.cfg.scene_length
        frame_indices = _sample_frame_indices(entry, T)

        fo_list = [self._load_frame(entry, i) for i in frame_indices]
        return SceneObject.from_frame_objects(
            fo_list,
            scene_object_id = obj_id,
            scene_id        = obj_id,
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _download_progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = "#" * int(pct // 2)
        sys.stdout.write(f"\r  [{bar:<50}] {pct:5.1f}%")
        sys.stdout.flush()


def _load_meta_file(path: Path, store: dict, fmt: str = "json") -> None:
    try:
        if fmt == "json":
            data = json.loads(path.read_text())
        else:
            import yaml
            data = yaml.safe_load(path.read_text())
        if isinstance(data, dict) and "instance_dict" in data:
            for obj_id, obj_meta in data["instance_dict"].items():
                store[obj_id] = obj_meta if isinstance(obj_meta, dict) else {"value": str(obj_meta)}
        else:
            store[path.stem] = data if isinstance(data, dict) else {"value": str(data)}
    except Exception as exc:
        print(f"  WARN: could not parse {path}: {exc}")


def _first_mesh_in_dir(directory: Path) -> Optional[Path]:
    for ext in (".obj", ".ply", ".glb", ".gltf", ".stl", ".fbx"):
        hits = sorted(directory.glob(f"*{ext}"))
        if hits:
            return hits[0]
    return None


def _to_text(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _want(name: str, mods: Optional[set]) -> bool:
    return mods is None or name in mods


def _n_frames(entry: Path) -> int:
    if entry.is_dir():
        return max(1, len(list(entry.glob("*.png"))) + len(list(entry.glob("*.jpg"))))
    return 1


def _sample_frame_indices(entry: Path, T: int) -> list[int]:
    n = _n_frames(entry)
    step = max(1, n // T)
    return list(range(0, min(n, T * step), step))[:T]


def _load_rgb(_entry: Path, _frame_idx: int):
    return None

def _load_depth(_entry: Path, _frame_idx: int):
    return None

def _load_intr(_entry: Path):
    return None

def _load_bbox2d(_entry: Path, _frame_idx: int):
    return None

def _load_pose(_entry: Path, _frame_idx: int):
    return None

def _load_category(_entry: Path):
    return None


def _load_kpts3d_by_id(
    obj_id: str,
    path_preprocess: Path,
    tform: Optional[torch.Tensor],
):
    kpts_file = path_preprocess / "obj_kpts3d" / obj_id / "kpts3d.pt"
    if not kpts_file.exists():
        return None, None
    try:
        t = torch.load(kpts_file, map_location="cpu")  # (K, 4)
        kpts3d = t[:, :3].float()
        mask   = t[:, 3].bool()
        if tform is not None:
            scale  = tform[0, 0]
            center = tform[:3, 3]
            kpts3d = (kpts3d - center) / scale
        return kpts3d, mask
    except Exception:
        return None, None
