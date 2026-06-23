from __future__ import annotations
import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

from o3b.dataset.dataset import register_dataset, ConfigurableDataset, ItemType

import torch
from o3b.data.modalities import (
    FrameObject, SceneObject, Object, ObjectPair,
)

from o3b.dataset.housecorr3d.enum import OMNI6DPOSE_CATEGORIES

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
        if self.cfg.item_type == ItemType.FRAME_OBJECT:
            return len(self._frame_rows_id)
        return len(self._object_rows_id)

    def _setup(self) -> None:
        self._object_rows: list[dict] = []
        self._object_rows_id: list = []
        self._obj_by_id: dict[str, dict] = {}
        self._frame_rows: list[dict] = []
        self._frame_rows_id: list[int] = []

        # Always load the objects table from index.db — used for mesh lookup in all item types.
        db_path = self.path_preprocess / "index.db"
        if db_path.exists():
            con = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True, timeout=30)
            con.row_factory = sqlite3.Row
            try:
                cur = con.cursor()
                self._object_rows = [dict(r) for r in cur.execute("SELECT * FROM objects").fetchall()]
                _id_to_idx: dict[str, int] = {r["object_id"]: i for i, r in enumerate(self._object_rows)}
                self._obj_by_id = {r["object_id"]: r for r in self._object_rows}

                cats = self.cfg.categories  # Optional[list[str]]

                if self.cfg.item_type == ItemType.OBJECT:
                    kpts_clause = " AND obj_kpts3d IS NOT NULL" if self.cfg.filter_has_kpts else ""
                    real_clause = (
                        " AND object_id LIKE '%real%'" if self.cfg.filter_is_real is True
                        else " AND object_id NOT LIKE '%real%'" if self.cfg.filter_is_real is False
                        else ""
                    )
                    limit_clause = f" LIMIT {self.cfg.filter_count_max}" if self.cfg.filter_count_max else ""
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
                        if self.cfg.filter_is_real is True
                        else " AND src_o.object_id NOT LIKE '%real%' AND trgt_o.object_id NOT LIKE '%real%'"
                        if self.cfg.filter_is_real is False
                        else ""
                    )
                    limit_clause = f" LIMIT {self.cfg.filter_count_max}" if self.cfg.filter_count_max else ""
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

        if self.cfg.item_type == ItemType.FRAME_OBJECT:
            frames_db = self.path_preprocess / "frames.db"
            if not frames_db.exists():
                return
            con2 = sqlite3.connect(f"file:{frames_db}?immutable=1", uri=True, timeout=30)
            con2.row_factory = sqlite3.Row
            try:
                cur2 = con2.cursor()
                self._frame_rows = [dict(r) for r in cur2.execute("SELECT * FROM frames").fetchall()]

                split = self.cfg.split or "train"
                is_real = self.cfg.filter_is_real
                cats2 = self.cfg.categories
                limit = self.cfg.filter_count_max
                has_kpts = self.cfg.filter_has_kpts

                split_clause = "" if split == "all" else f" AND split = '{split}'"
                if is_real is None:
                    real_clause2 = ""
                elif is_real:
                    real_clause2 = " AND data_type = 'real'"
                else:
                    real_clause2 = " AND data_type = 'synthetic'"
                kpts_clause2  = " AND has_kpts = 1" if has_kpts else ""
                limit_clause2 = f" LIMIT {limit}" if limit else ""

                if cats2:
                    placeholders2 = ", ".join("?" * len(cats2))
                    cat_clause2   = f" AND category IN ({placeholders2})"
                    params2: list = list(cats2)
                else:
                    cat_clause2, params2 = "", []

                frows = cur2.execute(
                    f"SELECT rowid FROM frames"
                    f" WHERE is_valid = 1{split_clause}{real_clause2}{kpts_clause2}{cat_clause2}{limit_clause2}",
                    params2,
                ).fetchall()
                # rowid is 1-based; our list is 0-based
                self._frame_rows_id = [r[0] - 1 for r in frows]
            finally:
                con2.close()

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
            from o3b.io import _load_mesh
            mesh, tform = _load_mesh(mesh_entry)

            if (need_mesh or need_verts3d or need_verts3d_feats) and \
                    self.cfg.mesh_type != "default" and mesh is not None:
                from o3b.data.datatypes.mesh import Mesh
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

        obj_size_ncds = 2.0 if tform is not None else None
        obj_size      = float(2.0 * tform[0, 0].item()) if tform is not None else None

        obj = Object(
            object_id               = oid,
            mesh                    = mesh if need_mesh else None,
            verts3d                 = mesh.verts if (need_verts3d and mesh is not None) else None,
            verts3d_feats           = mesh.vert_feats if (need_verts3d_feats and mesh is not None) else None,
            obj_ncds0c_tform4x4_obj = tform if (need_tform or need_mesh) else None,
            obj_size_ncds           = obj_size_ncds,
            obj_size                = obj_size,
            obj_kpts3d              = kpts3d,
            obj_kpts3d_mask         = kpts3d_mask,
            category                = category,
            category_id             = category_id,
        )
        
        if self.cfg.obj_tform4x4 is not None:
            import torch as _torch
            obj = obj.transform(_torch.tensor(self.cfg.obj_tform4x4, dtype=_torch.float32))
        
        return obj

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
    def index(cls, cfg, *, db: Optional[Path] = None, **kwargs) -> None:
        if cfg.item_type == ItemType.FRAME_OBJECT:
            cls._index_frames(cfg, db=db, **kwargs)
            return
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
    def _index_frames(
        cls,
        cfg,
        *,
        db: Optional[Path] = None,
        remove: bool = False,
        max_index: Optional[int] = None,
    ) -> None:
        """Index frame-object data from ROPE/SOPE into frames.db."""
        import sqlite3 as _sq
        from o3b.dataset.housecorr3d.frame_dataset import _index_scene

        path_raw        = cls._path_raw(cfg)
        path_preprocess = cls._path_preprocess(cfg)
        db_path         = db or path_preprocess / "frames.db"
        limit       = max_index or cfg.filter_count_max
        filter_kpts = cfg.filter_has_kpts
        is_real     = cfg.filter_is_real
        kpts_preprocess = path_preprocess / "obj_kpts3d"

        print(f"path_raw        : {path_raw}")
        print(f"path_preprocess : {path_preprocess}")
        print(f"db              : {db_path}")

        rope_root = path_raw / "ROPE"
        sope_root = path_raw / "SOPE"
        if not rope_root.exists() and not sope_root.exists():
            print(f"ERROR: neither {rope_root} nor {sope_root} found.", file=sys.stderr)
            sys.exit(1)

        db_path.parent.mkdir(parents=True, exist_ok=True)
        if remove and db_path.exists():
            db_path.unlink()
            print(f"Removed existing index: {db_path}")
        for suffix in ("-wal", "-shm"):
            db_path.with_name(db_path.name + suffix).unlink(missing_ok=True)

        con = _sq.connect(db_path, timeout=60)
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

        COMMIT_EVERY = 50
        total_rows = matched = scenes_since_commit = 0
        done = False

        def _remaining():
            return (limit - matched) if limit else None

        def _update(n_total: int, n_match: int) -> bool:
            nonlocal total_rows, matched, scenes_since_commit
            total_rows += n_total
            matched    += n_match
            scenes_since_commit += 1
            if scenes_since_commit >= COMMIT_EVERY:
                con.commit()
                scenes_since_commit = 0
            return bool(limit and matched >= limit)

        if rope_root.exists() and not done and is_real is not False:
            scene_dirs = sorted(d for d in rope_root.iterdir() if d.is_dir())
            print(f"\nIndexing ROPE ({len(scene_dirs)} scenes, split=test, data_type=real)")
            bar = tqdm(scene_dirs, unit="scene", desc="ROPE")
            for scene_dir in bar:
                n_total, n_match = _index_scene(
                    cur=cur, scene_dir=scene_dir, scene_name=scene_dir.name,
                    split="test", data_type="real", path_raw=path_raw,
                    kpts_preprocess=kpts_preprocess, limit=_remaining(), filter_kpts=filter_kpts,
                )
                bar.set_postfix(rows=total_rows, matched=matched)
                if _update(n_total, n_match):
                    done = True; bar.close(); break

        if sope_root.exists() and not done and is_real is not True:
            patch_dirs = sorted(d for d in sope_root.iterdir() if d.is_dir())
            print(f"\nIndexing SOPE ({len(patch_dirs)} patches, data_type=synthetic)")

            def _sope_scenes():
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
                    cur=cur, scene_dir=scene_dir, scene_name=scene_name,
                    split=split_name, data_type="synthetic", path_raw=path_raw,
                    kpts_preprocess=kpts_preprocess, limit=_remaining(), filter_kpts=filter_kpts,
                )
                bar.set_postfix(rows=total_rows, matched=matched)
                if _update(n_total, n_match):
                    bar.close(); break

        con.commit()
        con.close()
        print(f"\nDone. {total_rows} rows indexed ({matched} matching filter) → {db_path}")

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
        debug: bool = False,
        obj_centric: bool = False,
        **_,
    ) -> None:
        if cfg.item_type == ItemType.FRAME_OBJECT:
            cls._visualize_frame_objects(cfg, db=db, limit=limit, object_id=object_id,
                                          render=render, debug=debug, obj_centric=obj_centric)
            return

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

        from o3b.data.viz import visualize_dataset
        visualize_dataset(dataset, render=render, render_frames=render_frames, renderer=renderer, debug=debug)

    @classmethod
    def _visualize_frame_objects(
        cls,
        cfg,
        *,
        db: Optional[Path] = None,
        limit: int = 20,
        object_id: Optional[str] = None,
        render: bool = False,
        debug: bool = False,
        obj_centric: bool = False,
    ) -> None:
        from o3b.dataset.housecorr3d.frame_dataset import _visualize_frame_objects_viser
        from dataclasses import replace as _r

        path_preprocess = cls._path_preprocess(cfg)
        db_path = db or path_preprocess / "frames.db"
        if not db_path.exists():
            print(f"No index found at {db_path}. Run 'index' first.", file=sys.stderr)
            sys.exit(1)

        # Load dataset with all modalities for visualization
        viz_cfg = _r(cfg, modalities=None, object_modalities=None)
        dataset = cls(viz_cfg)
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

        if render:
            for seq_idx in range(len(dataset._frame_rows_id)):
                row = dataset._frame_rows[dataset._frame_rows_id[seq_idx]]
                print(
                    f"[{seq_idx + 1}/{len(dataset._frame_rows_id)}]"
                    f"  {row['frame_id']:<60}"
                    f"  cat={row.get('category', '?'):<20}"
                    f"  split={row['split']}  {row['data_type']}"
                )
                fo = dataset._load_frame_object(seq_idx)
                fo.viz(show=True)
            return

        _visualize_frame_objects_viser(dataset, debug=debug, obj_centric=obj_centric)

    # ── item loading ──────────────────────────────────────────────────────────

    def _load_frame_object(self, idx: int) -> FrameObject:
        """Load one frame-object from frames.db.

        Object geometry (mesh, kpts, tform) is delegated to _load_object_from_row,
        which applies obj_tform4x4, mesh_type conversion, and normalises kpts to
        NCDS space.  obj_size is rescaled from mesh units to metres via obj_scale.
        cam_tform4x4_obj_ncds = cam_tform4x4_obj_metric @ obj_ncds0c_tform4x4_obj
        maps NCDS → camera space in metric units.
        """
        from o3b.dataset.housecorr3d.frame_dataset import (
            _load_image_tensor, _load_depth_tensor, _load_mask_tensor,
        )

        row  = self._frame_rows[self._frame_rows_id[idx]]
        mods = self.cfg.modalities
        oid       = row.get("object_id", "")
        obj_scale = float(row.get("obj_scale") or 1.0)

        # ── frame modalities ──────────────────────────────────────────────────
        rgb = _load_image_tensor(self.path_raw / row["rgb_path"]) \
            if _want("rgb", mods) and row.get("rgb_path") else None
        depth = _load_depth_tensor(self.path_raw / row["depth_path"]) \
            if _want("depth", mods) and row.get("depth_path") else None
        depth_mask = (depth > 0) if (_want("depth_mask", mods) and depth is not None) else None
        mask = _load_mask_tensor(self.path_raw / row["mask_path"], int(row["mask_id"])) \
            if _want("fo_mask", mods) and row.get("mask_path") and row.get("mask_id") is not None else None

        cam_intr4x4 = (torch.tensor(json.loads(row["cam_intr4x4"]), dtype=torch.float32)
                       if _want("cam_intr4x4", mods) and row.get("cam_intr4x4") else None)

        # Always load — needed for cam_tform4x4_obj_ncds even when not directly requested.
        cam_tform4x4_obj_raw = (torch.tensor(json.loads(row["cam_tform4x4_obj"]), dtype=torch.float32)
                                if row.get("cam_tform4x4_obj") else None)
        
        #if cam_tform4x4_obj_raw is not None and self.cfg.obj_tform4x4 is not None:
        #    T_inv = torch.linalg.inv(torch.tensor(self.cfg.obj_tform4x4, dtype=torch.float32))
        #    cam_tform4x4_obj_raw = cam_tform4x4_obj_raw @ T_inv
        
        if cam_tform4x4_obj_raw is not None and self.cfg.cam_tform4x4_cam_raw is not None:
            C = torch.tensor(self.cfg.cam_tform4x4_cam_raw, dtype=torch.float32)
            cam_tform4x4_obj_raw = C @ cam_tform4x4_obj_raw
        
        cam_bbox2d = None
        if _want("cam_bbox2d", mods) and mask is not None:
            from o3b.cv.visual.draw import get_bboxs_from_masks
            cam_bbox2d = get_bboxs_from_masks(mask[None])[0].float()
        cam_bbox3d = (torch.tensor(json.loads(row["obj_size3d"]), dtype=torch.float32)
                      if _want("cam_bbox3d", mods) and row.get("obj_size3d") else None)

        # Build metric cam_tform4x4_obj: SVD-normalise rotation, embed obj_scale.
        # Robust to whether the stored matrix has scale embedded or not.
        cam_tform4x4_obj_metric = None
        if cam_tform4x4_obj_raw is not None:
            U, _, Vh = torch.linalg.svd(cam_tform4x4_obj_raw[:3, :3].float())
            cam_tform4x4_obj_metric = cam_tform4x4_obj_raw.clone().float()
            cam_tform4x4_obj_metric[:3, :3] = (U @ Vh) * obj_scale
        cam_tform4x4_obj = cam_tform4x4_obj_metric if _want("cam_tform4x4_obj", mods) else None

        # ── object modalities via _load_object_from_row ───────────────────────
        # Handles mesh_type conversion, obj_tform4x4, and kpts normalised to NCDS.
        obj_row = self._obj_by_id.get(oid)
        obj     = self._load_object_from_row(obj_row) if obj_row is not None else None

        tform = obj.obj_ncds0c_tform4x4_obj if obj is not None else None

        cam_tform4x4_obj_ncds = None
        obj_size_ncds = None
        obj_size      = None
        if tform is not None:
            obj_size_ncds = 2.0
            # _load_object_from_row gives obj_size in mesh units; scale to metres
            obj_size = (obj.obj_size * obj_scale) if obj.obj_size is not None else None
            if cam_tform4x4_obj_metric is not None and _want("cam_tform4x4_obj_ncds", mods):
                cam_tform4x4_obj_ncds = cam_tform4x4_obj_metric @ tform.float()

        return FrameObject(
            frame_id                = row.get("frame_id", ""),
            frame_object_id         = row.get("frame_id", ""),
            object_id               = oid,
            rgb                     = rgb,
            depth                   = depth,
            depth_mask              = depth_mask,
            fo_mask                 = mask,
            cam_intr4x4             = cam_intr4x4,
            cam_tform4x4_obj        = cam_tform4x4_obj,
            cam_tform4x4_obj_ncds   = cam_tform4x4_obj_ncds,
            cam_bbox2d              = cam_bbox2d,
            cam_bbox3d              = cam_bbox3d,
            mesh                    = obj.mesh if obj is not None else None,
            obj_ncds0c_tform4x4_obj = tform if _want("obj_ncds0c_tform4x4_obj", mods) else None,
            obj_size_ncds           = obj_size_ncds,
            obj_size                = obj_size,
            obj_kpts3d              = obj.obj_kpts3d      if obj is not None else None,
            obj_kpts3d_mask         = obj.obj_kpts3d_mask if obj is not None else None,
            category                = row.get("category") if _want("category", mods) else None,
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
