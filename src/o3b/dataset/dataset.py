from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union
import yaml

import torch
from torch.utils.data import Dataset as _TorchDataset, DataLoader
from o3b.data.modalities import (
    FrameObject, SceneObject,
    FrameObjectBatch, SceneObjectBatch,
    ObjectPair,
    collate_frame_objects, collate_scene_objects,
)


# ── Item / batch type enums ───────────────────────────────────────────────────

class ItemType(str, Enum):
    OBJECT            = "object"
    OBJECT_PAIR       = "object_pair"
    FRAME_OBJECT      = "frame_object"
    FRAME_OBJECT_PAIR = "frame_object_pair"
    SCENE_OBJECT      = "scene_object"

class BatchType(str, Enum):
    OBJECT            = "object"
    OBJECT_PAIR       = "object_pair"
    FRAME_OBJECT      = "frame_object"
    FRAME_OBJECT_PAIR = "frame_object_pair"
    SCENE_OBJECT      = "scene_object"


# ── Dataset config ────────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    # what and where
    class_name:      str                   # e.g. "HouseCorr3D"
    root:            Path                  = Path("data")
    path_raw:        Optional[Path]        = None   # raw (downloaded) data root
    path_preprocess: Optional[Path]        = None   # preprocessed data root
    split:           str                   = "train"

    # item / batch shape
    item_type:     ItemType                = ItemType.OBJECT
    batch_type:    BatchType               = BatchType.FRAME_OBJECT
    scene_length:  int                     = 8      # T; ignored for OBJECT / FRAME_OBJECT

    # mesh variant: "default" uses the raw mesh path; others load/create a
    # marching-cubes remesh at path_preprocess/mesh/<mesh_type>/<obj_id>.glb
    mesh_type:     str                     = "default"

    # which modalities to load (None = all available)
    modalities:        Optional[set[str]]  = None
    object_modalities: Optional[set[str]]  = None

    # filtering
    categories:        Optional[list[str]]   = None   # None = all
    subsets:           Optional[list[str]]   = None   # None = all; e.g. ["train"] or ["train","val"]
    filter_count_max:  Optional[int]         = None   # None = all; max number of samples to load
    filter_has_kpts:   bool                  = False
    # frame_object_pair: how many viewpoints (frames) to sample per object instance
    # when forming cross-instance pairs (1 = one representative frame per instance,
    # -1 = use all available frames of each instance)
    frame_pair_views_per_instance: int       = 1
    # how to combine the sampled viewpoints of two instances:
    #   "aligned" → index-aligned (view i of A with view i of B), ~n_views pairs
    #   "cross"   → full cross-product (every view of A with every view of B), ~n_views^2
    frame_pair_view_mode: str                = "aligned"
    filter_is_real:    Optional[bool]        = None   # None = all, True = real only, False = synthetic only
    filter_score_zero: bool                  = False  # OpenTT: drop clips where both scores == 0
    # dataset-wide rigid transform mapping the raw object frame → canonical (GL) object
    # frame, applied to every loaded object (4x4, R|t convention)
    obj_gl_tform4x4_obj_raw: Optional[list]   = None   # [[r00,r01,r02,tx],[...],[...],[0,0,0,1]]
    # camera-convention transform: left-multiplies cam_tform4x4_obj_raw (e.g. CV→OpenGL flip)
    cam_tform4x4_cam_raw:  Optional[list]    = None   # [[r00,...],[...],[...],[0,0,0,1]]

    # O3B_Transform applied to every loaded item (dict with class_name + kwargs)
    transform:             Optional[dict]    = None

    # HuggingFace sharding: when set, items are materialised once into
    # path_preprocess/sharded/<sharded_name> and loaded from there on subsequent
    # runs.  sharded_override=True rebuilds the shards even if they already exist.
    sharded_name:      Optional[str]        = None
    sharded_override:  bool                 = False
    sharded_shard_size: int                 = 1000

    # extra per-dataset kwargs passed through to the implementation
    extra: dict                            = field(default_factory=dict)

    # ── serialisation ────────────────────────────────────────────────────────

    def to_yaml(self, path: Path) -> None:
        d = {
            "class_name":      self.class_name,
            "root":            str(self.root),
            "path_raw":        str(self.path_raw) if self.path_raw else None,
            "path_preprocess": str(self.path_preprocess) if self.path_preprocess else None,
            "split":           self.split,
            "item_type":       self.item_type.value,
            "batch_type":      self.batch_type.value,
            "scene_length":    self.scene_length,
            "mesh_type":       self.mesh_type,
            "modalities":        sorted(self.modalities)        if self.modalities        else None,
            "object_modalities": sorted(self.object_modalities) if self.object_modalities else None,
            "categories":        self.categories,
            "subsets":           self.subsets,
            "filter_count_max":  self.filter_count_max,
            "filter_has_kpts":    self.filter_has_kpts,
            "frame_pair_views_per_instance": self.frame_pair_views_per_instance,
            "frame_pair_view_mode":          self.frame_pair_view_mode,
            "filter_is_real":     self.filter_is_real,
            "filter_score_zero":  self.filter_score_zero,
            "obj_gl_tform4x4_obj_raw": self.obj_gl_tform4x4_obj_raw,
            "cam_tform4x4_cam_raw": self.cam_tform4x4_cam_raw,
            "transform":            self.transform,
            "sharded_name":       self.sharded_name,
            "sharded_override":   self.sharded_override,
            "sharded_shard_size": self.sharded_shard_size,
            "extra":           self.extra,
        }
        path.write_text(yaml.safe_dump(d, sort_keys=False))

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetConfig":
        return cls(
            class_name      = d["class_name"],
            root            = Path(d["root"]) if d.get("root") else Path("data"),
            path_raw        = Path(d["path_raw"]) if d.get("path_raw") else None,
            path_preprocess = Path(d["path_preprocess"]) if d.get("path_preprocess") else None,
            split           = d.get("split", "train"),
            item_type       = ItemType(d.get("item_type", "object")),
            batch_type      = BatchType(d.get("batch_type", "frame_object")),
            scene_length    = d.get("scene_length", 8),
            mesh_type       = d.get("mesh_type", "default"),
            modalities        = set(d["modalities"])        if d.get("modalities")        else None,
            object_modalities = set(d["object_modalities"]) if d.get("object_modalities") else None,
            categories       = d.get("categories"),
            subsets          = d.get("subsets"),
            filter_count_max = d.get("filter_count_max") or d.get("max_samples"),
            filter_has_kpts   = bool(d.get("filter_has_kpts",   False)),
            frame_pair_views_per_instance = int(d.get("frame_pair_views_per_instance", 1)),
            frame_pair_view_mode = d.get("frame_pair_view_mode", "aligned"),
            filter_is_real    = None if "filter_is_real" not in d or d["filter_is_real"] is None
                                 else bool(d["filter_is_real"]),
            filter_score_zero = bool(d.get("filter_score_zero", False)),
            obj_gl_tform4x4_obj_raw = d.get("obj_gl_tform4x4_obj_raw", d.get("obj_tform4x4")),
            cam_tform4x4_cam_raw = d.get("cam_tform4x4_cam_raw"),
            transform            = d.get("transform"),
            sharded_name         = d.get("sharded_name"),
            sharded_override     = bool(d.get("sharded_override", False)),
            sharded_shard_size   = int(d.get("sharded_shard_size", 1000)),
            extra           = d.get("extra", {}),
        )

    @classmethod
    def from_yaml(cls, path: Path, overrides: "list[str] | None" = None) -> "DatasetConfig":
        return cls.from_dict(_load_yaml_with_defaults(Path(path), overrides=overrides))


def _load_yaml_with_defaults(path: Path, overrides: "list[str] | None" = None) -> dict:
    """Load a YAML config, resolving defaults and ${} interpolations."""
    from o3b.io import _load_yaml_with_defaults as _impl
    return _impl(path, overrides=overrides)


# ── Dataset registry ──────────────────────────────────────────────────────────

_REGISTRY_DATASETS: dict[str, type["ConfigurableDataset"]] = {}

_CLASS_TO_MODULE: dict[str, str] = {
    "HouseCorr3D":  "o3b.dataset.housecorr3d.dataset",
    "DenseMatcher": "o3b.dataset.densematcher.dataset",
    "OpenTT":       "o3b.dataset.opentt.dataset",
}


def _ensure_dataset_imported(name: str) -> None:
    if name not in _REGISTRY_DATASETS and name in _CLASS_TO_MODULE:
        import importlib
        importlib.import_module(_CLASS_TO_MODULE[name])


def register_dataset(name: str):
    """Class decorator: @register_dataset("HouseCorr3D")"""
    def decorator(cls):
        _REGISTRY_DATASETS[name] = cls
        return cls
    return decorator


def build_dataset(cfg: DatasetConfig) -> "ConfigurableDataset":
    _ensure_dataset_imported(cfg.class_name)
    if cfg.class_name not in _REGISTRY_DATASETS:
        raise KeyError(
            f"Unknown dataset '{cfg.class_name}'. "
            f"Registered: {sorted(_REGISTRY_DATASETS)}"
        )
    return _REGISTRY_DATASETS[cfg.class_name](cfg)


# ── Transform helper ─────────────────────────────────────────────────────────

def _build_transform(transform_cfg: Optional[dict]):
    """Instantiate an O3B_Transform from a config dict, or return None."""
    if transform_cfg is None:
        return None
    from omegaconf import OmegaConf
    from o3b.data.transforms.transform import O3B_Transform
    # import all known transform modules so subclasses register themselves
    _TRANSFORM_MODULES = [
        "o3b.data.transforms.frame_object.crop_cam_bbox2d",
    ]
    import importlib
    for mod in _TRANSFORM_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError:
            pass
    cfg = OmegaConf.create(transform_cfg)
    return O3B_Transform.create_from_config(cfg)


# ── Base class ────────────────────────────────────────────────────────────────

class ConfigurableDataset(_TorchDataset):
    """
    Subclass this and implement _load_frame_object / _load_scene_object.
    Everything else — collation, DataLoader wiring — is handled here.
    """

    categories: tuple[str, ...] = ()  # override in subclasses with dataset-specific names

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self._index: list = []
        self._sharded = None       # HuggingFace Dataset when sharding is active
        self._transform = _build_transform(cfg.transform)
        self._setup()
        if cfg.sharded_name:
            self._setup_sharded()

    def _setup(self) -> None:
        pass

    # ── HuggingFace sharding ──────────────────────────────────────────────────

    _ITEM_TYPE_TO_CLS = {
        ItemType.OBJECT:       "Object",
        ItemType.OBJECT_PAIR:  "ObjectPair",
        ItemType.FRAME_OBJECT: "FrameObject",
        ItemType.SCENE_OBJECT: "SceneObject",
    }

    def _item_type_cls(self):
        from o3b.data.modalities import (  # noqa: F401
            FrameObject, SceneObject, Object, ObjectPair, FrameObjectPair,
        )
        return {
            ItemType.OBJECT: Object,
            ItemType.OBJECT_PAIR: ObjectPair,
            ItemType.FRAME_OBJECT: FrameObject,
            ItemType.FRAME_OBJECT_PAIR: FrameObjectPair,
            ItemType.SCENE_OBJECT: SceneObject,
        }[self.cfg.item_type]

    def _sharded_dir(self) -> Path:
        if self.cfg.path_preprocess is None:
            raise ValueError(
                "sharded_name is set but path_preprocess is None; "
                "cannot resolve the sharded dataset location."
            )
        return Path(self.cfg.path_preprocess) / "sharded" / self.cfg.sharded_name

    def _setup_sharded(self) -> None:
        """Load the sharded dataset, building it from raw items if necessary."""
        from o3b.dataset.sharding import (
            build_sharded_dataset_from_generator, item_to_record,
            read_sharded_dataset, write_sharded_dataset,
        )

        path = self._sharded_dir()
        if path.exists() and not self.cfg.sharded_override:
            print(f"Loading sharded dataset from {path}")
            self._sharded = read_sharded_dataset(path)
            return

        from tqdm import tqdm

        action = "Overriding" if path.exists() else "Building"
        n = len(self)
        print(f"{action} sharded dataset at {path} ({n} items)…")

        pbar = tqdm(total=n, desc="Sharding", unit="item")

        def _gen():
            for i in range(n):
                item = self._load_item(i)
                pbar.update(1)
                if item is not None:
                    yield item_to_record(item)

        hf = build_sharded_dataset_from_generator(_gen, writer_batch_size=self.cfg.sharded_shard_size)
        pbar.close()
        write_sharded_dataset(hf, path, shard_size=self.cfg.sharded_shard_size)
        self._sharded = read_sharded_dataset(path)
        print(f"Done. Wrote {len(self._sharded)} items → {path}")

    # ── item loading dispatch ─────────────────────────────────────────────────

    def _load_item(self, idx: int):
        if self.cfg.item_type == ItemType.OBJECT:
            return self._load_object(idx)
        elif self.cfg.item_type == ItemType.OBJECT_PAIR:
            return self._load_object_pair(idx)
        elif self.cfg.item_type == ItemType.FRAME_OBJECT:
            return self._load_frame_object(idx)
        elif self.cfg.item_type == ItemType.FRAME_OBJECT_PAIR:
            return self._load_frame_object_pair(idx)
        else:
            return self._load_scene_object(idx)

    # ── item loading (implement the one(s) matching your item_type) ──────────

    def _load_object(self, idx: int):
        raise NotImplementedError

    def _load_object_pair(self, idx: int) -> ObjectPair:
        raise NotImplementedError

    def _load_frame_object(self, idx: int) -> FrameObject:
        raise NotImplementedError

    def _load_frame_object_pair(self, idx: int) -> ObjectPair:
        raise NotImplementedError

    def _load_scene_object(self, idx: int) -> SceneObject:
        raise NotImplementedError

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        if self._sharded is not None:
            return len(self._sharded)
        raise NotImplementedError

    def __getitem__(self, idx: int):
        if self._sharded is not None:
            from o3b.dataset.sharding import record_to_item
            item = record_to_item(self._sharded[int(idx)], self._item_type_cls())
        else:
            item = self._load_item(idx)
        if self._transform is not None and item is not None:
            from o3b.data.datatypes.object import ObjectPair
            from o3b.data.datatypes.frame_object import FrameObjectPair
            if isinstance(item, (ObjectPair, FrameObjectPair)):
                # per-frame transforms (e.g. CropCamBBox2D) apply to each side
                from dataclasses import replace as _replace
                item = _replace(
                    item,
                    src_object=self._transform(item.src_object),
                    trgt_object=self._transform(item.trgt_object),
                )
            else:
                item = self._transform(item)
        return item

    # ── Collation ─────────────────────────────────────────────────────────────

    def collate_fn(
        self, samples: list[Union[FrameObject, SceneObject]]
    ) -> Union[FrameObjectBatch, SceneObjectBatch]:
        if self.cfg.batch_type == BatchType.FRAME_OBJECT:
            return collate_frame_objects(samples, include=self.cfg.modalities)
        else:
            return collate_scene_objects(samples, include=self.cfg.modalities)

    # ── DataLoader factory ────────────────────────────────────────────────────

    def build_loader(self, batch_size: int = 8, **kwargs):
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            **kwargs,
        )

    # ── Dataset-level CLI hooks (override in subclasses) ──────────────────────

    @classmethod
    def fetch(cls, cfg: "DatasetConfig", *, url: Optional[str] = None) -> None:
        raise NotImplementedError(f"{cls.__name__} does not implement fetch()")

    @classmethod
    def index(cls, cfg: "DatasetConfig", *, db: Optional[Path] = None, **kwargs) -> None:
        raise NotImplementedError(f"{cls.__name__} does not implement index()")

    @classmethod
    def visualize(
        cls,
        cfg: "DatasetConfig",
        *,
        db: Optional[Path] = None,
        limit: int = 20,
        object_id: Optional[str] = None,
        render: bool = False,
        debug: bool = False,
        **_,
    ) -> None:
        raise NotImplementedError(f"{cls.__name__} does not implement visualize()")
