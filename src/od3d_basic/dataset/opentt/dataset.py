"""OpenTT – Extended OpenTTGames dataset.

Annotations (JSON) are fetched from:
  https://github.com/moamal01/table_tennis_data

Videos (mp4) are downloaded from:
  https://lab.osai.ai/datasets/openttgames/data/

Each JSON file maps frame numbers to event labels, e.g.:
  {"44": "bounce", "58": "left_forehand_lob right_leaning both_feet_planted", "98": "net"}

Events fall into three categories:
  - ball events   : "bounce", "net"
  - point events  : "left_net", "right_winner", "left_double_bounce", etc.
  - stroke events : "<side>_<hand>_<technique> <lean> <feet>"  (space-separated triple)
  - empty         : "empty_event"
Frames without an annotation receive event=None.

Directory layout after fetch():
  {path_raw}/
      annotations/
          train/  game_1.json … game_5.json
          test/   test_1.json … test_7.json
      videos/
          train/  game_1.mp4 … game_5.mp4
          test/   test_1.mp4 … test_7.mp4

After index():
  {path_preprocess}/
      manifest.json    # {video_name: n_frames} — speeds up repeated _setup() calls

DatasetConfig.extra knobs:
  frame_stride  int   step between consecutive sampled frames within a clip  (default 1)
  clip_stride   int   step between clip start frames                         (default scene_length * frame_stride)
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import torch

from od3d_basic.data.datatypes.scene import Scene, SceneBatch, collate_scenes
from od3d_basic.dataset.dataset import (
    ConfigurableDataset,
    DatasetConfig,
    register_dataset,
)

# ── constants ─────────────────────────────────────────────────────────────────

_TRAIN_NAMES = [f"game_{i}" for i in range(1, 6)]
_TEST_NAMES  = [f"test_{i}"  for i in range(1, 8)]

_VIDEO_BASE = "https://lab.osai.ai/datasets/openttgames/data"
_ANNO_RAW   = (
    "https://raw.githubusercontent.com/moamal01/table_tennis_data/main"
    "/data/raw/game_data"
)


# ── dataset class ─────────────────────────────────────────────────────────────

@register_dataset("OpenTT")
class OpenTT(ConfigurableDataset):
    """Sliding-window scene dataset built from the Extended OpenTTGames videos.

    Each item is a Scene containing:
      rgbs   (T, H, W, 3) float32 in [0, 1]
      events  list of T entries: str event label or None
    """

    # ── path helpers ───────────────────────────────────────────────────────────

    @property
    def _path_raw(self) -> Path:
        return self.cfg.path_raw or self.cfg.root / "opentt"

    @property
    def _path_preprocess(self) -> Path:
        return self.cfg.path_preprocess or self._path_raw

    @property
    def _path_annotations(self) -> Path:
        return self._path_raw / "annotations"

    @property
    def _path_videos(self) -> Path:
        return self._path_raw / "videos"

    @property
    def _manifest_path(self) -> Path:
        return self._path_preprocess / "manifest.json"

    # ── setup ──────────────────────────────────────────────────────────────────

    def _setup(self) -> None:
        self._annotations: dict[str, dict[int, str]] = {}
        self._clips: list[dict] = []

        frame_stride: int = int(self.cfg.extra.get("frame_stride", 1))
        scene_len:    int = self.cfg.scene_length
        clip_stride:  int = int(self.cfg.extra.get("clip_stride", scene_len * frame_stride))

        # Load manifest (written by index()) if available; avoids re-opening every video.
        manifest: dict[str, int] = {}
        if self._manifest_path.exists():
            with self._manifest_path.open() as fh:
                manifest = json.load(fh)

        split = self.cfg.split
        if split in ("train", "all"):
            self._ingest_split("train", _TRAIN_NAMES, frame_stride, scene_len, clip_stride, manifest)
        if split in ("test", "all"):
            self._ingest_split("test",  _TEST_NAMES,  frame_stride, scene_len, clip_stride, manifest)

        if self.cfg.filter_count_max:
            self._clips = self._clips[: self.cfg.filter_count_max]

        # Load scoreboard DB if it exists: {video_name: (sorted_frame_idxs, {fi: entry})}
        self._scoreboards: dict[str, tuple[list, dict]] = _load_scoreboards(
            self._path_preprocess / "scoreboards.db"
        )

    def _ingest_split(
        self,
        split: str,
        names: list[str],
        frame_stride: int,
        scene_len: int,
        clip_stride: int,
        manifest: dict[str, int],
    ) -> None:
        for name in names:
            anno_file  = self._path_annotations / split / f"{name}.json"
            video_file = self._path_videos      / split / f"{name}.mp4"

            if not anno_file.exists() or not video_file.exists():
                continue

            with anno_file.open() as fh:
                raw = json.load(fh)
            self._annotations[name] = {int(k): v for k, v in raw.items()}

            n_frames = manifest.get(name) or _video_frame_count(video_file)
            window   = scene_len * frame_stride

            for start in range(0, n_frames - window + 1, clip_stride):
                indices = [start + i * frame_stride for i in range(scene_len)]
                self._clips.append(
                    {
                        "name":    name,
                        "split":   split,
                        "path":    video_file,
                        "indices": indices,
                    }
                )

    # ── Dataset protocol ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, idx: int) -> Scene:
        clip   = self._clips[idx]
        name   = clip["name"]
        anno   = self._annotations.get(name, {})
        rgbs   = _extract_frames(clip["path"], clip["indices"])
        events = [anno.get(fi) for fi in clip["indices"]]

        scoreboards = None
        if self._scoreboards:
            scoreboards = [
                _nearest_scoreboard(self._scoreboards, name, fi)
                for fi in clip["indices"]
            ]

        return Scene(
            scene_id    = f"{name}_{clip['indices'][0]:07d}",
            rgbs        = rgbs,
            events      = events,
            scoreboards = scoreboards,
        )

    # ── collation ──────────────────────────────────────────────────────────────

    def collate_fn(self, samples: list[Scene]) -> SceneBatch:
        return collate_scenes(samples)

    # ── CLI hooks ──────────────────────────────────────────────────────────────

    @classmethod
    def preprocess(
        cls,
        cfg: DatasetConfig,
        *,
        db: Optional[Path] = None,
        model_id: str = "Qwen/Qwen3.5-4B",
        frame_stride: int = 30,
        batch_size: int = 4,
        video: Optional[str] = None,
        device: str = "auto",
        override: bool = False,
        debug: bool = False,
    ) -> None:
        """Detect scoreboards and read scores via a Qwen VLM; store in SQLite.

        Instantiates the OpenTT dataset configured for preprocessing (non-
        overlapping clips of *batch_size* frames sampled every *frame_stride*
        frames), then iterates over Scene instances and passes scene.rgbs to
        the VLM.  Results are written to *db* (default:
        {path_preprocess}/scoreboards.db).  Already-processed frames are
        skipped, so the command is safely resumable.

        Args:
            cfg:          dataset configuration (paths, split).
            db:           SQLite output path; defaults to {path_preprocess}/scoreboards.db.
            model_id:     Hugging Face model ID for the VLM.
            frame_stride: sample every Nth frame (default 30 → ~4 fps at 120 fps).
            batch_size:   frames per model forward pass (= scene_length of the
                          preprocess dataset).
            video:        if set, restrict to this single video name (e.g. "game_1").
            device:       "auto", "cuda:0", "cpu", …
        """
        import types
        from od3d_basic.dataset.opentt.preprocess import run_preprocess_from_dataset

        path_raw        = Path(cfg.path_raw)        if cfg.path_raw        else Path(cfg.root) / "opentt"
        path_preprocess = Path(cfg.path_preprocess) if cfg.path_preprocess else path_raw
        db_path         = Path(db) if db else path_preprocess / "scoreboards.db"

        # Lightweight config for the preprocess dataset:
        #   - scene_length = batch_size so each Scene holds exactly one forward-pass batch
        #   - clip_stride  = batch_size * frame_stride → non-overlapping, exhaustive coverage
        preprocess_cfg = types.SimpleNamespace(
            path_raw         = path_raw,
            path_preprocess  = path_preprocess,
            split            = cfg.split,
            scene_length     = batch_size,
            filter_count_max = cfg.filter_count_max,
            extra            = {
                "frame_stride": frame_stride,
                "clip_stride" : batch_size * frame_stride,
            },
        )

        dataset = cls(preprocess_cfg)

        if video:
            dataset._clips = [c for c in dataset._clips if c["name"] == video]

        if not dataset._clips:
            print(
                "No clips found. Run 'o3x dataset fetch -d opentt' first.",
                file=sys.stderr,
            )
            return

        n_videos = len({c["name"] for c in dataset._clips})
        print(
            f"Scoreboard preprocessing: {len(dataset._clips)} clips "
            f"({n_videos} video(s), stride={frame_stride}, batch={batch_size})"
            f"  →  {db_path}"
        )
        run_preprocess_from_dataset(dataset, db_path, model_id=model_id, device=device,
                                    override=override, debug=debug)

    @classmethod
    def fetch(cls, cfg: DatasetConfig, *, url: Optional[str] = None) -> None:
        """Download annotation JSONs from GitHub and MP4 videos from the CDN."""
        path_raw = cfg.path_raw or cfg.root / "opentt"

        print("=== OpenTT: fetching annotations ===")
        for split, names in [("train", _TRAIN_NAMES), ("test", _TEST_NAMES)]:
            anno_dir = path_raw / "annotations" / split
            anno_dir.mkdir(parents=True, exist_ok=True)
            for name in names:
                dest = anno_dir / f"{name}.json"
                if dest.exists():
                    print(f"  skip  annotations/{split}/{name}.json  (already exists)")
                    continue
                src = f"{_ANNO_RAW}/{split}/{name}.json"
                print(f"  fetch {src}")
                urllib.request.urlretrieve(src, dest, _progress)
                print()

        print("\n=== OpenTT: fetching videos ===")
        for split, names in [("train", _TRAIN_NAMES), ("test", _TEST_NAMES)]:
            vid_dir = path_raw / "videos" / split
            vid_dir.mkdir(parents=True, exist_ok=True)
            for name in names:
                dest = vid_dir / f"{name}.mp4"
                if dest.exists():
                    print(f"  skip  videos/{split}/{name}.mp4  (already exists)")
                    continue
                src = f"{_VIDEO_BASE}/{name}.mp4"
                print(f"  fetch {src}")
                urllib.request.urlretrieve(src, dest, _progress)
                print()

        print("\nDone. Run 'o3x dataset index -d opentt' next.")

    @classmethod
    def index(cls, cfg: DatasetConfig, *, db: Optional[Path] = None) -> None:
        """Scan all available videos, record frame counts, write manifest.json.

        The manifest lets _setup() skip re-opening every mp4 file on subsequent
        dataset instantiations.
        """
        path_raw        = cfg.path_raw        or cfg.root / "opentt"
        path_preprocess = cfg.path_preprocess or path_raw
        manifest_path   = db or Path(path_preprocess) / "manifest.json"

        Path(path_preprocess).mkdir(parents=True, exist_ok=True)

        manifest: dict[str, int] = {}
        total_clips = 0

        frame_stride: int = int(cfg.extra.get("frame_stride", 1))
        scene_len:    int = cfg.scene_length
        clip_stride:  int = int(cfg.extra.get("clip_stride", scene_len * frame_stride))

        print(f"{'Video':<20}  {'Split':<6}  {'Frames':>8}  {'Clips':>7}")
        print("-" * 50)

        for split, names in [("train", _TRAIN_NAMES), ("test", _TEST_NAMES)]:
            vid_dir  = Path(path_raw) / "videos"  / split
            anno_dir = Path(path_raw) / "annotations" / split

            for name in names:
                video_file = vid_dir  / f"{name}.mp4"
                anno_file  = anno_dir / f"{name}.json"

                if not video_file.exists():
                    print(f"  {'(missing)':20}  {split:<6}  {'—':>8}  {'—':>7}  {name}")
                    continue

                n_frames = _video_frame_count(video_file)
                manifest[name] = n_frames

                window     = scene_len * frame_stride
                n_clips    = max(0, (n_frames - window) // clip_stride + 1)
                total_clips += n_clips

                anno_ok = "✓" if anno_file.exists() else "✗ no annotation"
                print(f"  {name:<20}  {split:<6}  {n_frames:>8,}  {n_clips:>7,}  {anno_ok}")

        print("-" * 50)
        print(f"  {'TOTAL':<20}  {'':6}  {'':8}  {total_clips:>7,}")

        with manifest_path.open("w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"\nManifest written → {manifest_path}")

    @classmethod
    def visualize(
        cls,
        cfg: DatasetConfig,
        *,
        db: Optional[Path] = None,
        limit: int = 4,
        object_id: Optional[str] = None,
        render: bool = False,
        render_frames: int = 0,
        renderer: str = "pyrender",
        debug: bool = False,
    ) -> None:
        """Show up to *limit* scenes using Scene.viz().

        *object_id* can be used to restrict display to clips from a specific video
        (e.g. --object-id game_1).
        """
        dataset = cls(cfg)

        if not dataset._clips:
            print(
                "No clips found. Run 'o3x dataset fetch -d opentt' first, then make sure\n"
                f"  path_raw → {dataset._path_raw}\n"
                "contains annotations/ and videos/ sub-directories.",
                file=sys.stderr,
            )
            return

        clips = dataset._clips
        if object_id:
            clips = [c for c in clips if c["name"] == object_id]
            if not clips:
                print(
                    f"No clips found for video '{object_id}'. "
                    f"Available: {sorted({c['name'] for c in dataset._clips})}",
                    file=sys.stderr,
                )
                return

        n = min(limit, len(clips))
        print(
            f"Visualising {n} / {len(clips)} clips"
            + (f" (filtered to '{object_id}')" if object_id else "")
            + f"  [{dataset._path_raw}]"
        )

        for i in range(n):
            clip_info = clips[i]
            # Re-index into the dataset by matching the stored clip
            try:
                idx = dataset._clips.index(clip_info)
            except ValueError:
                continue
            scene = dataset[idx]
            strip = scene.viz(show=True)
            if debug and strip is not None:
                print(f"  [{i}] {scene.scene_id}  strip: {tuple(strip.shape)}  "
                      f"events: {scene.events}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _video_frame_count(path: Path) -> int:
    """Return total frame count for an mp4 via OpenCV."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(n, 0)


def _extract_frames(path: Path, indices: list[int]) -> Optional[torch.Tensor]:
    """Seek to each index in an mp4 and return (T, H, W, 3) float32 in [0, 1]."""
    import cv2
    cap    = cv2.VideoCapture(str(path))
    frames = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame).float() / 255.0)
    cap.release()
    return torch.stack(frames) if frames else None


def _load_scoreboards(db_path: Path) -> dict:
    """Load a scoreboards SQLite DB into memory for O(log n) per-frame lookup.

    Returns {video_name: (sorted_frame_idxs, {frame_idx: entry_dict})}
    where entry_dict may contain:
      "bbox"             – whole scoreboard region
      "score_left"       – left point score (backward-compat)
      "score_right"      – right point score (backward-compat)
      "bbox_left_point"  – bbox of left point score (big number)
      "score_left_point"
      "bbox_left_game"   – bbox of left game score (small number)
      "score_left_game"
      "bbox_right_point" – bbox of right point score (big number)
      "score_right_point"
      "bbox_right_game"  – bbox of right game score (small number)
      "score_right_game"
    Returns {} if *db_path* does not exist.
    """
    if not db_path.exists():
        return {}

    import sqlite3

    con = sqlite3.connect(db_path)
    try:
        existing_cols = {row[1] for row in con.execute("PRAGMA table_info(scoreboards)")}

        base_cols = [
            "video_name", "frame_idx",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "score_left", "score_right",
        ]
        extra_cols = [
            "bbox_left_point_x1",  "bbox_left_point_y1",
            "bbox_left_point_x2",  "bbox_left_point_y2",  "score_left_point",
            "bbox_left_game_x1",   "bbox_left_game_y1",
            "bbox_left_game_x2",   "bbox_left_game_y2",   "score_left_game",
            "bbox_right_point_x1", "bbox_right_point_y1",
            "bbox_right_point_x2", "bbox_right_point_y2", "score_right_point",
            "bbox_right_game_x1",  "bbox_right_game_y1",
            "bbox_right_game_x2",  "bbox_right_game_y2",  "score_right_game",
        ]
        all_cols = base_cols + [c for c in extra_cols if c in existing_cols]

        rows = con.execute(
            f"SELECT {', '.join(all_cols)} FROM scoreboards"
        ).fetchall()
    finally:
        con.close()

    raw: dict[str, dict[int, dict]] = {}
    for row in rows:
        r = dict(zip(all_cols, row))
        video_name = r["video_name"]
        fi = int(r["frame_idx"])

        entry: dict = {}
        if r.get("bbox_x1") is not None:
            entry["bbox"] = [r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"]]
        if r.get("score_left") is not None:
            entry["score_left"] = int(r["score_left"])
        if r.get("score_right") is not None:
            entry["score_right"] = int(r["score_right"])

        for region in ("left_point", "left_game", "right_point", "right_game"):
            x1k = f"bbox_{region}_x1"
            if r.get(x1k) is not None:
                entry[f"bbox_{region}"] = [
                    r[f"bbox_{region}_x1"], r[f"bbox_{region}_y1"],
                    r[f"bbox_{region}_x2"], r[f"bbox_{region}_y2"],
                ]
            sc = r.get(f"score_{region}")
            if sc is not None:
                entry[f"score_{region}"] = int(sc)

        raw.setdefault(video_name, {})[fi] = entry

    result: dict[str, tuple] = {}
    for name, frame_dict in raw.items():
        result[name] = (sorted(frame_dict.keys()), frame_dict)

    return result


def _nearest_scoreboard(
    scoreboards: dict,
    name: str,
    fi: int,
    max_dist: int = 45,
) -> Optional[dict]:
    """Return the nearest stored scoreboard entry for frame *fi*, or None.

    Searches within *max_dist* frames of *fi* (default 45 ≈ 1.5 × stride=30).
    """
    entry = scoreboards.get(name)
    if entry is None:
        return None
    sorted_keys, frame_dict = entry
    if not sorted_keys:
        return None

    import bisect

    pos = bisect.bisect_left(sorted_keys, fi)
    candidates: list[int] = []
    if pos < len(sorted_keys):
        candidates.append(sorted_keys[pos])
    if pos > 0:
        candidates.append(sorted_keys[pos - 1])

    nearest = min(candidates, key=lambda k: abs(k - fi))
    if abs(nearest - fi) <= max_dist:
        return frame_dict[nearest]
    return None


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    """urlretrieve reporthook — inline progress bar."""
    if total_size <= 0:
        sys.stdout.write(f"\r  {block_num * block_size / 1e6:.1f} MB")
    else:
        pct  = min(100.0, block_num * block_size * 100.0 / total_size)
        done = int(pct / 2)
        sys.stdout.write(f"\r  [{'=' * done}{' ' * (50 - done)}] {pct:5.1f}%")
    sys.stdout.flush()
