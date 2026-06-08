"""Scoreboard bbox annotation and VLM-based score reading for OpenTT.

Two-phase workflow
------------------
Phase 1 — annotate (interactive, once per video):
    o3b dataset preprocess -d opentt --annotate
    Shows a representative frame for each video and asks the user to draw
    three bounding boxes: whole scoreboard, left score, right score.
    Saved to {path_preprocess}/video_bboxes.json.

Phase 2 — VLM inference (per-frame, resumable):
    o3b dataset preprocess -d opentt --model Qwen/Qwen3-VL-2B-Instruct --device cuda:0
    Crops the left/right score regions using the saved bboxes and passes each
    crop to a VLM to read the integer score.
    Results are written to scoreboards.db.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

BBOXES_FILENAME = "video_bboxes.json"


# ── SQLite schema ─────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scoreboards (
    video_name        TEXT    NOT NULL,
    frame_idx         INTEGER NOT NULL,
    bbox_x1           REAL,
    bbox_y1           REAL,
    bbox_x2           REAL,
    bbox_y2           REAL,
    score_left        INTEGER,
    score_right       INTEGER,
    score_left_point  INTEGER,
    score_right_point INTEGER,
    score_raw         TEXT,
    PRIMARY KEY (video_name, frame_idx)
)
"""

_EXTRA_COLUMNS = [
    ("score_left_point",  "INTEGER"),
    ("score_right_point", "INTEGER"),
]


def _migrate_db(con: sqlite3.Connection) -> None:
    existing = {row[1] for row in con.execute("PRAGMA table_info(scoreboards)").fetchall()}
    for col_name, col_type in _EXTRA_COLUMNS:
        if col_name not in existing:
            con.execute(f"ALTER TABLE scoreboards ADD COLUMN {col_name} {col_type}")
    con.commit()


# ── Phase 1: interactive bbox annotation ──────────────────────────────────────

def annotate_videos(dataset, bboxes_path: Path, *, override: bool = False) -> None:
    """For each video draw scoreboard / left-score / right-score bboxes interactively."""
    bboxes_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if bboxes_path.exists():
        with bboxes_path.open() as f:
            existing = json.load(f)

    seen: set[str] = set()
    videos: list[tuple[str, Path]] = []
    for clip in dataset._clips:
        name = clip["name"]
        if name not in seen:
            seen.add(name)
            videos.append((name, clip["path"]))

    n_done = 0
    for name, video_path in videos:
        if not override and name in existing:
            print(f"  skip  {name}  (already annotated; use --override to redo)")
            continue

        n_frames = _video_frame_count(video_path)
        result = _annotate_video(name, video_path, n_frames)
        if result is None:
            print(f"  skipped {name}")
            continue

        existing[name] = result
        with bboxes_path.open("w") as f:
            json.dump(existing, f, indent=2)
        print(f"  saved  {name}  → {bboxes_path}")
        n_done += 1

    print(f"\nAnnotated {n_done} video(s). Bboxes → {bboxes_path}")


def _annotate_video(name: str, video_path: Path, n_frames: int) -> Optional[dict]:
    """Show a representative frame; ask the user to draw 3 ROIs. Returns bbox dict or None."""
    import cv2

    frame_idx = n_frames // 2
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        print(f"  ERROR: could not read frame {frame_idx} from {video_path}", file=sys.stderr)
        return None

    print(f"\n  {name}  ({n_frames} frames, showing frame {frame_idx})")
    print("  Drag to draw each box, press SPACE or ENTER to confirm, C to cancel.")

    colors = {
        "scoreboard": (0, 255, 0),
        "left_score":  (255, 80, 80),
        "right_score": (80, 80, 255),
    }

    # ── Step 1: draw scoreboard bbox on the full frame ────────────────────────
    win = f"{name} - 1/3  whole scoreboard region"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(win, frame_bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        print("  Cancelled.")
        return None

    sb_x, sb_y, sb_w, sb_h = (int(v) for v in roi)
    sb_bb = [sb_x, sb_y, sb_x + sb_w, sb_y + sb_h]
    print(f"    scoreboard: {sb_bb}")

    # ── Step 2 & 3: crop to scoreboard, draw left/right score bboxes ─────────
    sb_crop = frame_bgr[sb_y: sb_y + sb_h, sb_x: sb_x + sb_w]

    score_regions = [
        ("left_score",  "2/3  left player score  (large number, left half)"),
        ("right_score", "3/3  right player score (large number, right half)"),
    ]

    rel_bboxes: dict = {}   # relative to scoreboard crop
    for key, label in score_regions:
        display = sb_crop.copy()
        for k, bb in rel_bboxes.items():
            x1, y1, x2, y2 = bb
            cv2.rectangle(display, (x1, y1), (x2, y2), colors[k], 2)
            cv2.putText(display, k, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[k], 2)

        win = f"{name} - {label}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        roi = cv2.selectROI(win, display, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()

        if roi == (0, 0, 0, 0):
            print("  Cancelled.")
            return None

        rx, ry, rw, rh = (int(v) for v in roi)
        rel_bboxes[key] = [rx, ry, rx + rw, ry + rh]
        print(f"    {key}: {rel_bboxes[key]}  (relative to scoreboard crop)")

    # Convert score bboxes back to full-frame coordinates
    bboxes = {"scoreboard": sb_bb}
    for key, (rx1, ry1, rx2, ry2) in rel_bboxes.items():
        bboxes[key] = [sb_x + rx1, sb_y + ry1, sb_x + rx2, sb_y + ry2]
        print(f"    {key}: {bboxes[key]}  (full frame)")

    # Confirmation preview on full frame
    display = frame_bgr.copy()
    for k, bb in bboxes.items():
        x1, y1, x2, y2 = bb
        cv2.rectangle(display, (x1, y1), (x2, y2), colors[k], 2)
        cv2.putText(display, k, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[k], 2)
    win = f"{name} - confirm? (any key = accept, Q = redo)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, display)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

    if key in (ord("q"), ord("Q")):
        print("  Redoing annotation for this video…")
        return _annotate_video(name, video_path, n_frames)

    return bboxes


# ── Phase 2: VLM score extraction ─────────────────────────────────────────────

def _load_model(model_id: str, device: str):
    """Load VLM model + processor. Supports Qwen3-VL and generic image-text models."""
    from transformers import AutoProcessor

    print(f"  Loading model {model_id} on {device} …")

    if "qwen3-vl" in model_id.lower():
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map=device,
        )
    else:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map=device,
        )

    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    print(f"  Model ready.\n")
    return model, processor


def run_vlm_from_dataset(
    dataset,
    db_path: Path,
    bboxes_path: Path,
    *,
    model_id: str,
    device: str = "cpu",
    override: bool = False,
    debug: bool = False,
) -> None:
    """Crop annotated score regions per frame, run VLM, store results in SQLite."""
    if not bboxes_path.exists():
        print(
            f"ERROR: bbox annotations not found at {bboxes_path}.\n"
            "Run with --annotate first to draw the score bounding boxes.",
            file=sys.stderr,
        )
        return

    with bboxes_path.open() as f:
        video_bboxes: dict = json.load(f)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, timeout=30)
    con.execute(_SCHEMA)
    con.commit()
    _migrate_db(con)

    if override:
        done: set[tuple[str, int]] = set()
    else:
        done = {
            (row[0], int(row[1]))
            for row in con.execute("SELECT video_name, frame_idx FROM scoreboards").fetchall()
        }

    n_clips   = len(dataset)
    n_skipped = 0 if override else sum(
        1 for clip in dataset._clips
        if all((clip["name"], fi) in done for fi in clip["indices"])
    )
    print(f"  {len(done)} frames already in DB"
          + (" (override — will re-process all)" if override else ""))
    print(f"  {n_clips} clips total — {n_skipped} fully done, "
          f"{n_clips - n_skipped} to process\n")

    model, processor = _load_model(model_id, device)

    try:
        for idx in range(n_clips):
            clip    = dataset._clips[idx]
            name    = clip["name"]
            indices = clip["indices"]

            if name not in video_bboxes:
                continue

            bboxes  = video_bboxes[name]
            sb_bb   = bboxes.get("scoreboard")
            l_bb    = bboxes.get("left_score")
            r_bb    = bboxes.get("right_score")

            pending = [i for i, fi in enumerate(indices) if (name, fi) not in done]
            if not pending:
                continue

            scene = dataset[idx]
            if scene.rgbs is None:
                continue

            fis = [indices[i] for i in pending]
            sys.stdout.write(
                f"  [{idx + 1:>{len(str(n_clips))}}/{n_clips}]"
                f"  {name}  frames {fis[0]:,}…{fis[-1]:,}"
                f"  ({len(fis)} frame{'s' if len(fis) != 1 else ''})\n"
            )
            sys.stdout.flush()

            for i in pending:
                fi    = indices[i]
                frame = scene.rgbs[i]
                img   = Image.fromarray(
                    (frame.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
                )

                left_crop  = img.crop(l_bb) if l_bb else None
                right_crop = img.crop(r_bb) if r_bb else None

                if debug:
                    if left_crop:
                        _show_crop(left_crop)
                    if right_crop:
                        _show_crop(right_crop)

                left_score  = _vlm_score(left_crop,  model, processor, debug=debug) if left_crop  else None
                right_score = _vlm_score(right_crop, model, processor, debug=debug) if right_crop else None

                con.execute(
                    "INSERT OR REPLACE INTO scoreboards "
                    "(video_name, frame_idx, "
                    " bbox_x1, bbox_y1, bbox_x2, bbox_y2, "
                    " score_left, score_right, "
                    " score_left_point, score_right_point, score_raw) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        name, fi,
                        sb_bb[0] if sb_bb else None, sb_bb[1] if sb_bb else None,
                        sb_bb[2] if sb_bb else None, sb_bb[3] if sb_bb else None,
                        left_score, right_score,
                        left_score, right_score,
                        None,
                    ),
                )
                done.add((name, fi))

                sys.stdout.write(
                    f"    frame {fi:7,d}  L:{left_score}  R:{right_score}\n"
                )
                sys.stdout.flush()

            con.commit()

    finally:
        con.close()

    print(f"\nVLM inference done. Results → {db_path}")
    postprocess_scores(db_path)


_VLM_PROMPT = (
    "What integer number is shown in this image? "
    "Reply with ONLY the integer, nothing else."
)


def _vlm_score(
    crop: Image.Image,
    model,
    processor,
    *,
    debug: bool = False,
) -> Optional[int]:
    """Pass a score-region crop to the VLM and return the parsed integer."""
    import re
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": crop},
                {"type": "text", "text": _VLM_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=16)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    if debug:
        sys.stdout.write(f"      VLM raw output: {raw!r}\n")
        sys.stdout.flush()

    raw = raw.strip()
    try:
        return int(raw)
    except ValueError:
        m = re.search(r"\d+", raw)
        return int(m.group()) if m else None


def _show_crop(crop: Image.Image) -> None:
    import numpy as np
    from od3d_basic.cv.visual.show import show_img
    rgb = torch.from_numpy(
        np.array(crop.convert("RGB"))
    ).float().permute(2, 0, 1) / 255.0
    show_img(rgb)


def postprocess_scores(db_path: Path) -> None:
    """Enforce score monotonicity per video.

    A score is valid only if it equals the previous score or previous score + 1.
    Invalid (or None) scores are replaced with the previous valid score.
    """
    if not db_path.exists():
        print(f"No database found at {db_path} — nothing to post-process.")
        return

    con = sqlite3.connect(db_path, timeout=30)
    try:
        rows = con.execute(
            "SELECT video_name, frame_idx, score_left_point, score_right_point "
            "FROM scoreboards ORDER BY video_name, frame_idx"
        ).fetchall()

        prev: dict[str, tuple] = {}   # video_name -> (left, right)
        updates: list = []

        for video_name, frame_idx, left, right in rows:
            # Clamp out-of-range readings to 0 before monotonicity check
            if left  is not None and not (0 <= left  <= 11):
                left  = 0
            if right is not None and not (0 <= right <= 11):
                right = 0

            p_left, p_right = prev.get(video_name, (None, None))

            new_left = left
            if p_left is not None:
                if left is None or (left != p_left and left != p_left + 1):
                    new_left = p_left

            new_right = right
            if p_right is not None:
                if right is None or (right != p_right and right != p_right + 1):
                    new_right = p_right

            if new_left != left or new_right != right:
                updates.append((new_left, new_right, new_left, new_right, video_name, frame_idx))

            prev[video_name] = (
                new_left  if new_left  is not None else p_left,
                new_right if new_right is not None else p_right,
            )

        if updates:
            con.executemany(
                "UPDATE scoreboards "
                "SET score_left=?, score_right=?, score_left_point=?, score_right_point=? "
                "WHERE video_name=? AND frame_idx=?",
                updates,
            )
            con.commit()
            print(f"  Post-processed {len(updates)} frame(s) with out-of-range scores.")
        else:
            print("  Post-processing: all scores already valid.")
    finally:
        con.close()


def _video_frame_count(path: Path) -> int:
    import cv2
    cap = cv2.VideoCapture(str(path))
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(n, 0)


# ── public entry point ────────────────────────────────────────────────────────

def run_preprocess_from_dataset(
    dataset,
    db_path: Path,
    bboxes_path: Path,
    *,
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
    device: str = "cpu",
    annotate: bool = False,
    override: bool = False,
    debug: bool = False,
    remove: bool = False,
) -> None:
    if remove:
        _remove_scores(db_path)
    elif annotate:
        annotate_videos(dataset, bboxes_path, override=override)
    else:
        run_vlm_from_dataset(
            dataset, db_path, bboxes_path,
            model_id=model_id, device=device,
            override=override, debug=debug,
        )



def _remove_scores(db_path: Path) -> None:
    if not db_path.exists():
        print(f"No database found at {db_path} — nothing to remove.")
        return
    import sqlite3
    con = sqlite3.connect(db_path, timeout=30)
    n = con.execute("SELECT COUNT(*) FROM scoreboards").fetchone()[0]
    con.execute("DELETE FROM scoreboards")
    con.commit()
    con.close()
    print(f"Removed {n} rows from scoreboards table in {db_path}.")
