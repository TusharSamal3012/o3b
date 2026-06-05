"""Scoreboard detection and score reading for OpenTT via a Qwen VLM.

For every frame in an OpenTT dataset instance the model is asked to:
  1. Locate the on-screen scoreboard and return its 2-D bounding box.
  2. Find the four individual score regions (left/right × point/game) with
     their bounding boxes and integer values.

The dataset is configured with scene_length = batch_size and the preprocess
frame stride, so each Scene already contains exactly one batch of frames.
scene.rgbs (T, H, W, 3) float32 in [0, 1] is converted to PIL images and
passed to the VLM; results are keyed by (video_name, frame_idx) from the
dataset's internal clip list.

Results are written to a SQLite database with the schema:

  CREATE TABLE scoreboards (
      video_name   TEXT     NOT NULL,
      frame_idx    INTEGER  NOT NULL,
      bbox_x1      REAL,          -- whole scoreboard region (null if not found)
      bbox_y1      REAL,
      bbox_x2      REAL,
      bbox_y2      REAL,
      score_left   INTEGER,       -- left point score (backward-compat alias)
      score_right  INTEGER,       -- right point score (backward-compat alias)
      -- left player's point score (the big number)
      bbox_left_point_x1  REAL,
      bbox_left_point_y1  REAL,
      bbox_left_point_x2  REAL,
      bbox_left_point_y2  REAL,
      score_left_point    INTEGER,
      -- left player's game score (the small number)
      bbox_left_game_x1   REAL,
      bbox_left_game_y1   REAL,
      bbox_left_game_x2   REAL,
      bbox_left_game_y2   REAL,
      score_left_game     INTEGER,
      -- right player's point score (the big number)
      bbox_right_point_x1  REAL,
      bbox_right_point_y1  REAL,
      bbox_right_point_x2  REAL,
      bbox_right_point_y2  REAL,
      score_right_point    INTEGER,
      -- right player's game score (the small number)
      bbox_right_game_x1   REAL,
      bbox_right_game_y1   REAL,
      bbox_right_game_x2   REAL,
      bbox_right_game_y2   REAL,
      score_right_game     INTEGER,
      score_raw    TEXT,
      PRIMARY KEY (video_name, frame_idx)
  )

The run is fully resumable: already-stored (video_name, frame_idx) pairs are
skipped, so the command can be interrupted and restarted safely.
Pass override=True to force re-processing of already-stored frames.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

# Center-crop fractions applied before every VLM call.
_CROP_HEIGHT_FRAC = 0.40
_CROP_WIDTH_FRAC  = 0.25


def _center_crop(img: Image.Image) -> Image.Image:
    W, H = img.size
    new_h = max(1, int(H * _CROP_HEIGHT_FRAC))
    new_w = max(1, int(W * _CROP_WIDTH_FRAC))
    top  = (H - new_h) // 2
    left = (W - new_w) // 2
    return img.crop((left, top, left + new_w, top + new_h))


# ── SQLite schema ─────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scoreboards (
    video_name   TEXT    NOT NULL,
    frame_idx    INTEGER NOT NULL,
    bbox_x1      REAL,
    bbox_y1      REAL,
    bbox_x2      REAL,
    bbox_y2      REAL,
    score_left   INTEGER,
    score_right  INTEGER,
    bbox_left_point_x1  REAL,
    bbox_left_point_y1  REAL,
    bbox_left_point_x2  REAL,
    bbox_left_point_y2  REAL,
    score_left_point    INTEGER,
    bbox_left_game_x1   REAL,
    bbox_left_game_y1   REAL,
    bbox_left_game_x2   REAL,
    bbox_left_game_y2   REAL,
    score_left_game     INTEGER,
    bbox_right_point_x1 REAL,
    bbox_right_point_y1 REAL,
    bbox_right_point_x2 REAL,
    bbox_right_point_y2 REAL,
    score_right_point   INTEGER,
    bbox_right_game_x1  REAL,
    bbox_right_game_y1  REAL,
    bbox_right_game_x2  REAL,
    bbox_right_game_y2  REAL,
    score_right_game    INTEGER,
    score_raw    TEXT,
    PRIMARY KEY (video_name, frame_idx)
)
"""

# Columns added after the initial schema — used to migrate existing DBs.
_EXTRA_COLUMNS = [
    ("bbox_left_point_x1",  "REAL"),
    ("bbox_left_point_y1",  "REAL"),
    ("bbox_left_point_x2",  "REAL"),
    ("bbox_left_point_y2",  "REAL"),
    ("score_left_point",    "INTEGER"),
    ("bbox_left_game_x1",   "REAL"),
    ("bbox_left_game_y1",   "REAL"),
    ("bbox_left_game_x2",   "REAL"),
    ("bbox_left_game_y2",   "REAL"),
    ("score_left_game",     "INTEGER"),
    ("bbox_right_point_x1", "REAL"),
    ("bbox_right_point_y1", "REAL"),
    ("bbox_right_point_x2", "REAL"),
    ("bbox_right_point_y2", "REAL"),
    ("score_right_point",   "INTEGER"),
    ("bbox_right_game_x1",  "REAL"),
    ("bbox_right_game_y1",  "REAL"),
    ("bbox_right_game_x2",  "REAL"),
    ("bbox_right_game_y2",  "REAL"),
    ("score_right_game",    "INTEGER"),
]


def _migrate_db(con: sqlite3.Connection) -> None:
    """Add any missing score-bbox columns to an existing DB."""
    existing = {row[1] for row in con.execute("PRAGMA table_info(scoreboards)")}
    for col_name, col_type in _EXTRA_COLUMNS:
        if col_name not in existing:
            con.execute(f"ALTER TABLE scoreboards ADD COLUMN {col_name} {col_type}")
    con.commit()


# ── VLM prompt ────────────────────────────────────────────────────────────────

_PROMPT = (
    "This is a frame from a table tennis match broadcast.\n"
    "Find the on-screen scoreboard or score overlay.\n\n"
    "The scoreboard shows two sides separated by a divider:\n"
    "  LEFT SIDE  — the half of the scoreboard on the LEFT of your screen.\n"
    "  RIGHT SIDE — the half of the scoreboard on the RIGHT of your screen.\n\n"
    "Each side shows a LARGE, prominent point score (ignore any small game/set numbers).\n\n"
    "IMPORTANT: respond with a SINGLE flat JSON object — NOT an array, NOT nested objects.\n"
    "The object must have exactly these three keys:\n"
    '  "bbox"        – [x1,y1,x2,y2] where EVERY value is a FRACTION between 0.0 and 1.0\n'
    '                  (x divided by image width, y divided by image height).\n'
    '                  Do NOT use pixel values. Do NOT use percentages.\n'
    '                  Bounding the entire scoreboard region, or null if none visible.\n'
    '  "left_point"  – integer: the LARGE point score on the LEFT half of the scoreboard, or null.\n'
    '  "right_point" – integer: the LARGE point score on the RIGHT half of the scoreboard, or null.\n\n'
    "Both sides go into the SAME object. Do NOT use an array.\n\n"
    "Correct example — bbox values are fractions 0.0–1.0, NOT pixels, NOT percentages:\n"
    '{"bbox":[0.55,0.02,0.98,0.12],"left_point":9,"right_point":2}\n\n'
    "Wrong (pixel coordinates — do NOT do this):\n"
    '{"bbox":[1580,28,1900,85],"left_point":9,"right_point":2}\n\n'
    "Wrong (percentages — do NOT do this):\n"
    '{"bbox":[55,2,98,12],"left_point":9,"right_point":2}\n\n'
    "Return only the JSON — no prose, no markdown fences."
)


# ── public entry point ────────────────────────────────────────────────────────

def run_preprocess_from_dataset(
    dataset,
    db_path: Path,
    *,
    model_id: str = "Qwen/Qwen3.5-4B",
    device: str = "auto",
    override: bool = False,
    debug: bool = False,
) -> None:
    """Detect scoreboards by iterating over an OpenTT dataset instance.

    The dataset must already be configured with the desired frame stride and
    scene_length = batch_size.  Each Scene provides exactly one batch of
    frames via scene.rgbs (T, H, W, 3).

    Args:
        dataset:   OpenTT instance (scene_length = desired batch size,
                   frame_stride = preprocess stride, non-overlapping clips).
        db_path:   SQLite output file (created / appended to).
        model_id:  Hugging Face model ID for the VLM.
        device:    "auto", "cuda:0", "cpu", etc.
        override:  If True, re-process already-stored frames (default: skip).
        debug:     If True, show each center-cropped image and print the prompt
                   before sending to the VLM.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute(_SCHEMA)
    con.commit()
    _migrate_db(con)

    # Already-processed (video_name, frame_idx) pairs — for resumability.
    # When override=True we ignore the existing rows and re-process everything.
    if override:
        done: set[tuple[str, int]] = set()
    else:
        done = {
            (row[0], int(row[1]))
            for row in con.execute("SELECT video_name, frame_idx FROM scoreboards")
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

    model, processor = _load_model(model_id, device=device)

    if debug:
        print(
            f"\n{'='*60}\n"
            f"DEBUG — center crop: {_CROP_HEIGHT_FRAC*100:.0f}% height × "
            f"{_CROP_WIDTH_FRAC*100:.0f}% width\n"
            f"PROMPT:\n{_PROMPT}\n"
            f"{'='*60}\n"
        )

    try:
        for idx in range(n_clips):
            clip    = dataset._clips[idx]
            name    = clip["name"]
            indices = clip["indices"]  # list of absolute frame indices

            pending = [i for i, fi in enumerate(indices) if (name, fi) not in done]
            if not pending:
                continue

            scene = dataset[idx]
            if scene.rgbs is None:
                continue

            imgs: list[Image.Image] = []
            fis:  list[int]         = []
            for i in pending:
                frame  = scene.rgbs[i]  # (H, W, 3)
                np_img = (frame.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
                imgs.append(Image.fromarray(np_img))
                fis.append(indices[i])

            sys.stdout.write(
                f"  [{idx + 1:>{len(str(n_clips))}}/{n_clips}]"
                f"  {name}  frames {fis[0]:,}…{fis[-1]:,}"
                f"  ({len(fis)} frame{'s' if len(fis) != 1 else ''})\n"
            )
            sys.stdout.flush()

            _infer_and_save(name, fis, imgs, con, model, processor, debug=debug)
            done.update((name, fi) for fi in fis)

    finally:
        con.close()

    print(f"\nDone. Results → {db_path}")


# ── model loading ─────────────────────────────────────────────────────────────

def _load_model(model_id: str, *, device: str = "auto"):
    import os
    from transformers import AutoProcessor

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(f"Loading {model_id} …")
    processor = AutoProcessor.from_pretrained(model_id)

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if (use_cuda and device != "cpu") else torch.float32

    if "qwen3-vl" in model_id.lower():
        from transformers import Qwen3VLForConditionalGeneration
        model_cls = Qwen3VLForConditionalGeneration
    else:
        from transformers import AutoModelForImageTextToText
        model_cls = AutoModelForImageTextToText

    def _try_load(device_map):
        return model_cls.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device_map,
        )

    try:
        model = _try_load(device)
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower() or not use_cuda:
            raise
        print(f"  WARNING: GPU OOM — falling back to CPU (hint: use --device cuda:0 "
              f"to pin to your 24 GB GPU)\n  Error was: {exc}")
        torch.cuda.empty_cache()
        model = _try_load("cpu")

    model.eval()
    actual_device = next(model.parameters()).device
    print(f"  device: {actual_device}  dtype: {next(model.parameters()).dtype}\n")
    return model, processor


# ── VLM inference ─────────────────────────────────────────────────────────────

def _infer_and_save(
    name: str,
    indices: list[int],
    imgs: list[Image.Image],
    con: sqlite3.Connection,
    model,
    processor,
    *,
    debug: bool = False,
) -> None:
    rows = _query_batch(imgs, model, processor, debug=debug)

    for fi, row in zip(indices, rows):
        con.execute(
            "INSERT OR REPLACE INTO scoreboards "
            "(video_name, frame_idx, "
            " bbox_x1, bbox_y1, bbox_x2, bbox_y2, "
            " score_left, score_right, "
            " bbox_left_point_x1, bbox_left_point_y1,"
            " bbox_left_point_x2, bbox_left_point_y2, score_left_point, "
            " bbox_left_game_x1,  bbox_left_game_y1,"
            " bbox_left_game_x2,  bbox_left_game_y2,  score_left_game, "
            " bbox_right_point_x1, bbox_right_point_y1,"
            " bbox_right_point_x2, bbox_right_point_y2, score_right_point, "
            " bbox_right_game_x1,  bbox_right_game_y1,"
            " bbox_right_game_x2,  bbox_right_game_y2,  score_right_game, "
            " score_raw) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, "
            "        ?, ?, ?, ?, ?, "
            "        ?, ?, ?, ?, ?, "
            "        ?, ?, ?, ?, ?, "
            "        ?, ?, ?, ?, ?, "
            "        ?)",
            (
                name, fi,
                row.get("bbox_x1"), row.get("bbox_y1"),
                row.get("bbox_x2"), row.get("bbox_y2"),
                row.get("score_left"), row.get("score_right"),
                row.get("bbox_left_point_x1"), row.get("bbox_left_point_y1"),
                row.get("bbox_left_point_x2"), row.get("bbox_left_point_y2"),
                row.get("score_left_point"),
                row.get("bbox_left_game_x1"),  row.get("bbox_left_game_y1"),
                row.get("bbox_left_game_x2"),  row.get("bbox_left_game_y2"),
                row.get("score_left_game"),
                row.get("bbox_right_point_x1"), row.get("bbox_right_point_y1"),
                row.get("bbox_right_point_x2"), row.get("bbox_right_point_y2"),
                row.get("score_right_point"),
                row.get("bbox_right_game_x1"),  row.get("bbox_right_game_y1"),
                row.get("bbox_right_game_x2"),  row.get("bbox_right_game_y2"),
                row.get("score_right_game"),
                row.get("raw"),
            ),
        )
    con.commit()

    for fi, row in zip(indices, rows):
        has_bbox = "bbox_x1" in row
        bbox_str = (
            f"[{row['bbox_x1']:.0f},{row['bbox_y1']:.0f},"
            f"{row['bbox_x2']:.0f},{row['bbox_y2']:.0f}]"
            if has_bbox else "—"
        )
        lp = row.get("score_left_point")
        rp = row.get("score_right_point")
        score_str = f"L:{lp}  R:{rp}" if (lp is not None or rp is not None) else "—"
        sys.stdout.write(
            f"    frame {fi:7,d}  bbox {bbox_str:<32s}  score {score_str}\n"
        )
    sys.stdout.flush()


def _query_batch(imgs: list[Image.Image], model, processor, *, debug: bool = False) -> list[dict]:
    """Center-crop each image, optionally show it for debugging, then run the VLM."""
    results = []
    for i, img in enumerate(imgs):
        orig_w, orig_h = img.size
        cropped = _center_crop(img)
        crop_w, crop_h = cropped.size
        crop_left = (orig_w - crop_w) // 2
        crop_top  = (orig_h - crop_h) // 2

        if debug:
            import numpy as np
            from od3d_basic.cv.visual.show import show_img
            sys.stdout.write(
                f"  [debug] frame {i}  original {orig_w}×{orig_h}"
                f"  → crop {crop_w}×{crop_h}"
                f"  (center {_CROP_WIDTH_FRAC*100:.0f}%W × {_CROP_HEIGHT_FRAC*100:.0f}%H)"
                f"  offset ({crop_left}, {crop_top})\n"
            )
            sys.stdout.flush()
            rgb_tensor = torch.from_numpy(np.array(cropped)).float().permute(2, 0, 1) / 255.0
            show_img(rgb_tensor)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": cropped},
                    {"type": "text", "text": _PROMPT},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        if debug:
            sys.stdout.write(f"  [debug] raw output:\n{output}\n")
            sys.stdout.flush()

        row = _parse_response(output)
        row = _crop_frac_to_full_pixels(row, crop_left, crop_top, crop_w, crop_h)
        results.append(row)
    return results


def _crop_frac_to_full_pixels(row: dict, crop_left: int, crop_top: int,
                               crop_w: int, crop_h: int) -> dict:
    """Convert bbox from crop-relative fractions (0.0–1.0) to full-frame pixel coords."""
    if "bbox_x1" not in row:
        return row
    row["bbox_x1"] = round(crop_left + row["bbox_x1"] * crop_w)
    row["bbox_y1"] = round(crop_top  + row["bbox_y1"] * crop_h)
    row["bbox_x2"] = round(crop_left + row["bbox_x2"] * crop_w)
    row["bbox_y2"] = round(crop_top  + row["bbox_y2"] * crop_h)
    return row


# ── response parsing ──────────────────────────────────────────────────────────

def _parse_response(text: str) -> dict:
    text = text.strip()

    parsed = _try_json(text)
    if parsed is not None:
        if isinstance(parsed, list):
            parsed = _merge_array(parsed)
        return _normalise(parsed, text)

    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        parsed = _try_json(m.group())
        if parsed is not None:
            return _normalise(parsed, text)

    return {"raw": text}


def _merge_array(items: list) -> dict:
    """Merge a list of partial score dicts (model split left/right) into one."""
    merged: dict = {}
    for item in items:
        if isinstance(item, dict):
            merged.update(item)
    return merged


def _try_json(s: str) -> Optional[dict]:
    try:
        result = json.loads(s)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        return None


def _normalise(d: dict, raw: str) -> dict:
    result: dict = {"raw": raw}

    bbox = d.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
            result.update(bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2)
        except (TypeError, ValueError):
            pass

    # Parse the two point scores (plain integers).
    for region in ("left_point", "right_point"):
        val = d.get(region)
        if val is not None:
            try:
                result[f"score_{region}"] = int(val)
            except (TypeError, ValueError):
                pass

    # Populate backward-compat score_left / score_right from point scores.
    if "score_left_point" in result:
        result.setdefault("score_left", result["score_left_point"])
    if "score_right_point" in result:
        result.setdefault("score_right", result["score_right_point"])

    return result
