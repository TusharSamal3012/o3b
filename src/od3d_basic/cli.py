"""
o3x — od3d_basic command-line interface.

Usage:
  o3x dataset fetch     --config <yaml> [--url URL]
  o3x dataset index     --config <yaml> [--db FILE]
  o3x dataset visualize --config <yaml> [--db FILE] [--limit N] [--object-id ID]
                                         [--filter-has-kpts] [--render]
                                         [--render-frames N] [--renderer BACKEND]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ── dataset sub-parser ────────────────────────────────────────────────────────

def _build_dataset_parser(sub):
    p = sub.add_parser("dataset", help="Dataset commands (fetch, index, visualize)")
    ds_sub = p.add_subparsers(dest="dataset_command", required=True)

    def _add_config(q):
        q.add_argument(
            "--config", required=True, type=Path, metavar="YAML",
            help="Path to a DatasetConfig YAML (must contain class_name)",
        )

    p_fetch = ds_sub.add_parser("fetch", help="Download / prepare the dataset")
    _add_config(p_fetch)
    p_fetch.add_argument("--url", default=None, metavar="URL")

    p_index = ds_sub.add_parser("index", help="Build SQLite index from on-disk data")
    _add_config(p_index)
    p_index.add_argument("--db", type=Path, default=None, metavar="FILE")

    p_vis = ds_sub.add_parser("visualize", help="Summarize and optionally render dataset objects")
    _add_config(p_vis)
    p_vis.add_argument("--db", type=Path, default=None, metavar="FILE")
    p_vis.add_argument("--limit", type=int, default=20, metavar="N")
    p_vis.add_argument("--object-id", default=None, metavar="ID")
    p_vis.add_argument("--filter-has-kpts", action="store_true")
    p_vis.add_argument("--render", action="store_true")
    p_vis.add_argument("--render-frames", type=int, default=4, metavar="N")
    p_vis.add_argument("--renderer", choices=["pyrender", "nvdiffrast"], default="pyrender")


def _run_dataset(args):
    from od3d_basic.dataset.cli import _load_class_from_config

    cls, cfg = _load_class_from_config(args.config)

    if args.dataset_command == "fetch":
        cls.fetch(cfg, url=args.url)
    elif args.dataset_command == "index":
        cls.index(cfg, db=args.db)
    elif args.dataset_command == "visualize":
        if args.filter_has_kpts:
            cfg.filter_has_kpts = True
        cls.visualize(
            cfg,
            db=args.db,
            limit=args.limit,
            object_id=args.object_id,
            render=args.render,
            render_frames=args.render_frames,
            renderer=args.renderer,
        )


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="o3x",
        description="od3d_basic CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _build_dataset_parser(sub)

    args = parser.parse_args(argv)

    if args.command == "dataset":
        _run_dataset(args)


if __name__ == "__main__":
    main()
