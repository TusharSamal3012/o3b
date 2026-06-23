"""
o3b — o3b command-line interface.

Usage:
  o3b dataset fetch  -d housecorr3d_object_pair [--url URL] [--platform PLATFORM]
  o3b dataset index  -d housecorr3d_object_pair [--db FILE] [--platform PLATFORM]
  o3b dataset viz    -d housecorr3d_object_pair [--db FILE] [--limit N] [--object-id ID]
                                         [--filter-has-kpts] [--render]
                                         [--render-frames N] [--renderer BACKEND]
                                         [--debug] [--platform PLATFORM]
  o3b bench run      -b <benchmark> [-p <platform>] [-a <ablation>]
  o3b platform setup    -p <platform>
  o3b platform stop  -p <platform> [-y]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


# ── dataset sub-parser ────────────────────────────────────────────────────────

def _build_dataset_parser(sub):
    p = sub.add_parser("dataset", help="Dataset commands (fetch, index, viz)")
    ds_sub = p.add_subparsers(dest="dataset_command", required=True)

    def _add_config(q):
        from o3b.dataset.cli import _resolve_dataset_config
        q.add_argument(
            "-d", "--config", required=True, type=_resolve_dataset_config, metavar="DATASET",
            help="Dataset config name (e.g. housecorr3d_object_pair, resolved from "
                 "configs/dataset/) or full path to a YAML file",
        )
        q.add_argument(
            "-p", "--platform", default="default", metavar="PLATFORM",
            help="Platform name whose path_datasets_raw / path_datasets_preprocess "
                 "override the dataset config paths (default: default)",
        )

    p_fetch = ds_sub.add_parser("fetch", help="Download / prepare the dataset")
    _add_config(p_fetch)
    p_fetch.add_argument("--url", default=None, metavar="URL")

    p_index = ds_sub.add_parser("index", help="Build SQLite index from on-disk data")
    _add_config(p_index)
    p_index.add_argument("--db", type=Path, default=None, metavar="FILE")
    p_index.add_argument("--remove", action="store_true",
                         help="Delete any existing index for this dataset before indexing")
    p_index.add_argument(
        "--max", type=int, default=None, metavar="N", dest="max_index",
        help="Stop after indexing N total rows (for quick testing). "
             "filter_count_max in the config applies at query time only.",
    )

    p_vis = ds_sub.add_parser("viz", help="Summarize and optionally render dataset objects")
    _add_config(p_vis)
    p_vis.add_argument("--db", type=Path, default=None, metavar="FILE")
    p_vis.add_argument("--limit", type=int, default=20, metavar="N")
    p_vis.add_argument("--object-id", default=None, metavar="ID")
    p_vis.add_argument(
        "--frame-stride", type=int, default=None, metavar="N",
        help="Initial ←/→ jump size in frames (default: frame_stride from dataset config); "
             "can also be changed via the Stride trackbar",
    )
    p_vis.add_argument(
        "--frames-per-scene", type=int, default=None, metavar="N",
        help="Show a static grid of N evenly-sampled frames per clip instead of the interactive player",
    )
    p_vis.add_argument("--filter-has-kpts", action="store_true")
    p_vis.add_argument("--render", action="store_true")
    p_vis.add_argument("--render-frames", type=int, default=4, metavar="N")
    p_vis.add_argument("--renderer", choices=["pyrender", "nvdiffrast"], default="pyrender")
    p_vis.add_argument("--debug", action="store_true",
                       help="Show front/top/right camera frustums in the viser scene")
    p_vis.add_argument("--object-centric", action="store_true",
                       help="Object-centric view: place object at world origin, camera in object space")

    p_tform = ds_sub.add_parser(
        "tform",
        help="Interactive axis-convention viewer — determine obj_tform4x4 for the dataset",
    )
    _add_config(p_tform)
    p_tform.add_argument("--limit", type=int, default=20, metavar="N",
                         help="Max objects to browse (default: 20)")

    p_pre = ds_sub.add_parser(
        "preprocess",
        help="OpenTT: annotate score bboxes interactively, then extract scores via VLM",
    )
    _add_config(p_pre)
    p_pre.add_argument(
        "--db", type=Path, default=None, metavar="FILE",
        help="SQLite output file (default: <path_preprocess>/scoreboards.db)",
    )
    p_pre.add_argument(
        "--annotate", action="store_true",
        help="Draw the scoreboard / left-score / right-score bboxes interactively "
             "for each video (saved to video_bboxes.json). Run this once before VLM.",
    )
    p_pre.add_argument(
        "--model", default="Qwen/Qwen3-VL-2B-Instruct", metavar="MODEL_ID",
        help="HuggingFace model ID for VLM score reading "
             "(default: Qwen/Qwen3-VL-2B-Instruct)",
    )
    p_pre.add_argument(
        "--device", default="cpu", metavar="DEVICE",
        help="Torch device for VLM inference, e.g. cuda:0 (default: cpu)",
    )
    p_pre.add_argument(
        "--video", default=None, metavar="NAME",
        help="Restrict to a single video by name, e.g. game_1 or test_3",
    )
    p_pre.add_argument(
        "--override", action="store_true",
        help="Re-annotate / re-process already-handled videos or frames.",
    )
    p_pre.add_argument(
        "--debug", action="store_true",
        help="Show score crops and raw VLM output during processing.",
    )
    p_pre.add_argument(
        "--remove", action="store_true",
        help="Delete all rows from the scoreboards table and exit.",
    )


def _run_dataset(args):
    from o3b.dataset.cli import _load_class_from_config, _platform_to_dataset_overrides

    overrides = _platform_to_dataset_overrides(args.platform)
    cls, cfg = _load_class_from_config(args.config, overrides=overrides)

    if args.dataset_command == "fetch":
        cls.fetch(cfg, url=args.url)
    elif args.dataset_command == "index":
        cls.index(cfg, db=args.db, remove=args.remove, max_index=getattr(args, "max_index", None))
    elif args.dataset_command == "viz":
        if args.filter_has_kpts:
            cfg.filter_has_kpts = True
        cls.visualize(
            cfg,
            db=args.db,
            limit=args.limit,
            object_id=args.object_id,
            frame_stride=args.frame_stride,
            frames_per_scene=args.frames_per_scene,
            render=args.render,
            render_frames=args.render_frames,
            renderer=args.renderer,
            debug=args.debug,
            obj_centric=args.object_centric,
        )
    elif args.dataset_command == "tform":
        from o3b.dataset.tform import run_tform_viewer
        run_tform_viewer(cls, cfg, limit=args.limit)
    elif args.dataset_command == "preprocess":
        if not hasattr(cls, "preprocess"):
            print(
                f"ERROR: {cls.__name__} does not implement preprocess().\n"
                "This command is currently only available for OpenTT.",
                file=sys.stderr,
            )
            sys.exit(1)
        cls.preprocess(
            cfg,
            db=args.db,
            model_id=args.model,
            device=args.device,
            video=args.video,
            annotate=args.annotate,
            override=args.override,
            debug=args.debug,
            remove=args.remove,
        )


# ── platform sub-parser ───────────────────────────────────────────────────────

def _build_platform_parser(sub):
    p = sub.add_parser("platform", help="Platform management commands")
    plat_sub = p.add_subparsers(dest="platform_command", required=True)

    p_setup = plat_sub.add_parser(
        "setup",
        help="Copy and run the repository setup script on a remote platform",
    )
    p_setup.add_argument(
        "-p", "--platform", default="slurm", metavar="PLATFORM",
        help="Platform name matching a config in configs/platform/ (default: slurm)",
    )

    p_overview = plat_sub.add_parser(
        "overview",
        help="Show job queue status on a remote platform",
    )
    p_overview.add_argument(
        "-p", "--platform", default="slurm", metavar="PLATFORM",
        help="Platform name matching a config in configs/platform/ (default: slurm)",
    )
    p_overview.add_argument(
        "--configs", action="store_true",
        help="Also print the resolved platform config",
    )
    p_overview.add_argument(
        "--hours", type=float, default=2.0, metavar="N",
        help="Show sacct history for the last N hours (default: 2)",
    )

    p_runi = plat_sub.add_parser(
        "runi",
        help="Open an interactive shell on a compute node via srun --pty bash",
    )
    p_runi.add_argument(
        "-p", "--platform", default="slurm", metavar="PLATFORM",
        help="Platform name matching a config in configs/platform/ (default: slurm)",
    )

    p_setupi = plat_sub.add_parser(
        "setupi",
        help="Open an interactive shell, copy setup script, and print it for manual execution",
    )
    p_setupi.add_argument(
        "-p", "--platform", default="slurm", metavar="PLATFORM",
        help="Platform name matching a config in configs/platform/ (default: slurm)",
    )

    p_run = plat_sub.add_parser(
        "run",
        help="Run a command on a compute node (non-interactive srun)",
    )
    p_run.add_argument(
        "-p", "--platform", default="slurm", metavar="PLATFORM",
        help="Platform name matching a config in configs/platform/ (default: slurm)",
    )
    p_run.add_argument(
        "-c", "--command", required=True, metavar="CMD",
        help="Shell command to execute on the compute node",
    )

    p_stop = plat_sub.add_parser(
        "stop",
        help="Cancel all running jobs on the platform's configured partition",
    )
    p_stop.add_argument(
        "-p", "--platform", default="slurm", metavar="PLATFORM",
        help="Platform name matching a config in configs/platform/ (default: slurm)",
    )
    p_stop.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip confirmation prompt",
    )
    p_stop.add_argument(
        "-j", "--jobs", default=None, metavar="A-B",
        help="Cancel only jobs in the inclusive ID range A-B (e.g. 29175478-29175502)",
    )

    p_queue = plat_sub.add_parser(
        "queue",
        help="Show partition queue with per-job resource usage (GPUs, CPUs, memory)",
    )
    p_queue.add_argument(
        "-p", "--platform", default="slurm", metavar="PLATFORM",
        help="Platform name matching a config in configs/platform/ (default: slurm)",
    )
    p_queue.add_argument(
        "--partition", default=None, metavar="PARTITION",
        help="Override the partition from the platform config",
    )
    p_queue.add_argument(
        "--all-users", action="store_true",
        help="Show jobs from all users (default: only the configured username)",
    )



def _load_platform_config(platform: str):
    """Load a platform config, resolving its defaults chain via OmegaConf merge."""
    from omegaconf import OmegaConf
    from o3b.io import _load_yaml_with_defaults

    configs_dir = (Path(__file__).parent.parent / "configs" / "platform").resolve()
    if not configs_dir.is_dir():
        raise FileNotFoundError(f"Platform config directory not found: {configs_dir}")

    cfg_path = configs_dir / f"{platform}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Platform config not found: {cfg_path}")

    cfg = OmegaConf.create(_load_yaml_with_defaults(cfg_path))
    return cfg, configs_dir



def _multiply_metric_with_unit(metric_with_unit: str, factor: int) -> str:
    """Multiply a value-with-unit string (e.g. '5gb') by an integer factor."""
    import re
    m = re.match(r"^(\d+(?:\.\d+)?)(\D+)$", str(metric_with_unit).strip())
    if m:
        return f"{int(float(m.group(1)) * factor)}{m.group(2)}"
    return str(metric_with_unit)


def _save_script_locally(name: str, content: str, ts: str | None = None) -> Path:
    """Save *content* to ~/.o3b/scripts/<timestamp>_<name>.sh and return the path."""
    import re
    from datetime import datetime

    scripts_dir = Path.home() / ".o3b" / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    if ts is None:
        ts = datetime.now().strftime("%m%d_%H%M%S")
    safe = re.sub(r"[^A-Za-z0-9_\-]", "_", name)
    path = scripts_dir / f"{ts}_{safe}.sh"
    path.write_text(content)
    return path


def _make_sbatch_script(cfg, job_name: str, env_vars: dict, remote_setup_script: str) -> str:
    """Return a complete sbatch script string built from platform config values."""

    node_count      = cfg.get("node_count", 1)
    gpu_count       = cfg.get("gpu_count_per_node", 1)
    cpu_count       = cfg.get("cpu_count_per_gpu", 8)
    ram_per_cpu     = cfg.get("ram_per_cpu", "5gb")
    walltime        = cfg.get("walltime", "24:00:00")
    partition       = cfg.get("partition", None)
    nodes_exclude   = cfg.get("nodes_exclude", None)
    restart         = cfg.get("restart_upon_fail", False)
    # path_home is defined in the custom overlay; fall back to path_ws
    path_home       = cfg.get("path_home", cfg.get("path_ws", "/tmp"))

    total_mem = _multiply_metric_with_unit(ram_per_cpu, cpu_count)

    optional = {
        "requeue":        "#SBATCH --requeue"                    if restart       else "",
        "partition":      f"#SBATCH --partition {partition}"     if partition     else "",
        "nodes_exclude":  f"#SBATCH --exclude {nodes_exclude}"   if nodes_exclude else "",
    }

    env_block = "\n".join(f"export {k}={v!r}" for k, v in env_vars.items())

    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH -J {job_name}",
        f"#SBATCH --nodes {node_count}",
        "#SBATCH --ntasks-per-node 1",
        f"#SBATCH --time {walltime}",
        f"#SBATCH --gres gpu:{gpu_count}",
        f"#SBATCH --cpus-per-task {cpu_count}",
        f"#SBATCH --mem {total_mem}",
        "#SBATCH --open-mode=append",
        f"#SBATCH -o {path_home}/slurm_jobs/%x_%j.o",
        "#SBATCH --mail-type=FAIL",
        "#SBATCH --signal=B:SIGUSR1@60",
    ]
    for v in optional.values():
        if v:
            lines.append(v)

    lines += [
        "",
        "set -euo pipefail",
        "",
        env_block,
        "",
        f"bash {remote_setup_script}",
    ]
    return "\n".join(lines) + "\n"


def _run_platform_setup(args):
    import re
    import subprocess
    from omegaconf import OmegaConf, open_dict

    platform = args.platform

    print(f"Loading platform config '{platform}'…")
    cfg, configs_dir = _load_platform_config(platform)

    with open_dict(cfg):
        cfg.setup = True

    print(OmegaConf.to_yaml(cfg))

    ssh_host = cfg.get("ssh")
    if not ssh_host or ssh_host is False:
        raise ValueError(
            f"Platform '{platform}' has no ssh host configured (ssh: {ssh_host!r})"
        )

    path_ws        = cfg.get("path_ws", "")
    path_cuda      = cfg.get("path_cuda", "/usr/local/cuda-12.4")
    python_version = str(cfg.get("python_version", "3.10"))
    torch_version  = str(cfg.get("torch_version", "2.6.0"))
    deps                 = list(cfg.get("deps", []) or [])
    deps_tag             = "_".join(sorted(deps)) if deps else ""
    install_diff3f       = cfg.get("install_diff3f", False) or "diff3f" in deps
    install_densematcher = cfg.get("install_densematcher", False) or "densematcher" in deps
    branch         = cfg.get("branch", "main")
    pull           = cfg.get("pull", True)
    pull_submodules  = cfg.get("pull_submodules", True)
    skip_submodules  = " ".join(str(s) for s in list(cfg.get("skip_submodules", []) or []))
    username       = cfg.get("username", "")
    path_home      = cfg.get("path_home", path_ws)

    # Walk up from __file__ to find the outermost git repo (the repo that
    # contains o3b as a submodule) via --show-superproject-working-tree.
    try:
        submodule_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
            cwd=Path(__file__).parent,
        ).strip()
        superproject = subprocess.check_output(
            ["git", "rev-parse", "--show-superproject-working-tree"],
            text=True,
            cwd=submodule_root,
        ).strip()
        local_repo_root = superproject if superproject else submodule_root
        repo_name = Path(local_repo_root).name
    except subprocess.CalledProcessError:
        repo_name = "housecorr3d"
        local_repo_root = str(Path.cwd())

    # Inject the GitHub token into the HTTPS remote URL so git clone on the
    # cluster can authenticate through the proxy without SSH keys.
    token = OmegaConf.select(cfg, "credentials.github.token", default="") or ""
    try:
        raw_remote = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            text=True, cwd=local_repo_root,
        ).strip()
        # Convert SSH → HTTPS if needed: git@github.com:Org/Repo → https://github.com/Org/Repo
        if raw_remote.startswith("git@"):
            raw_remote = re.sub(r"git@github\.com:", "https://github.com/", raw_remote)
        # Strip any existing token then prepend the new one
        plain = re.sub(r"https://[^@]+@", "https://", raw_remote)
        repo_url  = plain.replace("https://", f"https://{token}@") if token else plain
        repo_name = Path(re.sub(r"\.git$", "", plain.split("/")[-1])).name
    except subprocess.CalledProcessError:
        repo_url  = ""
        repo_name = Path(local_repo_root).name

    # Find the setup script in the parent repo
    setup_script_local = Path(local_repo_root) / "setup" / "setup_slurm.sh"
    if not setup_script_local.is_file():
        raise FileNotFoundError(f"Setup script not found: {setup_script_local}")

    remote_setup  = f"{path_ws}/setup_slurm.sh"
    remote_sbatch = f"{path_ws}/setup_slurm_job.sh"

    def _scp(local, remote):
        target = f"{ssh_host}:{remote}"
        if username:
            target = f"{username}@{ssh_host}:{remote}"
        print(f"Copying {local} → {target}")
        subprocess.run(["scp", str(local), target], check=True)

    # Build sbatch wrapper with #SBATCH headers from the platform config
    _proxy = "http://tfproxy.informatik.intra.uni-freiburg.de:8080"
    env_vars = {
        "PATH_WS":         path_ws,
        "PATH_CUDA":       path_cuda,
        "PYTHON_VERSION":  python_version,
        "TORCH_VERSION":   torch_version,
        "INSTALL_DIFF3F":        "true" if install_diff3f else "false",
        "INSTALL_DENSEMATCHER":  "true" if install_densematcher else "false",
        "DEPS_TAG":              deps_tag,
        "REPO_URL":        repo_url,   # housecorr3d HTTPS URL with token
        "REPO_NAME":       repo_name,  # derived from remote URL, e.g. HouseCorr3Dv2
        "GITHUB_TOKEN":    token,
        "BRANCH":          branch,
        "PULL":            "true" if pull else "false",
        "PULL_SUBMODULES": "true" if pull_submodules else "false",
        "SKIP_SUBMODULES": skip_submodules,
        "HTTP_PROXY":      _proxy,
        "HTTPS_PROXY":     _proxy,
        "http_proxy":      _proxy,
        "https_proxy":     _proxy,
    }
    sbatch_script = _make_sbatch_script(
        cfg,
        job_name=f"setup_{repo_name}",
        env_vars=env_vars,
        remote_setup_script=remote_setup,
    )

    # Save both scripts locally with timestamp before sending to remote
    local_setup_copy = _save_script_locally(f"setup_{repo_name}_script", setup_script_local.read_text())
    local_sbatch     = _save_script_locally(f"setup_{repo_name}_sbatch", sbatch_script)
    print(f"  saved locally: {local_setup_copy}")
    print(f"  saved locally: {local_sbatch}")

    # Write sbatch script to a temp file and SCP both scripts
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as tmp:
        tmp.write(sbatch_script)
        tmp_path = tmp.name

    try:
        _scp(setup_script_local, remote_setup)
        _scp(tmp_path, remote_sbatch)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Ensure output log directory exists, then submit via sbatch
    remote_cmd = (
        f"mkdir -p {path_home}/slurm_jobs && "
        f"chmod +x {remote_setup} {remote_sbatch} && "
        f"sbatch {remote_sbatch}"
    )
    print(f"Submitting setup job on {ssh_host}…")
    subprocess.run(["ssh", ssh_host, remote_cmd], check=True)


def _fetch_jobs(ssh_host: str, username: str, hours: float = 2.0) -> list:
    """Return job list from sacct for the last *hours* hours as a list of dicts."""
    import subprocess
    fields      = ["JobID", "JobName", "State", "ExitCode", "Elapsed", "Start", "End", "Partition", "NodeList"]
    start_expr  = f"$(date -d '{hours} hours ago' +'%Y-%m-%dT%H:%M:%S')"
    cmd = (
        f"sacct --starttime={start_expr}"
        f" --format={','.join(fields)!r}"
        f" --parsable2 --allocations"
    )
    if username:
        cmd += f" -u {username}"
    result = subprocess.run(["ssh", ssh_host, cmd], capture_output=True, text=True, check=True)
    lines  = [l for l in result.stdout.splitlines() if l.strip()]
    if len(lines) < 2:
        return []
    jobs = [dict(zip(fields, row.split("|"))) for row in lines[1:]]
    jobs.sort(key=lambda j: int(j.get("JobID", 0) or 0), reverse=True)
    return jobs


def _open_log(ssh_host: str, path_home: str, job: dict) -> None:
    """Open the job log in less via ssh -t. Prints an error if the file is missing."""
    import subprocess
    log_path = f"{path_home}/slurm_jobs/{job['JobName']}_{job['JobID']}.o"
    check = subprocess.run(
        ["ssh", ssh_host, f"test -f {log_path!r} && echo yes || echo no"],
        capture_output=True, text=True,
    )
    if check.stdout.strip() != "yes":
        print(f"\nLog not found: {log_path}")
        print("(Jobs not submitted via `o3b platform setup` may write logs elsewhere.)")
        input("Press Enter to return…")
        return
    subprocess.run(["ssh", "-t", ssh_host, f"less +G {log_path!r}"])


def _kill_job(ssh_host: str, job: dict) -> None:
    """Cancel a SLURM job via scancel after confirmation."""
    import subprocess
    job_id = job["JobID"]
    print(f"\nscancel {job_id} ({job['JobName']})  [y/N] ", end="", flush=True)
    if input().strip().lower() != "y":
        return
    result = subprocess.run(["ssh", ssh_host, f"scancel {job_id}"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"scancel failed: {result.stderr.strip()}")
    else:
        print(f"Job {job_id} cancelled.")
    input("Press Enter to continue…")


def _overview_tui(stdscr, jobs: list, ssh_host: str, title: str):
    """
    Curses TUI for job selection.
    Returns: ('view', job_dict) | ('refresh',) | ('quit',)
    """
    import curses

    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_BLACK,  curses.COLOR_CYAN)   # selected row
    curses.init_pair(2, curses.COLOR_BLACK,  curses.COLOR_WHITE)  # column header
    curses.init_pair(3, curses.COLOR_RED,    -1)                  # FAILED / ERROR
    curses.init_pair(4, curses.COLOR_GREEN,  -1)                  # COMPLETED
    curses.init_pair(5, curses.COLOR_YELLOW, -1)                  # RUNNING / other
    curses.init_pair(6, curses.COLOR_WHITE,  curses.COLOR_BLUE)   # title / status bar

    COLS = [
        ("JobID",     10, ">"),
        ("JobName",   60, "<"),
        ("State",     14, "<"),
        ("ExitCode",   8, ">"),
        ("Elapsed",   10, "<"),
        ("Start",     19, "<"),
        ("Partition", 12, "<"),
    ]

    def fmt_row(job):
        parts = []
        for field, w, align in COLS:
            val = str(job.get(field, ""))
            val = val[:w] if align == "<" else val
            parts.append(f"{val:{align}{w}}")
        return "  ".join(parts)

    def state_attr(state):
        if any(s in state for s in ("FAIL", "ERROR", "TIMEOUT", "OUT_OF")):
            return curses.color_pair(3)
        if "COMPLET" in state:
            return curses.color_pair(4)
        return curses.color_pair(5)

    current = 0
    offset  = 0

    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        # ── title bar ──────────────────────────────────────────────
        bar = f" {title}  [R] refresh  [K] kill  [Q] quit "
        stdscr.attron(curses.color_pair(6) | curses.A_BOLD)
        stdscr.addstr(0, 0, bar[:w - 1].ljust(w - 1))
        stdscr.attroff(curses.color_pair(6) | curses.A_BOLD)

        # ── column header ──────────────────────────────────────────
        hdr_job  = {f: f for f, *_ in COLS}
        stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
        stdscr.addstr(1, 0, fmt_row(hdr_job)[:w - 1].ljust(w - 1))
        stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)

        # ── job rows ───────────────────────────────────────────────
        list_h  = h - 3          # title + header + status bar
        visible = jobs[offset: offset + list_h]
        for i, job in enumerate(visible):
            y   = i + 2
            idx = i + offset
            row = fmt_row(job)[:w - 1]
            if idx == current:
                stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
                stdscr.addstr(y, 0, row.ljust(w - 1))
                stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)
            else:
                attr = state_attr(job.get("State", ""))
                stdscr.attron(attr)
                stdscr.addstr(y, 0, row)
                stdscr.attroff(attr)

        # ── status bar ─────────────────────────────────────────────
        count = f"[{current + 1}/{len(jobs)}]" if jobs else "[0/0]"
        status = f" {count}  ↑↓ navigate   Enter / L : show logs   K kill   R refresh   Q quit"
        stdscr.attron(curses.color_pair(6))
        stdscr.addstr(h - 1, 0, status[:w - 1].ljust(w - 1))
        stdscr.attroff(curses.color_pair(6))

        stdscr.refresh()
        key = stdscr.getch()

        if key in (ord("q"), ord("Q"), 27):
            return ("quit",)
        elif key in (ord("r"), ord("R")):
            return ("refresh",)
        elif key == curses.KEY_UP:
            if current > 0:
                current -= 1
                if current < offset:
                    offset -= 1
        elif key == curses.KEY_DOWN:
            if current < len(jobs) - 1:
                current += 1
                if current >= offset + list_h:
                    offset += 1
        elif key in (ord("\n"), ord("l"), ord("L"), curses.KEY_ENTER) and jobs:
            return ("view", jobs[current])
        elif key in (ord("k"), ord("K")) and jobs:
            return ("kill", jobs[current])


def _run_platform_overview(args):
    import curses
    import subprocess
    from omegaconf import OmegaConf

    platform = args.platform
    cfg, _   = _load_platform_config(platform)

    ssh_host  = cfg.get("ssh")
    if not ssh_host or ssh_host is False:
        raise ValueError(
            f"Platform '{platform}' has no ssh host configured (ssh: {ssh_host!r})"
        )

    username  = cfg.get("username", "")
    path_home = cfg.get("path_home", cfg.get("path_ws", ""))

    if args.configs:
        print("=" * 60)
        print(f"Platform config: {platform}")
        print("=" * 60)
        print(OmegaConf.to_yaml(cfg))
        input("Press Enter to open the job overview…")

    # ── active jobs: show as plain text before entering the TUI ─────
    squeue_fmt = "%.10i %.12P %.30j %.10u %.10T %.12M %.12l %.5D %R"
    squeue_cmd = f"squeue --format={squeue_fmt!r}"
    if username:
        squeue_cmd += f" -u {username}"
    print("=" * 60)
    print(f"Active jobs on {ssh_host}" + (f" (user: {username})" if username else ""))
    print("=" * 60)
    subprocess.run(["ssh", ssh_host, squeue_cmd], check=True)
    print()

    # ── TUI loop ────────────────────────────────────────────────────
    hours = getattr(args, "hours", 2.0)
    hours_label = f"{hours:g} h"
    tui_title = f"SLURM overview · {ssh_host}" + (f" · {username}" if username else "") + f" · last {hours_label}"
    jobs = _fetch_jobs(ssh_host, username, hours=hours)

    while True:
        action = curses.wrapper(lambda scr: _overview_tui(scr, jobs, ssh_host, tui_title))

        if action[0] == "quit":
            break
        elif action[0] == "refresh":
            jobs = _fetch_jobs(ssh_host, username)
        elif action[0] == "view":
            _open_log(ssh_host, path_home, action[1])
        elif action[0] == "kill":
            _kill_job(ssh_host, action[1])
            jobs = _fetch_jobs(ssh_host, username)


def _platform_srun_context(platform: str):
    """Return (ssh_host, srun_base, repo_path, venv_path, path_cuda, path_ws) for srun commands."""
    import os, re, subprocess
    from omegaconf import OmegaConf

    cfg, _ = _load_platform_config(platform)

    ssh_host = cfg.get("ssh")
    if not ssh_host or ssh_host is False:
        raise ValueError(
            f"Platform '{platform}' has no ssh host configured (ssh: {ssh_host!r})"
        )

    partition     = cfg.get("partition", None)
    node_count    = cfg.get("node_count", 1)
    gpu_count     = cfg.get("gpu_count_per_node", 1)
    cpu_count     = cfg.get("cpu_count_per_gpu", 8)
    ram_per_cpu   = cfg.get("ram_per_cpu", "5gb")
    walltime      = cfg.get("walltime", "24:00:00")
    nodes_exclude = cfg.get("nodes_exclude", None)
    total_mem     = _multiply_metric_with_unit(ram_per_cpu, cpu_count)

    path_ws        = cfg.get("path_ws", "")
    path_cuda      = cfg.get("path_cuda", "/usr/local/cuda-12.4")
    python_version = str(cfg.get("python_version", "3.10"))
    torch_version  = str(cfg.get("torch_version", "2.6.0"))
    deps                 = list(cfg.get("deps", []) or [])
    deps_tag             = "_".join(sorted(deps)) if deps else ""
    install_diff3f       = "true" if (cfg.get("install_diff3f", False) or "diff3f" in deps) else "false"
    install_densematcher = "true" if (cfg.get("install_densematcher", False) or "densematcher" in deps) else "false"
    setup          = "true" if cfg.get("setup", False) else "false"
    branch         = str(cfg.get("branch", "main"))
    pull           = str(cfg.get("pull", True)).lower()
    pull_subs      = str(cfg.get("pull_submodules", True)).lower()
    skip_subs      = " ".join(str(s) for s in list(cfg.get("skip_submodules", []) or []))

    cuda_tag  = "cu" + os.path.basename(path_cuda).replace("cuda-", "").replace(".", "")
    py_tag    = "py" + python_version.replace(".", "")
    torch_tag = "torch" + ".".join(torch_version.split(".")[:2]).replace(".", "")

    token = OmegaConf.select(cfg, "credentials.github.token", default="") or ""
    try:
        submodule_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True, cwd=Path(__file__).parent,
        ).strip()
        superproject = subprocess.check_output(
            ["git", "rev-parse", "--show-superproject-working-tree"], text=True, cwd=submodule_root,
        ).strip()
        local_repo_root = superproject if superproject else submodule_root
        raw_remote = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], text=True, cwd=local_repo_root,
        ).strip()
        if raw_remote.startswith("git@"):
            raw_remote = re.sub(r"git@github\.com:", "https://github.com/", raw_remote)
        plain    = re.sub(r"https://[^@]+@", "https://", raw_remote)
        repo_url  = plain.replace("https://", f"https://{token}@") if token else plain
        repo_name = Path(re.sub(r"\.git$", "", plain.split("/")[-1])).name
    except subprocess.CalledProcessError:
        repo_url  = ""
        repo_name = ""

    repo_path = f"{path_ws}/{repo_name}" if (path_ws and repo_name) else path_ws
    _venv_suffix = f"_{deps_tag}" if deps_tag else ""
    venv_path = f"{repo_path}/venv_{py_tag}_{cuda_tag}_{torch_tag}{_venv_suffix}" if repo_path else ""

    srun = (
        f"srun"
        f" --nodes {node_count}"
        f" --ntasks-per-node 1"
        f" --gres gpu:{gpu_count}"
        f" --cpus-per-task {cpu_count}"
        f" --mem {total_mem}"
        f" --time {walltime}"
    )
    if partition:
        srun += f" --partition {partition}"
    if nodes_exclude:
        srun += f" --exclude {nodes_exclude}"
    if path_ws:
        srun += f" --chdir {path_ws}"

    _proxy = "http://tfproxy.informatik.intra.uni-freiburg.de:8080"
    srun += (
        f" --export=ALL"
        f",HTTP_PROXY={_proxy}"
        f",HTTPS_PROXY={_proxy}"
        f",http_proxy={_proxy}"
        f",https_proxy={_proxy}"
        f",PATH_WS={path_ws}"
        f",PATH_CUDA={path_cuda}"
        f",PYTHON_VERSION={python_version}"
        f",TORCH_VERSION={torch_version}"
        f",INSTALL_DIFF3F={install_diff3f}"
        f",INSTALL_DENSEMATCHER={install_densematcher}"
        f",REPO_URL={repo_url}"
        f",REPO_NAME={repo_name}"
        f",SETUP={setup}"
        f",BRANCH={branch}"
        f",PULL={pull}"
        f",PULL_SUBMODULES={pull_subs}"
        f",SKIP_SUBMODULES={skip_subs}"
        f",GITHUB_TOKEN={token}"
        f",CUDA_HOME={path_cuda}"
        f",CUDACXX={path_cuda}/bin/nvcc"
        f",DEPS_TAG={deps_tag}"
    )

    return ssh_host, srun, repo_path, venv_path, path_cuda, path_ws


def _srun_env_lines(path_cuda: str, venv_path: str, repo_path: str, path_ws: str) -> list[str]:
    """Shell lines that run on the compute node before the actual command.

    Order: CUDA env → conditional setup script → conditional pull/checkout
           → venv activation → cd into repo.
    The SETUP / PULL / PULL_SUBMODULES / BRANCH values come from the srun
    --export env vars so the same script works regardless of platform config.
    """
    lines = [
        "[ -f ~/.bashrc ] && . ~/.bashrc",
        f"export CUDA_HOME={path_cuda}",
        f"export CUDACXX={path_cuda}/bin/nvcc",
        f"export PATH={path_cuda}/bin:$PATH",
        f"export LD_LIBRARY_PATH={path_cuda}/lib64:${{LD_LIBRARY_PATH:-}}",
        f"export CPATH=${{CPATH:-}}:{path_cuda}/targets/x86_64-linux/include",
        f"export LIBRARY_PATH=${{LIBRARY_PATH:-}}:{path_cuda}/targets/x86_64-linux/lib",
    ]
    # acquire the same directory lock used by setup_slurm.sh so concurrent
    # srun jobs don't race on git pull / submodule update / venv install
    if path_ws:
        lines += [
            f'exec 200>{path_ws}/setup_slurm.lock',
            f'flock -x 200',
        ]
    # run full setup script (e.g. install deps) when SETUP=true
    if path_ws:
        lines += [
            f'if [ "${{SETUP:-false}}" = "true" ]; then',
            f'    bash {path_ws}/setup_slurm.sh',
            f'fi',
        ]
    # checkout branch and pull when PULL=true
    if repo_path:
        lines += [
            f'if [ "${{PULL:-false}}" = "true" ]; then',
            f'    git -C {repo_path} fetch',
            f'    git -C {repo_path} checkout "${{BRANCH:-main}}"',
            f'    git -C {repo_path} pull',
            f'fi',
            f'if [ "${{PULL_SUBMODULES:-false}}" = "true" ]; then',
            f'    if [ -n "${{GITHUB_TOKEN:-}}" ]; then',
            f'        git config --global url."https://${{GITHUB_TOKEN}}@github.com/".insteadOf "https://github.com/"',
            f'    fi',
            f'    if [ -z "${{SKIP_SUBMODULES:-}}" ]; then',
            f'        git -C {repo_path} submodule update --init --recursive',
            f'    else',
            f'        git -C {repo_path} submodule init',
            f"        for sub in $(git -C {repo_path} submodule status | awk '{{print $2}}'); do",
            f'            _skip=false',
            f'            for s in ${{SKIP_SUBMODULES}}; do',
            f'                [ "$sub" = "$s" ] && _skip=true && break',
            f'            done',
            f'            if [ "$_skip" = "false" ]; then',
            f'                git -C {repo_path} submodule update --init --recursive -- "$sub"',
            f'            fi',
            f'        done',
            f'    fi',
            f'fi',
        ]
    if path_ws:
        lines.append('exec 200>&-')
    if venv_path:
        lines.append(f"[ -d {venv_path} ] && source {venv_path}/bin/activate")
    if repo_path:
        lines.append(f"cd {repo_path}")
    return lines


def _run_platform_runi(args):
    import subprocess

    ssh_host, srun, repo_path, venv_path, path_cuda, path_ws = _platform_srun_context(args.platform)

    # Write a small activation script so bash --init-file can source it without
    # wrapping srun in a bash -c subshell (which breaks the PTY).
    init_lines = _srun_env_lines(path_cuda, venv_path, repo_path, path_ws)
    remote_init = f"{path_ws}/.od3d_init" if path_ws else "~/.od3d_init"
    subprocess.run(
        ["ssh", ssh_host, f"cat > {remote_init}"],
        input="\n".join(init_lines), text=True, check=True,
    )

    srun += f" --pty bash --init-file {remote_init}"
    print(f"Opening interactive session on {ssh_host} in {repo_path or path_ws or '~'}…")
    subprocess.run(["ssh", "-t", ssh_host, srun])


def _run_platform_setupi(args):
    """Interactive setup: open a compute-node shell, copy setup_slurm.sh, and print it."""
    import re
    import subprocess

    ssh_host, srun, repo_path, venv_path, path_cuda, path_ws = _platform_srun_context(args.platform)

    cfg, _ = _load_platform_config(args.platform)
    username = cfg.get("username", "")

    # Locate setup_slurm.sh in the outermost git repo (same logic as _run_platform_setup)
    try:
        submodule_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True, cwd=Path(__file__).parent,
        ).strip()
        superproject = subprocess.check_output(
            ["git", "rev-parse", "--show-superproject-working-tree"], text=True, cwd=submodule_root,
        ).strip()
        local_repo_root = superproject if superproject else submodule_root
    except subprocess.CalledProcessError:
        local_repo_root = str(Path.cwd())

    setup_script_local = Path(local_repo_root) / "setup" / "setup_slurm.sh"
    if not setup_script_local.is_file():
        raise FileNotFoundError(f"Setup script not found: {setup_script_local}")

    remote_setup = f"{path_ws}/setup_slurm.sh" if path_ws else "~/setup_slurm.sh"

    def _scp(local, remote):
        target = f"{ssh_host}:{remote}"
        if username:
            target = f"{username}@{ssh_host}:{remote}"
        print(f"Copying {local} → {target}")
        subprocess.run(["scp", str(local), target], check=True)

    _scp(setup_script_local, remote_setup)

    init_lines = _srun_env_lines(path_cuda, venv_path, repo_path, path_ws)
    init_lines += [
        "",
        f'echo "================================================================"',
        f'echo "  Setup script : {remote_setup}"',
        f'echo "  To execute   : bash {remote_setup}"',
        f'echo "================================================================"',
        f'echo ""',
        f'cat {remote_setup}',
        f'echo ""',
        f'echo "================================================================"',
        f'echo "  Run: bash {remote_setup}"',
        f'echo "================================================================"',
    ]

    remote_init = f"{path_ws}/.od3d_setupi_init" if path_ws else "~/.od3d_setupi_init"
    subprocess.run(
        ["ssh", ssh_host, f"cat > {remote_init}"],
        input="\n".join(init_lines), text=True, check=True,
    )

    srun += f" --pty bash --init-file {remote_init}"
    print(f"Opening interactive setup session on {ssh_host}…")
    subprocess.run(["ssh", "-t", ssh_host, srun])


def _run_platform_run_cmd(platform: str, command: str) -> None:
    import subprocess

    ssh_host, srun, repo_path, venv_path, path_cuda, path_ws = _platform_srun_context(platform)

    script_lines = _srun_env_lines(path_cuda, venv_path, repo_path, path_ws) + [command]
    remote_script = f"{path_ws}/.od3d_run" if path_ws else "~/.od3d_run"
    subprocess.run(
        ["ssh", ssh_host, f"cat > {remote_script} && chmod +x {remote_script}"],
        input="\n".join(script_lines), text=True, check=True,
    )

    srun += f" bash {remote_script}"
    print(f"Running on {ssh_host} in {repo_path or path_ws or '~'}: {command}")
    subprocess.run(["ssh", ssh_host, srun])


def _run_platform_run(args):
    _run_platform_run_cmd(args.platform, args.command)


def _run_platform_stop(args) -> None:
    import subprocess

    platform = args.platform
    cfg, _   = _load_platform_config(platform)

    ssh_host = cfg.get("ssh")
    if not ssh_host or ssh_host is False:
        raise ValueError(
            f"Platform '{platform}' has no ssh host configured (ssh: {ssh_host!r})"
        )

    username  = cfg.get("username", "")
    partition = cfg.get("partition", None)
    job_range = getattr(args, "jobs", None)

    if job_range:
        # ── range mode: cancel job IDs A through B inclusive ──────────
        try:
            a, b = job_range.split("-")
            id_start, id_end = int(a), int(b)
        except ValueError:
            raise ValueError(f"Invalid job range {job_range!r} — expected format A-B")

        job_ids = list(range(id_start, id_end + 1))
        ids_str = " ".join(str(j) for j in job_ids)

        # Preview: query only those specific job IDs for this user
        squeue_cmd = f"squeue -j {','.join(str(j) for j in job_ids)} --format='%.10i %.12P %.30j %.10T %.12M' 2>/dev/null"
        if username:
            squeue_cmd += f" -u {username}"
        result = subprocess.run(["ssh", ssh_host, squeue_cmd], capture_output=True, text=True)
        lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
        n_found = max(0, len(lines) - 1)

        if lines:
            print(result.stdout.strip())
        print(f"\n{n_found} job(s) found in range {id_start}–{id_end} ({len(job_ids)} IDs checked).")

        if not args.yes:
            print(f"Cancel all {len(job_ids)} job IDs in range? [y/N] ", end="", flush=True)
            if input().strip().lower() != "y":
                print("Aborted.")
                return

        scancel_range_cmd = f"scancel {ids_str}"
        if username:
            scancel_range_cmd += f" --user={username}"
        print(f"Running: scancel {id_start}…{id_end}")
        subprocess.run(["ssh", ssh_host, scancel_range_cmd], check=True)
        print(f"Cancelled job IDs {id_start}–{id_end}.")
        return

    # ── default mode: cancel all jobs for user/partition ──────────────
    squeue_cmd = "squeue --format='%.10i %.12P %.30j %.10T %.12M'"
    if username:
        squeue_cmd += f" -u {username}"
    if partition:
        squeue_cmd += f" -p {partition}"

    info = (f"partition={partition}" if partition else "") + \
           (f"  user={username}" if username else "")
    print(f"Querying jobs on {ssh_host}  [{info.strip()}]…")
    result = subprocess.run(["ssh", ssh_host, squeue_cmd], capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()

    if len(lines) <= 1:
        print("No running jobs found.")
        return

    print(result.stdout.strip())
    n_jobs = len(lines) - 1  # subtract header

    if not args.yes:
        print(f"\nCancel all {n_jobs} job(s)? [y/N] ", end="", flush=True)
        if input().strip().lower() != "y":
            print("Aborted.")
            return

    scancel_cmd = "scancel"
    if username:
        scancel_cmd += f" -u {username}"
    if partition:
        scancel_cmd += f" -p {partition}"

    print(f"Running: {scancel_cmd}")
    subprocess.run(["ssh", ssh_host, scancel_cmd], check=True)
    print(f"Cancelled {n_jobs} job(s).")


def _run_platform_queue(args):
    import subprocess

    cfg, _ = _load_platform_config(args.platform)

    ssh_host = cfg.get("ssh")
    if not ssh_host or ssh_host is False:
        raise ValueError(f"Platform '{args.platform}' has no ssh host configured")

    username  = cfg.get("username", "")
    partition = getattr(args, "partition", None) or cfg.get("partition", None)
    all_users = getattr(args, "all_users", False)

    # %.Nf = right-align in N chars; %b = gres (gpu:type:count); %m = mem/node
    fmt = "%.10i %.12P %.60j %.10u %.10T %5C %10m %10b %.12M %.12l %R"
    cmd = f"squeue --format={fmt!r}"
    if not all_users and username:
        cmd += f" -u {username}"
    if partition:
        cmd += f" -p {partition}"

    header = f"Queue on {ssh_host}"
    if partition:
        header += f" · partition {partition}"
    if not all_users and username:
        header += f" · user {username}"
    # Partition totals from scontrol (TRES line has exact cpu/mem/gpu counts)
    import re
    sctl_cmd = f"scontrol show partition {partition}" if partition else "scontrol show partition"
    sctl_out = subprocess.run(
        ["ssh", ssh_host, sctl_cmd], capture_output=True, text=True,
    ).stdout
    tres_match = re.search(r"TRES=([^\n]+)", sctl_out)
    total_cpus = total_gpus = mem_gb = 0
    if tres_match:
        tres = tres_match.group(1)
        m = re.search(r"(?<![/\w])cpu=(\d+)", tres)
        if m:
            total_cpus = int(m.group(1))
        m = re.search(r"(?<![/\w])mem=(\d+)([KMGT]?)", tres)
        if m:
            val, unit = int(m.group(1)), m.group(2) or "M"
            mem_gb = val if unit == "G" else val * 1024 if unit == "T" else val // 1024
        m = re.search(r"gres/gpu=(\d+)", tres)
        if m:
            total_gpus = int(m.group(1))

    summary = f"cpu={total_cpus}  mem={mem_gb}GB  gpu={total_gpus}"
    if total_gpus:
        summary += (f"  cpu/gpu={total_cpus / total_gpus:.1f}"
                    f"  mem/gpu={mem_gb / total_gpus:.0f}GB")

    print("=" * 70)
    print(header)
    print("=" * 70)
    subprocess.run(["ssh", ssh_host, cmd], check=True)
    print("=" * 70)
    print(summary)
    print("=" * 70)


def _run_platform(args):
    if args.platform_command == "setup":
        _run_platform_setup(args)
    elif args.platform_command == "overview":
        _run_platform_overview(args)
    elif args.platform_command == "runi":
        _run_platform_runi(args)
    elif args.platform_command == "setupi":
        _run_platform_setupi(args)
    elif args.platform_command == "run":
        _run_platform_run(args)
    elif args.platform_command == "stop":
        _run_platform_stop(args)
    elif args.platform_command == "queue":
        _run_platform_queue(args)


# ── bench sub-parser ──────────────────────────────────────────────────────────

def _resolve_bench_config(name_or_path: str) -> Path:
    """Resolve a benchmark config name or path to an absolute Path.

    Accepts a full/relative path (used as-is if it exists) or a short name
    resolved against src/configs/eval/ or src/configs/ relative to CWD.
    """
    p = Path(name_or_path)
    if p.exists():
        return p.resolve()
    stem = name_or_path if not name_or_path.endswith(".yaml") else name_or_path[:-5]
    for subdir in ("src/configs/bench", "src/configs/eval", "src/configs"):
        candidate = Path.cwd() / subdir / f"{stem}.yaml"
        if candidate.exists():
            return candidate
    raise argparse.ArgumentTypeError(
        f"Benchmark config not found: {name_or_path!r}\n"
        f"  Tried: {p.resolve()}\n"
        f"  Tried: {Path.cwd() / 'src/configs/bench' / (stem + '.yaml')}\n"
        f"  Tried: {Path.cwd() / 'src/configs/eval' / (stem + '.yaml')}\n"
        f"  Tried: {Path.cwd() / 'src/configs' / (stem + '.yaml')}"
    )


def _resolve_ablation(name_or_path: str) -> list[Path]:
    """Resolve a comma-separated list of ablation names/paths to resolved Paths.

    Each entry may be a full/relative path (file or directory) or a short name
    resolved against src/configs/ablation/ relative to CWD.
    """
    result = []
    for part in name_or_path.split(","):
        part = part.strip()
        if not part:
            continue
        p = Path(part)
        if p.exists():
            result.append(p.resolve())
            continue
        candidate = Path.cwd() / "src" / "configs" / "ablation" / part
        if candidate.exists():
            result.append(candidate)
            continue
        raise argparse.ArgumentTypeError(
            f"Ablation not found: {part!r}\n"
            f"  Tried: {p.resolve()}\n"
            f"  Tried: {candidate}"
        )
    return result


def _ablation_files(ablation: Path) -> list[Path]:
    """Return sorted list of YAML files for a single dir or file."""
    if ablation.is_file():
        return [ablation]
    return sorted(ablation.glob("*.yaml"))


def _ablation_combinations(ablations: list[Path]) -> list[tuple[Path, ...]]:
    """Return the Cartesian product of YAML files across all ablation entries.

    Each entry expands to its set of YAML files (or itself if a single file).
    A single entry with N files → N 1-tuples; two entries with M and N files →
    M*N 2-tuples where both configs are merged for each run.
    """
    import itertools
    per_entry = [_ablation_files(a) for a in ablations]
    return list(itertools.product(*per_entry))


def _repo_rel(path: Path) -> str:
    """Return path relative to CWD (repo root) when possible, else absolute."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _build_bench_parser(sub):
    p = sub.add_parser("bench", help="Benchmark commands")
    bench_sub = p.add_subparsers(dest="bench_command", required=True)

    def _add_bench_args(q):
        q.add_argument(
            "-b", "--benchmark", required=True, type=_resolve_bench_config, metavar="BENCHMARK",
            help="Benchmark config name (resolved from src/configs/bench/) or full path to YAML",
        )
        q.add_argument(
            "-p", "--platform", default=None, metavar="PLATFORM",
            help="Override the platform from the benchmark config's defaults list",
        )
        q.add_argument(
            "-a", "--ablation", default=None, type=_resolve_ablation, metavar="ABLATION",
            help="Comma-separated ablation dirs/files (names resolved from src/configs/ablation/); "
                 "each YAML across all entries is merged on top of the benchmark config and run in sequence",
        )

    p_run = bench_sub.add_parser("run", help="Run benchmark(s) locally")
    _add_bench_args(p_run)

    p_rrun = bench_sub.add_parser("rrun", help="Submit benchmark(s) as remote jobs via o3b platform run")
    _add_bench_args(p_rrun)
    p_rrun.add_argument(
        "-d", "--deps", default=None, metavar="DEPS",
        help="Comma-separated dep sets that override the platform config's deps field "
             "(e.g. -d diff3f or -d densematcher,diff3f). Controls venv selection.",
    )
    p_rrun.add_argument(
        "--force", action="store_true",
        help="Submit jobs even if they are already running, pending, or recently completed.",
    )
    p_rrun.add_argument(
        "--skip-fetched", action="store_true",
        help="Skip jobs whose ablation combo already has a row in the fetched tables/ CSV.",
    )

    def _add_fetch_args(q):
        _add_bench_args(q)
        q.add_argument(
            "-e", "--entity", default=None, metavar="ENTITY",
            help="W&B entity (user or team). Defaults to the logged-in user's default entity.",
        )
        q.add_argument(
            "-o", "--output", default=None, type=Path, metavar="FILE",
            help="Output CSV path (default: <benchmark>.csv in CWD)",
        )

    p_fetch = bench_sub.add_parser("fetch", help="Fetch eval metrics from wandb and save to CSV")
    _add_fetch_args(p_fetch)

    p_viz = bench_sub.add_parser("viz", help="Interactively plot eval metrics from a bench CSV")
    _add_fetch_args(p_viz)


def _run_bench_fetch(args) -> None:
    """Fetch eval/* metrics from wandb for all job names that rrun would submit."""
    import csv
    import re
    import yaml

    bench_stem = args.benchmark.stem

    with open(args.benchmark) as f:
        raw = yaml.safe_load(f) or {}
    wandb_cfg = raw.get("wandb") or {}
    wb_project = wandb_cfg.get("project", bench_stem)
    wb_entity   = args.entity or wandb_cfg.get("entity", None)
    project_path = f"{wb_entity}/{wb_project}" if wb_entity else wb_project

    if args.ablation:
        combos = _ablation_combinations(args.ablation)
        if not combos:
            print(f"WARNING: no YAML files found in {args.ablation!r}")
            return
    else:
        combos = [()]

    try:
        import wandb as _wb_mod
        api = _wb_mod.Api()
    except ImportError:
        print("ERROR: wandb is not installed. Run: pip install wandb")
        return

    _ablation_base = Path.cwd() / "src" / "configs" / "ablation"
    def _ablation_col_name(a: Path) -> str:
        try:
            return str(a.relative_to(_ablation_base))
        except ValueError:
            return a.name
    ablation_col_names = [_ablation_col_name(a) for a in args.ablation] if args.ablation else []

    rows = []
    for combo in combos:
        job_name = (
            f"{bench_stem}__{'__'.join(f.stem for f in combo)}"
            if combo else bench_stem
        )
        # wandb run name is f"{timestamp}__{job_name}" where timestamp is MMDD_HHMMSS
        pattern = rf"^\d{{4}}_\d{{6}}__{re.escape(job_name)}$"
        print(f"Querying {project_path!r} for {job_name!r} …")
        try:
            matched = list(api.runs(project_path, filters={"display_name": {"$regex": pattern}}))
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        if not matched:
            print("  no runs found")
            continue

        finished = [r for r in matched if r.state == "finished"]
        if not finished:
            print(f"  no finished runs (found {len(matched)} with other states)")
            continue
        run = max(finished, key=lambda r: r.name)

        row: dict = {"job": job_name, "wandb_run": run.name, "state": run.state}
        for col, f in zip(ablation_col_names, combo):
            row[col] = f.stem
        for k, v in run.summary.items():
            if k.startswith("eval/") and isinstance(v, (int, float)):
                row[k] = round(float(v), 6)
        rows.append(row)
        n_metrics = sum(1 for k in row if k.startswith("eval/"))
        print(f"  {run.name}  state={run.state}  eval_metrics={n_metrics}")

    if not rows:
        print("No matching runs found.")
        return

    meta_cols   = ["job", "wandb_run", "state"] + ablation_col_names
    metric_cols = sorted({k for row in rows for k in row if k.startswith("eval/")})
    fieldnames  = meta_cols + metric_cols

    if ablation_col_names:
        safe = [n.replace("/", "_") for n in ablation_col_names]
        table_stem = f"{bench_stem}__{'__'.join(safe)}"
    else:
        table_stem = bench_stem
    output = args.output or Path("tables") / f"{table_stem}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} row(s) → {output}")


def _run_bench_viz(args) -> None:
    """Load (or fetch) the bench CSV and show an interactive bar-plot for a chosen metric."""
    import csv
    from collections import defaultdict

    bench_stem = args.benchmark.stem
    _ablation_base = Path.cwd() / "src" / "configs" / "ablation"
    if args.ablation:
        abl_names = []
        for a in args.ablation:
            try:
                abl_names.append(str(a.relative_to(_ablation_base)).replace("/", "_"))
            except ValueError:
                abl_names.append(a.name.replace("/", "_"))
        table_stem = f"{bench_stem}__{'__'.join(abl_names)}"
    else:
        table_stem = bench_stem
    csv_path   = args.output or Path("tables") / f"{table_stem}.csv"

    if not csv_path.exists():
        print(f"CSV not found at {csv_path} — fetching from wandb first …")
        _run_bench_fetch(args)

    if not csv_path.exists():
        print("ERROR: could not obtain CSV.")
        return

    with open(csv_path, newline="") as f:
        reader    = csv.DictReader(f)
        all_rows  = list(reader)
        fieldnames = reader.fieldnames or []

    metric_cols = [c for c in fieldnames if c.startswith("eval/") and c != "eval/n_samples"]
    if not metric_cols:
        print("No eval/* columns found in CSV.")
        return

    # For each job keep only the latest finished run that has at least one metric value
    by_job: dict[str, list] = defaultdict(list)
    for row in all_rows:
        by_job[row["job"]].append(row)

    best_rows = []
    for job in sorted(by_job):
        candidates = [
            r for r in by_job[job]
            if r.get("state") == "finished"
            and any(r.get(m, "").strip() != "" for m in metric_cols)
        ]
        if candidates:
            best_rows.append(max(candidates, key=lambda r: r["wandb_run"]))

    if not best_rows:
        print("No finished runs with metrics found in CSV.")
        return

    available = [m for m in metric_cols if any(r.get(m, "").strip() != "" for r in best_rows)]
    if not available:
        print("No non-empty metric values found.")
        return

    # Interactive metric selection
    print("\nAvailable metrics:")
    for i, m in enumerate(available, 1):
        print(f"  [{i}] {m}")
    while True:
        raw = input(f"\nSelect metric [1-{len(available)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(available):
            metric = available[int(raw) - 1]
            break
        print(f"  Enter a number between 1 and {len(available)}.")

    # Filter to runs that have a value for the chosen metric
    plot_rows = [r for r in best_rows if r.get(metric, "").strip() != ""]
    if not plot_rows:
        print(f"No rows with values for {metric!r}.")
        return

    import matplotlib.pyplot as plt
    import numpy as np

    # ── two ablation directions → heatmap ─────────────────────────────
    if args.ablation and len(args.ablation) == 2:
        _abl_base = Path.cwd() / "src" / "configs" / "ablation"
        def _abl_col(a: Path) -> str:
            try:
                return str(a.relative_to(_abl_base))
            except ValueError:
                return a.name
        col_a, col_b = [_abl_col(a) for a in args.ablation]

        vals_a = sorted({r[col_a] for r in plot_rows if col_a in r})
        vals_b = sorted({r[col_b] for r in plot_rows if col_b in r})

        mat = np.full((len(vals_a), len(vals_b)), float("nan"))
        for r in plot_rows:
            if col_a not in r or col_b not in r:
                continue
            ia = vals_a.index(r[col_a])
            ib = vals_b.index(r[col_b])
            mat[ia, ib] = float(r[metric])

        # extend matrix with average row (bottom) and average column (right)
        avg_col  = np.nanmean(mat, axis=1, keepdims=True)   # per row  → right column
        avg_row  = np.nanmean(mat, axis=0, keepdims=True)   # per col  → bottom row
        avg_all  = np.nanmean(mat, keepdims=True).reshape(1, 1)
        ext_mat  = np.block([
            [mat,     avg_col],
            [avg_row, avg_all],
        ])
        xticks = list(vals_b) + ["Avg"]
        yticks = list(vals_a) + ["Avg"]
        na, nb = len(vals_a), len(vals_b)

        fig, ax = plt.subplots(figsize=(max(6, (nb + 1) * 0.9 + 1), max(3, (na + 1) * 0.7 + 1)))
        im = ax.imshow(ext_mat, aspect="auto", cmap="viridis")
        plt.colorbar(im, ax=ax, label=metric)

        ax.set_xticks(range(nb + 1))
        ax.set_xticklabels(xticks, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(na + 1))
        ax.set_yticklabels(yticks, fontsize=8)
        ax.set_xlabel(col_b, fontsize=9)
        ax.set_ylabel(col_a, fontsize=9)
        ax.set_title(f"{bench_stem}  —  {metric}", fontsize=10)

        # separator lines before the average row/column
        ax.axhline(na - 0.5, color="white", linewidth=1.5, linestyle="--")
        ax.axvline(nb - 0.5, color="white", linewidth=1.5, linestyle="--")

        vmin, vmax = np.nanmin(ext_mat), np.nanmax(ext_mat)
        mid = (vmin + vmax) / 2
        for ia in range(na + 1):
            for ib in range(nb + 1):
                v = ext_mat[ia, ib]
                if not np.isnan(v):
                    color = "white" if v < mid else "black"
                    ax.text(ib, ia, f"{v:.4f}", ha="center", va="center", fontsize=7, color=color)

        plt.tight_layout()
        plt.show()
        return

    # ── single ablation direction → bar chart ─────────────────────────
    labels = [r["job"].split("__")[-1] for r in plot_rows]
    values = [float(r[metric]) for r in plot_rows]

    pairs  = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
    labels, values = [p[0] for p in pairs], [p[1] for p in pairs]

    avg = sum(values) / len(values)
    labels.append("Average")
    values.append(avg)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
    colors  = ["steelblue"] * (len(labels) - 1) + ["darkorange"]
    bars    = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=7)
    ax.set_title(f"{bench_stem}  —  {metric}", fontsize=10)
    ax.set_ylabel(metric)
    ax.set_xlabel("category")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.show()


def _run_bench(args) -> None:
    if args.bench_command == "run":
        _run_bench_run(args)
    elif args.bench_command == "rrun":
        _run_bench_rrun(args)
    elif args.bench_command == "fetch":
        _run_bench_fetch(args)
    elif args.bench_command == "viz":
        _run_bench_viz(args)


def _run_bench_run(args) -> None:
    import yaml
    from omegaconf import OmegaConf

    from o3b.dataset.dataset import _load_yaml_with_defaults
    from o3b.dataset.cli import _platform_to_dataset_overrides, _resolve_dataset_config
    from o3b.run import _run_bench_run_with_cfg

    with open(args.benchmark) as f:
        raw = yaml.safe_load(f)

    # ── resolve platform and dataset from defaults list ───────────────────────
    defaults = raw.pop("defaults", []) or []
    default_platform = None
    default_dataset  = None
    for item in defaults:
        if isinstance(item, dict):
            default_platform = item.get("platform", default_platform)
            default_dataset  = item.get("dataset",  default_dataset)

    platform = args.platform if args.platform is not None else (default_platform or "default")

    # ── load base dataset config once (shared across all ablations) ───────────
    overrides = _platform_to_dataset_overrides(platform)
    if default_dataset:
        ds_base = _load_yaml_with_defaults(_resolve_dataset_config(default_dataset), overrides=overrides)
    else:
        ds_base = {}

    # ── collect ablation combinations (Cartesian product across entries) ────────
    if args.ablation:
        combos = _ablation_combinations(args.ablation)
        if not combos:
            print(f"WARNING: no YAML files found in {args.ablation!r}")
            return
    else:
        combos = [()]  # single run with no ablation

    # ── run once per combination ──────────────────────────────────────────────
    for combo in combos:
        if combo:
            # merge all ablation YAMLs in the combo sequentially
            merged_ablation = OmegaConf.create({})
            for ablation_file in combo:
                with open(ablation_file) as f:
                    merged_ablation = OmegaConf.merge(
                        merged_ablation, OmegaConf.create(yaml.safe_load(f) or {})
                    )
            run_raw = OmegaConf.to_container(
                OmegaConf.merge(OmegaConf.create(dict(raw)), merged_ablation),
                resolve=True,
            )
            combo_stem = "__".join(f.stem for f in combo)
            print(f"\n{'='*60}")
            print(f"Ablation: {combo_stem}")
            print(f"{'='*60}")
        else:
            run_raw = raw
            combo_stem = None

        # merge benchmark/ablation dataset section on top of base dataset config
        run_ds = run_raw.get("dataset") or {}
        ds_merged = OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.create(ds_base), OmegaConf.create(run_ds)),
            resolve=True,
        ) if run_ds else dict(ds_base)

        from datetime import datetime
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        if combo_stem:
            run_name = f"{timestamp}__{args.benchmark.stem}__{combo_stem}"
        else:
            run_name = f"{timestamp}__{args.benchmark.stem}"
        _run_bench_run_with_cfg({**run_raw, "dataset": ds_merged}, run_name)


def _run_bench_sbatch_cmd(platform: str, command: str, job_name: str, deps_override: list | None = None) -> None:
    """Upload a run script + sbatch wrapper and submit via sbatch."""
    import os, re, subprocess
    from omegaconf import OmegaConf

    cfg, _ = _load_platform_config(platform)

    ssh_host = cfg.get("ssh")
    if not ssh_host or ssh_host is False:
        raise ValueError(f"Platform '{platform}' has no ssh host configured")

    path_ws        = cfg.get("path_ws", "")
    path_cuda      = cfg.get("path_cuda", "/usr/local/cuda-12.4")
    python_version = str(cfg.get("python_version", "3.10"))
    torch_version  = str(cfg.get("torch_version", "2.6.0"))
    deps                 = deps_override if deps_override is not None else list(cfg.get("deps", []) or [])
    deps_tag             = "_".join(sorted(deps)) if deps else ""
    install_diff3f       = "true" if (cfg.get("install_diff3f", False) or "diff3f" in deps) else "false"
    install_densematcher = "true" if (cfg.get("install_densematcher", False) or "densematcher" in deps) else "false"
    setup          = "true" if cfg.get("setup", False) else "false"
    branch         = str(cfg.get("branch", "main"))
    pull           = str(cfg.get("pull", True)).lower()
    pull_subs      = str(cfg.get("pull_submodules", True)).lower()
    path_home      = cfg.get("path_home", path_ws)

    cuda_tag  = "cu" + os.path.basename(path_cuda).replace("cuda-", "").replace(".", "")
    py_tag    = "py" + python_version.replace(".", "")
    torch_tag = "torch" + ".".join(torch_version.split(".")[:2]).replace(".", "")

    token = OmegaConf.select(cfg, "credentials.github.token", default="") or ""
    try:
        submodule_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True, cwd=Path(__file__).parent,
        ).strip()
        superproject = subprocess.check_output(
            ["git", "rev-parse", "--show-superproject-working-tree"], text=True, cwd=submodule_root,
        ).strip()
        local_repo_root = superproject if superproject else submodule_root
        raw_remote = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], text=True, cwd=local_repo_root,
        ).strip()
        if raw_remote.startswith("git@"):
            raw_remote = re.sub(r"git@github\.com:", "https://github.com/", raw_remote)
        plain    = re.sub(r"https://[^@]+@", "https://", raw_remote)
        repo_url  = plain.replace("https://", f"https://{token}@") if token else plain
        repo_name = Path(re.sub(r"\.git$", "", plain.split("/")[-1])).name
    except subprocess.CalledProcessError:
        repo_url  = ""
        repo_name = ""

    repo_path = f"{path_ws}/{repo_name}" if (path_ws and repo_name) else path_ws
    _venv_suffix = f"_{deps_tag}" if deps_tag else ""
    venv_path = f"{repo_path}/venv_{py_tag}_{cuda_tag}_{torch_tag}{_venv_suffix}" if repo_path else ""

    _proxy = "http://tfproxy.informatik.intra.uni-freiburg.de:8080"
    env_vars = {
        "PATH_WS":         path_ws,
        "PATH_CUDA":       path_cuda,
        "PYTHON_VERSION":  python_version,
        "TORCH_VERSION":   torch_version,
        "INSTALL_DIFF3F":        install_diff3f,
        "INSTALL_DENSEMATCHER":  install_densematcher,
        "DEPS_TAG":              deps_tag,
        "REPO_URL":        repo_url,
        "REPO_NAME":       repo_name,
        "SETUP":           setup,
        "BRANCH":          branch,
        "PULL":            pull,
        "PULL_SUBMODULES": pull_subs,
        "HTTP_PROXY":      _proxy,
        "HTTPS_PROXY":     _proxy,
        "http_proxy":      _proxy,
        "https_proxy":     _proxy,
    }

    from datetime import datetime
    ts = datetime.now().strftime("%m%d_%H%M%S")

    # run script: env preamble (CUDA, venv, cd, setup/pull) + the actual command
    run_script_content = "\n".join(
        ["#!/usr/bin/env bash", "set -euo pipefail", ""] +
        _srun_env_lines(path_cuda, venv_path, repo_path, path_ws) +
        ["", command]
    )
    remote_run_script = f"{path_ws}/.bench_run_{job_name}_{ts}.sh"

    sbatch_script = _make_sbatch_script(
        cfg,
        job_name=job_name,
        env_vars=env_vars,
        remote_setup_script=remote_run_script,
    )
    remote_sbatch = f"{path_ws}/.bench_sbatch_{job_name}_{ts}.sh"

    # Save both scripts locally with the same timestamp before sending to remote
    local_run    = _save_script_locally(f"bench_run_{job_name}",    run_script_content, ts=ts)
    local_sbatch = _save_script_locally(f"bench_sbatch_{job_name}", sbatch_script,      ts=ts)
    print(f"  saved locally: {local_run}")
    print(f"  saved locally: {local_sbatch}")

    subprocess.run(
        ["ssh", ssh_host, f"cat > {remote_run_script} && chmod +x {remote_run_script}"],
        input=run_script_content, text=True, check=True,
    )
    subprocess.run(
        ["ssh", ssh_host, f"cat > {remote_sbatch} && chmod +x {remote_sbatch}"],
        input=sbatch_script, text=True, check=True,
    )

    remote_submit = f"mkdir -p {path_home}/slurm_jobs && sbatch {remote_sbatch}"
    print(f"Submitting sbatch job '{job_name}' on {ssh_host}…")
    subprocess.run(["ssh", ssh_host, remote_submit], check=True)


def _get_existing_jobs_on_platform(platform: str) -> set[str]:
    """Return job names that are pending/running OR completed in the last 24 hours."""
    import subprocess

    try:
        cfg, _ = _load_platform_config(platform)
    except Exception:
        return set()

    ssh_host = cfg.get("ssh")
    if not ssh_host or ssh_host is False:
        return set()

    username = cfg.get("username", "")

    # pending / running jobs via squeue
    cmd = "squeue --noheader --format=%j"
    if username:
        cmd += f" -u {username}"
    result = subprocess.run(["ssh", ssh_host, cmd], capture_output=True, text=True)
    names: set[str] = set()
    if result.returncode == 0:
        names = {line.strip() for line in result.stdout.splitlines() if line.strip()}

    # completed jobs in the last 24 hours via sacct
    try:
        completed = _fetch_jobs(ssh_host, username, hours=24.0)
        names |= {j["JobName"] for j in completed if j.get("State", "").startswith("COMPLETED")}
    except Exception:
        pass

    return names


def _run_bench_rrun(args) -> None:
    """Submit each benchmark/ablation run as a separate sbatch job."""
    import shlex

    from omegaconf import OmegaConf as _OC

    platform = args.platform or "slurm"
    bench_stem = args.benchmark.stem
    cli_deps = [d.strip() for d in args.deps.split(",")] if getattr(args, "deps", None) else None

    if args.ablation:
        combos = _ablation_combinations(args.ablation)
        if not combos:
            print(f"WARNING: no YAML files found in {args.ablation!r}")
            return
    else:
        combos = [()]

    force = getattr(args, "force", False)
    n_total = len(combos)
    print(f"Checking {n_total} job(s) on {platform}…")
    existing_jobs = set() if force else _get_existing_jobs_on_platform(platform)

    # build set of already-fetched ablation combos from the tables/ CSV
    fetched_combos: set[tuple] = set()
    if getattr(args, "skip_fetched", False):
        import csv as _csv
        _ablation_base = Path.cwd() / "src" / "configs" / "ablation"
        def _abl_col(a: Path) -> str:
            try:
                return str(a.relative_to(_ablation_base))
            except ValueError:
                return a.name
        abl_col_names = [_abl_col(a) for a in args.ablation] if args.ablation else []
        if abl_col_names:
            safe = [n.replace("/", "_") for n in abl_col_names]
            table_stem = f"{bench_stem}__{'__'.join(safe)}"
        else:
            table_stem = bench_stem
        table_path = Path("tables") / f"{table_stem}.csv"
        if table_path.exists():
            with open(table_path, newline="") as _f:
                for _row in _csv.DictReader(_f):
                    fetched_combos.add(tuple(_row.get(col, "") for col in abl_col_names))
            print(f"Loaded {len(fetched_combos)} fetched combo(s) from {table_path}")
        else:
            print(f"WARNING: --skip-fetched set but table not found: {table_path}")

    n_existing = 0
    n_submitted = 0
    width = len(str(n_total))

    for i, combo in enumerate(combos, 1):
        # collect platform.deps from ablation YAMLs (union across all files in combo)
        ablation_deps: list | None = None
        for f in combo:
            try:
                acfg = _OC.load(f)
                file_deps = _OC.select(acfg, "platform.deps", default=None)
                if file_deps is not None:
                    if ablation_deps is None:
                        ablation_deps = []
                    ablation_deps.extend(list(file_deps))
            except Exception:
                pass
        # precedence: CLI -d > ablation platform.deps > platform config deps
        effective_deps = cli_deps if cli_deps is not None else ablation_deps

        parts = ["o3b", "bench", "run",
                 "-b", _repo_rel(args.benchmark),
                 "-p", platform]
        if combo:
            # pass each file individually; the remote `bench run` will receive a
            # comma-joined -a arg so its own outer-product logic runs the same combo
            parts += ["-a", ",".join(_repo_rel(f) for f in combo)]
            job_name = f"{bench_stem}__{'__'.join(f.stem for f in combo)}"
        else:
            job_name = bench_stem
        remote_cmd = " ".join(shlex.quote(p) for p in parts)

        prefix = f"[{i:{width}}/{n_total}]"
        combo_key = tuple(f.stem for f in combo)
        if fetched_combos and combo_key in fetched_combos:
            n_existing += 1
            print(f"{prefix} skip (in table) {job_name}")
            continue
        if job_name in existing_jobs:
            n_existing += 1
            print(f"{prefix} skip   {job_name}")
            continue

        print(f"{prefix} submit {job_name}")
        _run_bench_sbatch_cmd(platform, remote_cmd, job_name, deps_override=effective_deps)
        n_submitted += 1
        time.sleep(1)

    print(f"\nDone — {n_submitted} submitted, {n_existing} already running/pending/completed"
          + (f", {n_total - n_submitted - n_existing} skipped for other reasons" if n_submitted + n_existing < n_total else ""))


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="o3b",
        description="o3b CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _build_dataset_parser(sub)
    _build_bench_parser(sub)
    _build_platform_parser(sub)

    args = parser.parse_args(argv)

    if args.command == "dataset":
        _run_dataset(args)
    elif args.command == "bench":
        _run_bench(args)
    elif args.command == "platform":
        _run_platform(args)


if __name__ == "__main__":
    main()
