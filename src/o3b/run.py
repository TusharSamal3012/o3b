from __future__ import annotations


def _run_bench_run_with_cfg(run_raw: dict, run_name: str) -> None:
    """Execute one benchmark evaluation pass given a fully-resolved config dict.

    Args:
        run_raw:  Merged config dict. ``run_raw["dataset"]`` must already be the
                  fully-merged dataset config (base defaults + benchmark + ablation).
        run_name: Used as the W&B run name and as the fallback W&B project name.
    """
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf

    from o3b.dataset.dataset import DatasetConfig, build_dataset
    from o3b.task.task import build_task
    from o3b.data.datatypes.object import collate_object_pairs

    dataset_cfg = DatasetConfig.from_dict(run_raw["dataset"])
    dataset     = build_dataset(dataset_cfg)
    print(f"Dataset: {dataset_cfg.class_name}  ({len(dataset)} items)")

    eval_cfg   = run_raw.get("eval") or {}
    batch_size = eval_cfg.get("batch_size", 4)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_object_pairs,
        shuffle=False,
        num_workers=0,
    )

    task_cfg = OmegaConf.create(run_raw["task"])
    task     = build_task(task_cfg)
    print(f"Task:    {run_raw['task']['class_name']}")
    print(f"Eval:    batch_size={batch_size}  n_batches={len(loader)}\n")

    # ── wandb init ────────────────────────────────────────────────────────────
    _wb = None
    wandb_cfg = run_raw.get("wandb") or {}
    if wandb_cfg is not False:
        try:
            import wandb as _wb_mod
            wb_project = wandb_cfg.get("project", run_name)
            _wb_mod.init(
                project=wb_project,
                name=run_name,
                config=run_raw,
                reinit=True,
            )
            _wb = _wb_mod
            print(f"W&B:     project={wb_project}  run={run_name}")
        except ImportError:
            print("INFO: wandb not installed — skipping W&B logging")

    accum: dict[str, list] = {}
    n_samples = 0
    qualit_log_batches = eval_cfg.get("qualit_log_batches", 8)

    for batch_idx, batch in enumerate(loader):
        quant, qualit = task(batch)

        B = (batch.src_obj_kpts3d.shape[0]
             if batch.src_obj_kpts3d is not None else batch_size)
        n_samples += B

        for metric_name, value in quant.mean().items():
            accum.setdefault(metric_name, []).append(value)

        if _wb is not None:
            wb_log = quant.to_wandb_log(prefix="batch", wb=_wb)
            wb_log.update(qualit.to_wandb_log(
                prefix="qualit",
                wb=_wb,
                log_imgs=(batch_idx < qualit_log_batches),
            ))
            wb_log["batch/n_samples"] = n_samples
            _wb.log(wb_log, step=batch_idx)

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            print(f"  [{batch_idx + 1:4d}/{len(loader)}]  samples={n_samples}", end="")
            for k, vals in accum.items():
                print(f"  {k}={sum(vals)/len(vals):.4f}", end="")
            print()

    print(f"\n{'─'*50}")
    print(f"Results  ({n_samples} samples)")
    for k, vals in accum.items():
        print(f"  {k:<25} {sum(vals)/len(vals):.4f}")

    if _wb is not None:
        final_metrics = {f"eval/{k}": sum(v) / len(v) for k, v in accum.items()}
        final_metrics["eval/n_samples"] = n_samples
        _wb.log(final_metrics)
        _wb.finish()
