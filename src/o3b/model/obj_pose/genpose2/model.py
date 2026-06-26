"""GenPose2 object-pose model.

Wraps the GenPose2 RGB-D pose estimator (``genpose2.runners.infer``) as an
:class:`OD3D_Model`.  It consumes a frame-object (rgb + depth + instance mask +
camera intrinsics) and writes back the predicted metric object pose and size:

  * ``cam_tform4x4_obj``      – metric cam←obj SE(3)
  * ``cam_tform4x4_obj_ncds`` – ncds→cam (= cam_tform4x4_obj @ inv(ncds scale))
  * ``obj_size3d``            – (3,) per-axis bounding-box side lengths (metres)

The ROPE/SOPE frame data and GenPose2 both use the CV camera convention
(+Z forward), so the predicted pose is compatible with ``cam_tform4x4_obj``
without an axis swap.  The NCDS cube follows the dataset convention
``obj_size_ncds = 2.0`` with a uniform scale of ``max(size3d) / 2``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from o3b.model.model import OD3D_Model, register_model

logger = logging.getLogger(__name__)

# NCDS cube max extent (dataset convention: obj_size_ncds == 2.0).
_OBJ_SIZE_NCDS = 2.0

# Default checkpoint download (zip with ScoreNet/EnergyNet/ScaleNet *.pth).
# NOTE: this is a *signed* Dropbox URL and will eventually expire; pass
# ``ckpt_url`` to override it, or place the checkpoints under genpose2_fpath.
_DEFAULT_CKPT_URL = (
    "https://ucfad7d821ef9bcc92e64f8f3c85.dl.dropboxusercontent.com/zip_download_get/"
    "Cmj0-h7_g5zBWMvGABE35k_Z3GzigWOkPelvtAAPMTtQ15S8X7AJ5W4E0Mbd7AreaQN5_YlDTVL1nHBzdlVOOdIaf8igTSmA1ZXoT9lxkBbJ-w"
    "?_download_id=727486970084019193448685378584681882790785785253308828416779983372"
    "&_log_download_success=1&_notify_domain=www.dropbox.com&dl=1"
)


@register_model("GenPose2")
class GenPose2(OD3D_Model):
    def __init__(
        self,
        genpose2_fpath: Optional[str] = None,
        score_model_fname: str = "ScoreNet/scorenet.pth",
        energy_model_fname: str = "EnergyNet/energynet.pth",
        scale_model_fname: str = "ScaleNet/scalenet.pth",
        img_size: int = 224,
        n_pts: int = 1024,
        depth_max: float = 4.0,
        depth_normalize: bool = True,
        tracking: bool = False,
        tracking_T0: float = 0.55,
        ckpt_url: Optional[str] = _DEFAULT_CKPT_URL,
    ):
        super().__init__()
        self.genpose2_fpath = Path(genpose2_fpath) if genpose2_fpath is not None else None
        self.score_model_fname = score_model_fname
        self.energy_model_fname = energy_model_fname
        self.scale_model_fname = scale_model_fname
        self.ckpt_url = ckpt_url
        self.img_size = img_size
        self.n_pts = n_pts
        self.depth_max = depth_max
        self.depth_normalize = depth_normalize
        self.tracking = tracking
        self.tracking_T0 = tracking_T0
        self._prev_pose = None

        self.genpose2 = None
        if self.genpose2_fpath is not None:
            self._init_genpose2()

    def _ckpt_paths(self) -> list[Path]:
        return [
            self.genpose2_fpath / self.score_model_fname,
            self.genpose2_fpath / self.energy_model_fname,
            self.genpose2_fpath / self.scale_model_fname,
        ]

    def _ensure_checkpoints(self) -> None:
        """Download + extract the checkpoints into genpose2_fpath if any are missing."""
        if all(p.exists() for p in self._ckpt_paths()):
            return
        if not self.ckpt_url:
            missing = [str(p) for p in self._ckpt_paths() if not p.exists()]
            raise FileNotFoundError(
                f"GenPose2 checkpoints missing and no ckpt_url set: {missing}",
            )

        import tempfile, urllib.request, zipfile

        self.genpose2_fpath.mkdir(parents=True, exist_ok=True)
        logger.info(f"GenPose2 checkpoints missing → downloading to {self.genpose2_fpath}")

        def _progress(block_num, block_size, total_size):
            if total_size > 0:
                pct = min(block_num * block_size / total_size * 100, 100)
                print(f"\r  [{'#' * int(pct // 2):<50}] {pct:5.1f}%", end="", flush=True)

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            urllib.request.urlretrieve(self.ckpt_url, tmp_path, _progress)
            print()
            with zipfile.ZipFile(tmp_path) as zf:
                # skip the leading "/" / absolute-path entry in the archive
                members = [m for m in zf.namelist() if m not in ("/", "")]
                zf.extractall(self.genpose2_fpath, members=members)
        finally:
            tmp_path.unlink(missing_ok=True)

        still_missing = [str(p) for p in self._ckpt_paths() if not p.exists()]
        if still_missing:
            raise FileNotFoundError(
                f"GenPose2 checkpoints still missing after download (URL may have "
                f"expired — pass a fresh ckpt_url): {still_missing}",
            )

    def _init_genpose2(self) -> None:
        import sys

        self._ensure_checkpoints()

        # GenPose2's get_config() parses sys.argv; mirror the args used in od3d.
        sys.argv = [
            sys.argv[0],
            "--sampler_mode", "ode",
            "--percentage_data_for_test", "1.0",
            "--batch_size", "128", "--seed", "0",
            "--result_dir", "single", "--eval_repeat_num", "50",
            "--clustering", "1", "--T0", "0.55",
            "--dino", "pointwise", "--num_worker", "32",
        ]
        from genpose2.runners.infer import create_genpose2

        self.genpose2 = create_genpose2(
            score_model_path=str(self.genpose2_fpath / self.score_model_fname),
            energy_model_path=str(self.genpose2_fpath / self.energy_model_fname),
            scale_model_path=str(self.genpose2_fpath / self.scale_model_fname),
        )
        total = sum(
            p.numel()
            for agent in (self.genpose2.energy_agent, self.genpose2.score_agent, self.genpose2.scale_agent)
            for p in agent.parameters()
        )
        logger.info(f"genpose2: total params {total}")

    # ── inference ─────────────────────────────────────────────────────────────

    def _predict_single(
        self,
        rgb: torch.Tensor,            # (3, H, W) in [0, 1] or [0, 255]
        depth: torch.Tensor,         # (H, W)
        fo_mask: torch.Tensor,       # (H, W) bool, object instance
        cam_intr4x4: torch.Tensor,   # (4, 4)
        depth_mask: Optional[torch.Tensor] = None,  # (H, W) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run GenPose2 on one frame-object → (cam_tform4x4_obj (4,4), size3d (3,))."""
        from genpose2.runners.infer import InferDataset

        device, dtype = cam_intr4x4.device, cam_intr4x4.dtype
        height, width = rgb.shape[-2:]
        fx = cam_intr4x4[0, 0].cpu().numpy()
        fy = cam_intr4x4[1, 1].cpu().numpy()
        cx = cam_intr4x4[0, 2].cpu().numpy()
        cy = cam_intr4x4[1, 2].cpu().numpy()

        _mask = fo_mask.float().clone()
        if depth_mask is not None:
            _mask = _mask * (depth_mask > 0.999)

        if self.depth_normalize and (_mask > 0).any():
            scale_depth = 1.0 / depth[_mask > 0].median()
        else:
            scale_depth = torch.tensor(1.0, device=device, dtype=depth.dtype)

        _depth = (scale_depth * depth.clone()).to(torch.float)
        _mask = _mask * (_depth > 0.0) * (_depth <= self.depth_max)
        # GenPose2 expects non-object pixels flagged as 255.
        _mask[_mask == 0] = 255

        data = InferDataset(
            data={
                "depth": _depth.cpu().numpy(),
                "mask": _mask.cpu().numpy(),
                "color": rgb.permute(1, 2, 0).cpu().numpy(),
                "meta": {"camera": {"intrinsics": {
                    "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                    "width": width, "height": height,
                }}},
            },
            img_size=self.img_size,
            device=device,
            n_pts=self.n_pts,
        )

        try:
            pose, length = self.genpose2.inference(
                data=data, prev_pose=self._prev_pose,
                tracking=self.tracking, tracking_T0=self.tracking_T0,
            )
            cam_tform4x4_obj = pose[0][-1].to(dtype=dtype, device=device)
            size3d = length[0][-1].to(dtype=dtype, device=device)
            # undo the depth normalisation applied above
            cam_tform4x4_obj[:3, 3] = cam_tform4x4_obj[:3, 3] / scale_depth
            size3d = size3d / scale_depth
        except Exception:
            logger.warning("genpose2 inference failed; returning fallback pose.")
            cam_tform4x4_obj = torch.eye(4, dtype=dtype, device=device)
            cam_tform4x4_obj[2, 3] = 100.0
            size3d = torch.ones(3, dtype=dtype, device=device)

        return cam_tform4x4_obj, size3d

    @staticmethod
    def _cam_tform4x4_obj_ncds(cam_tform4x4_obj: torch.Tensor, size3d: torch.Tensor) -> torch.Tensor:
        """ncds→cam: cam_tform4x4_obj @ uniform-scale(max(size3d) / obj_size_ncds)."""
        scale = size3d.max() / _OBJ_SIZE_NCDS
        scale4x4 = torch.eye(4, dtype=cam_tform4x4_obj.dtype, device=cam_tform4x4_obj.device)
        scale4x4[0, 0] = scale4x4[1, 1] = scale4x4[2, 2] = scale
        return cam_tform4x4_obj @ scale4x4

    def forward(self, frames_gt, frames_pred=None):
        frames = frames_pred if frames_pred is not None else frames_gt

        if self.genpose2 is None:
            raise RuntimeError(
                "GenPose2 checkpoints not loaded; pass `genpose2_fpath` in the model config.",
            )

        cam_intr4x4 = frames.cam_intr4x4
        rgb, depth, fo_mask = frames.rgb, frames.depth, frames.fo_mask
        depth_mask = getattr(frames, "depth_mask", None)

        batched = cam_intr4x4.ndim == 3  # (B, 4, 4) vs (4, 4)
        if not batched:
            cam_tform4x4_obj, size3d = self._predict_single(
                rgb, depth, fo_mask, cam_intr4x4, depth_mask,
            )
            cam_tform4x4_obj_ncds = self._cam_tform4x4_obj_ncds(cam_tform4x4_obj, size3d)
        else:
            poses, ncds, sizes = [], [], []
            for b in range(cam_intr4x4.shape[0]):
                _pose, _size = self._predict_single(
                    rgb[b], depth[b], fo_mask[b], cam_intr4x4[b],
                    None if depth_mask is None else depth_mask[b],
                )
                poses.append(_pose)
                sizes.append(_size)
                ncds.append(self._cam_tform4x4_obj_ncds(_pose, _size))
            cam_tform4x4_obj = torch.stack(poses, dim=0)
            cam_tform4x4_obj_ncds = torch.stack(ncds, dim=0)
            size3d = torch.stack(sizes, dim=0)

        frames.cam_tform4x4_obj = cam_tform4x4_obj
        frames.cam_tform4x4_obj_ncds = cam_tform4x4_obj_ncds
        frames.obj_size3d = size3d

        return frames_gt, frames
