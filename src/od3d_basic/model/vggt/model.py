import logging

from od3d_basic.cv.geometry.transform import transf4x4_from_rot3x3_and_transl3

logger = logging.getLogger(__name__)
from od3d_basic.model.model import OD3D_Model, register_model
import torch
from od3d_basic.cv.visual.resize import resize
try:
    from vggt.models.vggt import VGGT as VGGT_Orig
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri, extri_intri_to_pose_encoding
except ImportError as e:
    logger.warning(f"vggt not available, skipping registration: {e}")
    VGGT_Orig = None
    pose_encoding_to_extri_intri = None
    extri_intri_to_pose_encoding = None

@register_model("VGGT")
class VGGT(OD3D_Model):
    def __init__(self, 
                 img_size=518,  # 512
                 patch_size=14, 
                 embed_dim=1024,
                 enable_camera=True, 
                 enable_point=False, 
                 enable_depth=False, 
                 enable_track=False,
                 pretrained=True,):
        super().__init__()
        
        self.img_size = img_size

        if pretrained:
            self.model = VGGT_Orig.from_pretrained(
                "facebook/VGGT-1B",
                img_size=img_size, 
                patch_size=patch_size, 
                embed_dim=embed_dim,
                enable_camera=enable_camera,
                enable_point=enable_point,
                enable_depth=enable_depth,
                enable_track=enable_track
            )
        else:
            self.model = VGGT_Orig(
                img_size=img_size, 
                patch_size=patch_size, 
                embed_dim=embed_dim,
                enable_camera=enable_camera,
                enable_point=enable_point,
                enable_depth=enable_depth,
                enable_track=enable_track
            )


    def forward(self, frames_gt, frames_pred=None):
        if frames_pred is None:
            frames_pred = frames_gt

        x = frames_gt.rgb

        if x.dim() == 3:
            C, H, W = x.shape
            B = 1
        elif x.dim() == 4:
            B, C, H, W = x.shape
        else:
            raise NotImplementedError

        x = resize(x, H_out=self.img_size, W_out=self.img_size)

        res = self.model(x)

        frames_pred.feat = res['pose_enc'].reshape(B, -1,)
        # please note that vggt does iterative 4 times camera estimation, where the last once are more heavly weighted than the previous once.
        # def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        # """
        # Forward pass of the VGGT model.

        # Args:
        #     images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
        #         B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
        #     query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
        #         Shape: [N, 2] or [B, N, 2], where N is the number of query points.
        #         Default: None

        # Returns:
        #     dict: A dictionary containing the following predictions:
        #         - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
        #         - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
        #         - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
        #         - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
        #         - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
        #         - images (torch.Tensor): Original input images, preserved for visualization

        #         If query_points is provided, also includes:
        #         - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
        #         - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
        #         - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        # """        

        extr3x4, intr3x3 = pose_encoding_to_extri_intri(frames_pred.feat[:, None, :], image_size_hw=[self.img_size, self.img_size], pose_encoding_type="absT_quaR_FoV", build_intrinsics=True)


        frames_pred.cam_tform4x4_obj = transf4x4_from_rot3x3_and_transl3(rot3x3=extr3x4[..., :3, :3], transl3=extr3x4[..., :3, 3])[:, 0]


        frames_pred.obj_size3d = torch.ones_like(frames_pred.cam_tform4x4_obj[..., :3, 3])
        
        return frames_gt, frames_pred
    
