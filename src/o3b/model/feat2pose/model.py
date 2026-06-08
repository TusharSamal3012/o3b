import logging
logger = logging.getLogger(__name__)

from o3b.cv.geometry.transform import rotation_6d_to_matrix, transf4x4_from_rot3x3_and_transl3
from o3b.model.model import OD3D_Model, register_model

@register_model("Feat2Pose")
class Feat2Pose(OD3D_Model):
    def __init__(
        self,
    ):
        super().__init__()
        self.out_dims = [3, 6, 3]
        
    def forward(self, frames_gt, frames_pred=None):

        feat = frames_pred.feat
        out_dim_offset = 0
        
        if self.out_dims[0] == 3: # transl
            poses_transl3d = feat[..., out_dim_offset:self.out_dims[0]].clone()
            
            if frames_pred.obj_tform4x4_obj_nocs_0c is not None:
                from o3b.cv.geometry.transform import transf3d_broadcast, tform4x4_broadcast, inv_tform4x4
                poses_transl3d = transf3d_broadcast(pts3d=poses_transl3d, transf4x4=frames_pred.obj_tform4x4_obj_nocs_0c)

            out_dim_offset += 3
        else:
            poses_transl3d = None
            msg = f"PoseMLP out_dims[0] != 3"
            raise ValueError(msg)
        
        if self.out_dims[1] == 6: # rot
            poses_rot6d = feat[..., out_dim_offset:out_dim_offset + self.out_dims[1]].clone()
            out_dim_offset += 6

            cam_rot3x3_obj = rotation_6d_to_matrix(poses_rot6d)
            
            
            if frames_pred.obj_tform4x4_obj_nocs_0c is not None:
                from o3b.cv.geometry.transform import transf3d_broadcast, rot3x3, inv_tform4x4, rem_scale_tform4x4
                cam_rot3x3_obj = rot3x3(rem_scale_tform4x4(frames_pred.obj_tform4x4_obj_nocs_0c)[..., :3, :3], cam_rot3x3_obj)
                
            frames_pred.cam_tform4x4_obj = transf4x4_from_rot3x3_and_transl3(rot3x3=cam_rot3x3_obj, transl3=poses_transl3d)
        else:
            msg = f"PoseMLP out_dims[1] != 6"
            raise ValueError(msg)
        
        if self.out_dims[2] == 3:
            frames_pred.obj_size3d = feat[..., out_dim_offset:out_dim_offset + self.out_dims[2]].clone()
        
            if frames_pred.obj_tform4x4_obj_nocs_0c is not None:
                from o3b.cv.geometry.transform import rot3d_broadcast, get_scale1d_tform4x4
                scale1d_to_obj_from_obj_nocs0 = get_scale1d_tform4x4(frames_pred.obj_tform4x4_obj_nocs_0c, keepdim=False)
                frames_pred.obj_size3d = frames_pred.obj_size3d * scale1d_to_obj_from_obj_nocs0[:, None]

        else:
            msg = f"PoseMLP out_dims[2] != 3"
            raise ValueError(msg)

        if frames_pred.obj_tform4x4_obj_nocs_0c is not None:
            frames_pred.obj_tform4x4_obj_nocs_0c = None

        return frames_gt, frames_pred
