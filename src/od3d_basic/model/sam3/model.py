import logging
logger = logging.getLogger(__name__)

#from omegaconf import DictConfig
#import torch
#from od3d.models.backbones.backbone import OD3D_Backbone
#from segment_anything import SamPredictor, sam_model_registry
#from pathlib import Path
#from od3d_basic.io import download
#from od3d_basic.data.batch_datatypes import OD3D_ModelData


import os
from od3d_basic.model.model import OD3D_Model, register_model

import matplotlib.pyplot as plt
import numpy as np
import torch

# import sam3
# from PIL import Image
# from sam3 import build_sam3_image_model
# from sam3.model.box_ops import box_xywh_to_cxcywh
# from sam3.model.sam3_image_processor import Sam3Processor
# from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results
# sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
# import torch
# # turn on tfloat32 for Ampere GPUs
# # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# # use bfloat16 for the entire notebook
# torch.autocast("cuda", dtype=torch.bfloat16).__enter__()



@register_model("SAM3")
class SAM3(OD3D_Model):
    def __init__(
        self,
    ):
        super().__init__()

        #super().__init__(config=config)

        # device = "cuda"
        # if not Path(config.sam_checkpoint).exists():
        #     logger.info(
        #         "SAM checkpoint not found, downloading from the official source",
        #     )
        #     download(
        #         url=f"https://dl.fbaipublicfiles.com/segment_anything/{Path(config.sam_checkpoint).name}",
        #         fpath=Path(config.sam_checkpoint),
        #     )

        # sam = sam_model_registry[config.model_type](checkpoint=config.sam_checkpoint)
        # sam.to(device=device)
        # self.predictor = SamPredictor(sam)
        
        #self.out_downsample_scales = []
        #self.downsample_rate = 1
        #self.out_dims = [1]

        from transformers import Sam3Processor, Sam3Model
        import torch
        from PIL import Image
        import requests

        #device = "cuda" if torch.cuda.is_available() else "cpu"
        # hf auth login
        self.model = Sam3Model.from_pretrained("facebook/sam3")
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        # Load image

        #image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
        #image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

        # Segment using text prompt

    def forward(
        self, frames_gt, frames_pred=None,
        #x,
        #points_xy=None,
        #bbox=None,
        #oppress_single_point=False,
        #bbox_from_single_point=False,
        #bbox_from_points=False,
    ):
        device = frames_gt.device
        image = frames_gt.rgb
        
        text = frames_gt.category[0] if frames_gt.category is not None else "object"
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        print(f"Found {len(results['masks'])} objects")

        #masks = model_out.masks
        #scores = model_out.masks_scores

        # from od3d.cv.visual.show import show_imgs
        # show_imgs(mask[None,])

        return  frames_gt, frames_pred
