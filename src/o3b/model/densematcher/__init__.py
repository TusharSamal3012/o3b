# DenseMatcher model wrapper for o3b.
#
# Required packages (install from the bundled third_party sources):
#   cd third_party/o3b/src/o3b/model/densematcher/third_party
#   pip install --no-build-isolation --no-cache-dir ./featup
#   pip install --no-cache-dir ./dift
#   pip install --no-build-isolation --no-cache-dir ./ODISE
#   pip install --no-build-isolation --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'
#   export CUDA_HOME="/usr/local/cuda-12.4" & pip install --no-build-isolation --no-cache-dir ./Mask2Former
#   pip install --no-cache-dir ./stablediffusion
#   pip install  xformers  # not required
#   pip install --no-cache-dir robust-laplacian
#   pip install --no-cache-dir potpourri3d

# Checkpoints (set paths in configs/model/dm.yaml):
#   pretrained_upsampler_path: featup_imsize=384_channelnorm=False_unitnorm=False_rotinv=True/final.ckpt
#   aggre_net_weights_folder:  checkpoints/SDDINO_weights
