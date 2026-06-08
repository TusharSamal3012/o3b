
# for densematcher
# cd third_party
# pip install igraph pywavefront
# pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
# pip install --no-build-isolation 'git+https://github.com/facebookresearch/Mask2Former.git' # doesnt work
# pip install setuptools-rust
# pip install --no-build-isolation 'git+https://github.com/NVlabs/ODISE.git' 
# File "<string>", line 2, in <module>
#      ModuleNotFoundError: No module named 'setuptools_rust'
# # doesnt work due to missing setuptools_rust but is required for featup
# pip install --no-build-isolation --no-cache-dir ./featup
# pip install --no-cache-dir ./dift
# git clone https://github.com/nvlabs/odise


# for diff3f
# pip install diffusers transformers accelerate
# pip install pytorch3d==0.7.8+pt2.6.0cu124 --extra-index-url https://miropsota.github.io/torch_packages_builder




# pip install libigl
# pip install pythreejs
# pip install git+https://github.com/skoch9/meshplot.git

# cd third_party/pyFM 
# pip install .
# cd ../..


# pip install diffusers

# diffusers # fixed positionnet and other imports, now works with latest diffusers version
# loading an older diffusers conflicts with huggingface-hu
# huggingface-hub

# doesnt work iwth hugginface 
# pip install diffusers==0.25.1 # for positionnet / dift
