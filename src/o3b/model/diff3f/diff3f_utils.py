
import gc
from unicodedata import category

import torch
from o3b.model.diff3f.diff3f.diff3f import get_features_per_vertex
from time import time
from o3b.model.diff3f.diff3f.utils import convert_mesh_container_to_torch_mesh, cosine_similarity, double_plot, get_colors, generate_colors
from o3b.model.diff3f.diff3f.dataloaders.mesh_container import MeshContainer
from o3b.model.diff3f.diff3f.diffusion import init_pipe
from o3b.model.diff3f.diff3f.dino import init_dino
from o3b.model.diff3f.diff3f.functional_map import compute_surface_map

#INFO:o3b.model.diff3f.method:pred_target_kpts3d_acc_01: 0.46153846153846156
#INFO:o3b.model.diff3f.method:pred_target_kpts3d_acc_005: 0.23076923076923078
#INFO:o3b.model.diff3f.method:pred_target_kpts3d_acc_01: 0.8461538461538461
#INFO:o3b.model.diff3f.method:pred_target_kpts3d_acc_005: 0.7692307692307693


def get_source_to_target_map(verts_source, verts_target, faces_source, faces_target, feats_source, feats_target, use_functional_map=False, fpath_source_mesh=None, fpath_target_mesh=None, device=torch.device("cuda:0")):
    if not use_functional_map:
        s = cosine_similarity(feats_source, feats_target)
        s = torch.argmax(s, dim=1)
        source_to_target_map = s
    else:
        # center verts_target and verts_source
        target_center = verts_target.max(dim=0).values - verts_target.min(dim=0).values
        source_center = verts_source.max(dim=0).values - verts_source.min(dim=0).values
        verts_target = verts_target - target_center
        verts_source = verts_source - source_center 
        # normalize verts_target and verts_source by their max value to avoid numerical issues in functional map computation
        verts_target = verts_target / target_center.max()
        verts_source = verts_source / source_center.max()
        
        surface_map = compute_surface_map(verts_target, verts_source, 
                                          faces_target, faces_source, 
                                          feats_target, feats_source, 
                                          n_descr=feats_source.shape[-1],
                                          fpath_source_mesh=fpath_source_mesh,
                                          fpath_target_mesh=fpath_target_mesh) # 2048 # 768
        source_to_target_map = surface_map
            
    return source_to_target_map

def compute_features(device, pipe, dino_model, m, prompt, num_views, H, W, tolerance, num_images_per_prompt, use_normal_map, extractor_fn=None):
    mesh = convert_mesh_container_to_torch_mesh(m, device=device, is_tosca=False)
    mesh_vertices = mesh.verts_list()[0]
    features = get_features_per_vertex(
        device=device,
        pipe=pipe,
        dino_model=dino_model,
        mesh=mesh,
        prompt=prompt,
        mesh_vertices=mesh_vertices,
        num_views=num_views,
        H=H,
        W=W,
        tolerance=tolerance,
        num_images_per_prompt=num_images_per_prompt,
        use_normal_map=use_normal_map,
        extractor_fn=extractor_fn,
    )
    return features.cpu()

def get_features_from_mesh(vertices, faces, prompt, device='cuda:0'):

    device = torch.device(device)
    torch.cuda.set_device(device)
    num_views = 100
    H = 512
    W = 512
    num_images_per_prompt = 1
    tolerance = 0.004
    random_seed = 42
    use_normal_map = True

    pipe = init_pipe(device)
    dino_model = init_dino(device)

    source_mesh = MeshContainer(vert=vertices, face=faces)

    features = compute_features(device, pipe, dino_model, source_mesh, prompt, num_views, H, W, tolerance, num_images_per_prompt, use_normal_map)
    
    vertices = torch.from_numpy(source_mesh.vert).to(device).float()
    faces = torch.from_numpy(source_mesh.face).to(device).long()
    features = features.to(device).float()

    del pipe     
    del dino_model
    torch.cuda.empty_cache()

    gc.collect()
    
    return vertices, faces, features