import numpy as np
import torch


#def compute_surface_map(path_1, path_2, c1, c2, source_index=None, target_index=None, use_wks=False, device=torch.device("cuda:0"), n_descr=2048):
    #mesh1 = TriMesh(path_1)
    #mesh2 = TriMesh(path_2)
    
def compute_surface_map(verts1, verts2, faces1, faces2, c1, c2, source_index=None, target_index=None, use_wks=False, 
                        device=torch.device("cuda:0"), n_descr=2048, fpath_source_mesh=None, fpath_target_mesh=None):
    from pyFM.mesh import TriMesh
    from pyFM.functional import FunctionalMapping

    if fpath_source_mesh is not None and fpath_target_mesh is not None:
        mesh1 = TriMesh(str(fpath_source_mesh))
        mesh2 = TriMesh(str(fpath_target_mesh))
    else:
        mesh1 = TriMesh(verts1.detach().cpu().numpy(), faces1.detach().cpu().numpy())
        mesh2 = TriMesh(verts2.detach().cpu().numpy(), faces2.detach().cpu().numpy())

    print("mesh1", mesh1.vertlist.shape)
    print("mesh2", mesh2.vertlist.shape)
    if not use_wks:
        process_params = {
        'n_ev': (50, 50),  # Number of eigenvalues on source and Target
        'n_descr': n_descr,
        'landmarks': None,
        'descr1': c1.detach().cpu().numpy(),
        'descr2': c2.detach().cpu().numpy(),
        'subsample_step': 0,
        # 'descr_type': 'neural',  # neural or WKS or HKS
        }
    else:
        process_params = {
        'n_ev': (50, 50),  # Number of eigenvalues on source and Target
        'n_descr': n_descr,
        'landmarks': None,
        'subsample_step': 1,  # In order not to use too many descriptors
        'descr_type': 'WKS',  # WKS or HKS
        'subsample_step': 0
        }
    model = FunctionalMapping(mesh1, mesh2)
    # NOTE: problem if mesh 1 has more vertices than mesh 2, as the mapping is from mesh 1 to mesh 2
    model.preprocess(**process_params,verbose=True)
    fit_params = {
    'w_descr': 1e0,
    'w_lap': 1e-2,
    'w_dcomm': 1e-1,
    'w_orient': 0
    }
    model.fit(**fit_params, verbose=True)
    p = model.get_p2p(n_jobs=1)
    if source_index is not None:
        p = p[source_index]
    p = torch.from_numpy(mesh1.vertices[p]).to(device)
    if target_index is not None:
        vertices = torch.from_numpy(mesh1.vertices[target_index]).to(device)
        p = torch.cdist(p, vertices)
        p = torch.argmin(p, dim=2)[0]
    else:
        vertices = torch.from_numpy(mesh1.vertices).to(device)
        p = torch.cdist(p, vertices)
        p = torch.argmin(p, dim=1)
    return p
