import os
import numpy as np
import open3d as o3d



def get_mesh_bottle(root, bottle_idx=1):
    path_bottle = os.path.join(root, f'{bottle_idx}', 'models', 'model_normalized.obj')
    mesh_bottle = o3d.io.read_triangle_mesh(path_bottle)
    R = mesh_bottle.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    mesh_bottle.rotate(R, center=(0, 0, 0))
    mesh_bottle.translate([0, 0, 0.])

    with open(os.path.join(root, f'{bottle_idx}', 'spec.txt'), 'rb') as f:
        spec = f.readlines()
    bottle_height = float(spec[0].decode('utf-8').split(':')[1])
    bottle_width = float(spec[1].decode('utf-8').split(':')[1])
    
    # bottle_rescalling_translation
    vertices_bottle_numpy = np.asarray(mesh_bottle.vertices) # n * 3 
    bottle_width_current = vertices_bottle_numpy[:, 0].max() - vertices_bottle_numpy[:, 0].min()
    bottle_height_current = vertices_bottle_numpy[:, 2].max() - vertices_bottle_numpy[:, 2].min()
    vertices_bottle_numpy[:, :2] *= bottle_width/bottle_width_current
    vertices_bottle_numpy[:, 2] *= bottle_height/bottle_height_current
    min_x_bottle = vertices_bottle_numpy[:, 0].min()
    min_z_bottle = vertices_bottle_numpy[:, 2].min()
    mesh_bottle.translate([(-bottle_width/2 - min_x_bottle), 0, -(min_z_bottle)])
    return mesh_bottle

def get_mesh_mug(root, mug_idx=1):
    path_mug = os.path.join(root, f'{mug_idx}', 'models', 'model_normalized.obj')
    mesh_mug = o3d.io.read_triangle_mesh(path_mug)
    R = mesh_mug.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    mesh_mug.rotate(R, center=(0, 0, 0))
    
    with open(os.path.join(root, f'{mug_idx}', 'spec.txt'), 'rb') as f:
        spec = f.readlines()
    mug_height = float(spec[0].decode('utf-8').split(':')[1])
    mug_width_outer = float(spec[2].decode('utf-8').split(':')[1])
    
    # mug_rescalling_translation
    vertices_mug_numpy = np.asarray(mesh_mug.vertices) # n * 3
    mug_width_current = vertices_mug_numpy[:, 0].max() - vertices_mug_numpy[:, 0].min()
    mug_height_current = vertices_mug_numpy[:, 2].max() - vertices_mug_numpy[:, 2].min()
    vertices_mug_numpy[:, :2] *= mug_width_outer/mug_width_current
    vertices_mug_numpy[:, 2] *= mug_height/mug_height_current
    min_y_mug = vertices_mug_numpy[:, 1].min()
    min_z_mug = vertices_mug_numpy[:, 2].min()
    mesh_mug.translate([0, (-mug_width_outer/2 - min_y_mug), -min_z_mug])
    return mesh_mug