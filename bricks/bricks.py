import os

import numpy as np
import trimesh
import trimesh.transformations as tr

from common import ROOT

bricks_dir = os.path.join(ROOT, 'data/new_parts_center')
brick_types = []

scale_d = {'x': 20, 'y': [8, 8], 'z': 20}
origin_ = np.array([0, 0, 0])

brick_canonical_meshes = {}


def get_brick_canonical_mesh(brick_type):
    global brick_canonical_meshes
    if brick_type not in brick_canonical_meshes:
        brick_obj_path = os.path.join(bricks_dir, f'{brick_type}.obj')
        mesh = trimesh.load(brick_obj_path, skip_materials=True, group_material=False)
        brick_canonical_meshes[brick_type] = mesh
    return brick_canonical_meshes[brick_type].copy()


def brick2mesh(brick, color=None):
    mesh = get_brick_canonical_mesh(brick.brick_type)
    if color is not None:
        mesh.visual.vertex_colors = np.array(color)
    position = np.array(brick.position)
    trans = [position[0] * scale_d['x'], position[1] * scale_d['y'][0], position[2] * scale_d['z']]
    T = trimesh.transformations.translation_matrix(trans)
    # T_ = trimesh.transformations.translation_matrix(-origin_) # Move brick to the center of ts left top 1x1 cell
    R = tr.quaternion_matrix(brick.rotation)
    # mesh.apply_transform(T_).apply_transform(R).apply_transform(T)
    mesh.apply_transform(R).apply_transform(T)
    return mesh


def bricks2mesh(bricks, num_bricks=-1, canonical=False):
    if num_bricks < 0:
        num_bricks = num_bricks + len(bricks.bricks) + 1
    mesh = trimesh.util.concatenate(list(map(lambda x: brick2mesh(x), bricks.bricks[:num_bricks])))
    if not canonical:
        transform_matrix = bricks.transform_.matrix.copy()
        transform_matrix[0, 3] *= scale_d['x']
        transform_matrix[1, 3] *= scale_d['y'][0]
        transform_matrix[2, 3] *= scale_d['z']
        mesh.apply_transform(transform_matrix)
    return mesh

# return available stud if the bricks are facing up, else return available astud
def get_brick_valid_positions_brute_force(bricks, brick_type, rotation):
    up_direction = [0, 1 + bricks.grid_size[1] // 2, 0, 1]
    isBottomUp = ((bricks.transform_.matrix @ up_direction)[1]) <= 0
    res = []

    for i in range(bricks.grid_size[0]):
        for j in range(bricks.grid_size[1]):
            for k in range(bricks.grid_size[2]):
                pos = tuple(map(int, np.round(tr.transform_points([[i, j, k]], bricks.transform_.matrix)[0])))
                if bricks.add_brick(brick_type, pos, rotation, 1 if isBottomUp else 0, only_check=True):
                    res.append(pos)
    return set(res)
