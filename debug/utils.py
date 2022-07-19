import json
import os
import pickle
import time
from copy import deepcopy
from functools import partial
from typing import Union, List

import numpy as np
import pytorch3d.transforms as pt
import torch
import trimesh.transformations as tr
from PIL import Image
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
)
from pytorch3d.structures import join_meshes_as_scene

from bricks.brick_info import BricksPC, Brick, HBrick, VBrick, CBrick, add_cbrick_to_bricks_pc, dict_to_cbrick
from common import DEBUG_DIR
from data_generation.utils import bricks2meshes, brick2p3dmesh, render_lego_scene, get_brick_masks, highlight_edge, \
    transform_mesh
from lego.utils.camera_utils import get_cameras
from lego.utils.data_generation_utils import flatten_nested_list, unflatten_nested_list


# import pytorch3d.renderer.cameras as prc


def sample_colors(n, allow_repeats):
    colors = []

    for j in range(n):
        while True:
            rgb = [np.random.randint(0, 256) for _ in range(3)]
            if allow_repeats or rgb not in colors:
                break
        colors.append(rgb)

    return colors


def get_cam_params(mesh):
    # elev, azim = 40, -35
    elev, azim = 30, 225
    # R, T = look_at_view_transform(dist=2000, elev=elev, azim=azim, at=((0, 0, 0),))
    # cameras = FoVOrthographicCameras(device=mesh.device, R=R, T=T,
    #                                  scale_xyz=[(0.0024, 0.0024, 0.0024)])
    cameras = get_cameras(elev=elev, azim=azim)
    bbox = mesh.get_bounding_boxes()[0]
    center = (bbox[:, 1] + bbox[:, 0]) / 2
    bbox_oct = torch.cartesian_prod(bbox[0], bbox[1], bbox[2])
    screen_points = cameras.get_full_projection_transform().transform_points(bbox_oct)[:, :2]
    min_screen_points = screen_points.min(dim=0).values
    max_screen_points = screen_points.max(dim=0).values
    size_screen_points = max_screen_points - min_screen_points
    margin = 0.05
    scale_screen_points = (2 - 2 * margin) / size_screen_points
    return scale_screen_points.min().item(), center


def visualize_bricks(bricks: List[Brick], highlight=False, adjust_camera=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def expand_cbrick(brick):
        if isinstance(brick, Brick):
            return brick
        else:
            assert isinstance(brick, CBrick)
            return brick.bricks_raw

    bricks_template = list(map(expand_cbrick, bricks))
    flat_bricks = flatten_nested_list(bricks_template, bricks_template)

    colors = sample_colors(len(flat_bricks), allow_repeats=True)
    mask_colors = sample_colors(len(bricks), allow_repeats=False)

    colors = unflatten_nested_list(colors, bricks_template)

    brick_meshes = bricks2meshes(bricks, colors)

    if adjust_camera:
        obj_scale, obj_center = get_cam_params(join_meshes_as_scene(brick_meshes))
        transform = pt.Transform3d().translate(*(-obj_center)).scale(obj_scale).cuda()
        brick_meshes = list(map(partial(transform_mesh, transform=transform), brick_meshes))

    R, T = look_at_view_transform(dist=2000, elev=30, azim=225, at=((0, 0, 0),))
    cameras = FoVOrthographicCameras(device=device, R=R, T=T, scale_xyz=[[0.0024, 0.0024, 0.0024]])
    # cameras = prc.OpenGLOrthographicCameras(device=device, R=R, T=T, scale_xyz=[(0.0024,) * 3, ])

    mesh = join_meshes_as_scene(brick_meshes)

    image, depth_map = render_lego_scene(mesh, cameras)
    image[:, :, :, 3][image[:, :, :, 3] == 0] = 0
    image[:, :, :, 3][image[:, :, :, 3] > 0] = 1
    image_pil = Image.fromarray((image[0, :, :].detach().cpu().numpy() * 255).astype(np.uint8))
    image_pil = image_pil.resize((512, 512))

    if highlight:
        mask_brick_meshes = [brick2p3dmesh(bricks[i], mask_colors[i]).to(device) for i in range(len(bricks))]
        mask_brick_meshes = list(map(partial(transform_mesh, transform=transform), mask_brick_meshes))
        masks_step, image_shadeless = get_brick_masks(mask_brick_meshes, mask_colors, range(len(bricks)), cameras)
        image_pil = highlight_edge(image_pil, depth_map, image_shadeless)

    return image_pil


def visualize_bricks_pc(bs: BricksPC, highlight=False, adjust_camera=True):
    return visualize_bricks(get_elements(bs.bricks), highlight=highlight, adjust_camera=adjust_camera)


def visualize_brick(b_or_b_type: Union[str, Brick], highlight=False, adjust_camera=True):
    if isinstance(b_or_b_type, Brick):
        return visualize_bricks([b_or_b_type], highlight=highlight, adjust_camera=adjust_camera)

    return visualize_bricks([Brick(b_or_b_type, (0, 0, 0), (1, 0, 0, 0))],
                            highlight=highlight, adjust_camera=adjust_camera)


# helpers

def get_save_path(name, ext, add_timestamp=False, verbose=True):
    if add_timestamp:
        path = os.path.join(DEBUG_DIR, f"{name}_{time.time()}{ext}")
    else:
        # overwrites!
        path = os.path.join(DEBUG_DIR, f"{name}{ext}")

    return path


# BricksPC


def load_bricks_pc_from_dict(d: dict, return_steps=False) -> Union[BricksPC, List[BricksPC]]:
    # ignore object_rotation_quat

    if return_steps:
        bs_steps = []
    bs = BricksPC(np.array(d['grid_size']))

    for i_str, op in d['operations'].items():
        b_step = op['bricks']
        for j in range(len(b_step)):
            if 'canonical_state' in b_step[j]:
                b_state = b_step[j]['canonical_state']
                cls = b_state.pop('cls')
                if cls == 'HBrick':
                    b = HBrick(**b_state)
                    assert bs.add_hbrick(b, b_step[j]['op_type'])
                elif cls == 'VBrick':
                    b = VBrick(**b_state)
                    assert bs.add_vbrick(b, op_type=b_step[j]['op_type'])
                elif cls == 'CBrick':
                    rec_bricks_pc = b_state.pop('bricks_pc')
                    rec_bricks_pc = load_bricks_pc_from_dict(rec_bricks_pc)
                    b = CBrick(rec_bricks_pc, **b_state)
                    assert add_cbrick_to_bricks_pc(bs, b, op_type=b_step[j]['op_type'])
                else:
                    raise NotImplementedError(cls)
            else:
                if not bs.add_brick(b_step[j]['brick_type'], b_step[j]['canonical_position'],
                                    b_step[j]['canonical_rotation'],
                                    b_step[j]['op_type'], canonical=True, verbose=False):
                    print('Cannot add brick at #', i_str)
                    import ipdb;
                    ipdb.set_trace()
                    print()
        if return_steps:
            bs_steps.append(deepcopy(bs))

    if return_steps:
        return bs_steps

    return bs


def save_bricks_pc(bs: BricksPC, add_timestamp=False, as_dict=True, name='bs', info=None):
    ext = '.json' if as_dict else '.pkl'
    path = get_save_path(name, ext, add_timestamp=add_timestamp)

    if not as_dict:
        with open(path, 'wb') as f:
            pickle.dump((bs, info), f)
        return

    with open(path, 'w') as f:
        json.dump((bs.to_dict(), info), f, indent=4)


def load_bricks_pc(path=None, return_info=False):
    if path is None:
        path = os.path.join(DEBUG_DIR, 'bs.json')
    if os.path.splitext(path)[1] == '.pkl':
        with open(path, 'rb') as f:
            ret = pickle.load(f)
            bs, info = ret

    else:
        with open(path, 'r') as f:
            ret = json.load(f)
            if isinstance(ret, tuple):
                bs, info = ret
                bs = load_bricks_pc_from_dict(bs)
            else:
                bs, info = ret, None
                bs = load_bricks_pc_from_dict(bs)

    print('info from saved bricks pc', info)

    if return_info:
        return bs, info

    return bs


def save_bricks_pc_image(bs: BricksPC, add_timestamp=False, highlight=False, name='bs'):

    path = get_save_path(name, '.png', add_timestamp=add_timestamp)
    im = visualize_bricks_pc(bs, highlight)
    im.save(path)


def get_elements(bricks):
    return sum([[b] if isinstance(b, Brick) else b.bricks for b in bricks], [])


# Brick


def save_brick(brick, add_timestamp=False, as_dict=True, name='brick'):
    ext = '.json' if as_dict else '.pkl'
    path = get_save_path(name, ext, add_timestamp=add_timestamp)

    if not as_dict:
        with open(path, 'wb') as f:
            pickle.dump(brick, f)
        return

    with open(path, 'w') as f:
        json.dump(dict(brick_type=brick.brick_type,
                       position=list(map(float, brick.position)),
                       rotation=list(map(float, brick.rotation))), f, indent=4)


def load_brick(path=None):
    if path is None:
        path = os.path.join(DEBUG_DIR, 'brick.json')
    if os.path.splitext(path)[1] == '.pkl':
        with open(path, 'rb') as f:
            return pickle.load(f)

    with open(path, 'r') as f:
        d = json.load(f)
    return Brick(d['brick_type'], d['position'], d['rotation'])


# List[Brick]

def save_bricks_image(bricks: List[Brick], add_timestamp=False, highlight=False, name='bricks'):
    path = get_save_path(name, '.png', add_timestamp=add_timestamp)
    im = visualize_bricks(bricks, highlight)
    im.save(path)


@torch.no_grad()
def render_dict_simple(d, only_final=False, no_check=False):
    '''
    :param d: bricks dict
    :param azims: If given, overwrite dict camera parameters
    :param elevs: If given, overwrite dict camera parameters
    :return:
    '''
    bricks = []

    images = []
    occs = []
    colors = []
    for i_str, op in d['operations'].items():
        b_step = d['operations'][i_str]['bricks']
        bricks_num = len(b_step)
        for j in range(bricks_num):
            rotation = b_step[j]['brick_transform']['rotation']
            position = b_step[j]['brick_transform']['position']
            rotation_euler = np.round(np.array(tr.euler_from_quaternion(rotation)) / np.pi * 180).astype(int)
            rotation_euler = list(map(int, rotation_euler))
            b_step[j]['brick_transform']['rotation_euler'] = rotation_euler
            colors.append(tuple(b_step[j]['color']))
            if 'brick_type' in b_step[j]:
                brick_type = b_step[j]['brick_type']
                bricks.append(Brick(brick_type, position, rotation))
            else:
                cbrick = dict_to_cbrick(b_step[j], no_check=no_check)
                bricks.append(cbrick)

    mask_colors = []
    for j in range(len(bricks)):
        while True:
            r, g, b = [np.random.randint(0, 256) for _ in range(3)]
            if not (r, g, b) in mask_colors:
                break
        mask_colors.append((r, g, b))

    brick_meshes = bricks2meshes(bricks, colors)
    mask_brick_meshes = bricks2meshes(bricks, mask_colors)

    obj_scale, obj_center = d['obj_scale'], np.array(d['obj_center'])

    transform = pt.Transform3d().translate(*(-obj_center)).scale(obj_scale).cuda()
    # transform = pt.Transform3d().scale(obj_scale).cuda()
    for i in range(len(brick_meshes)):
        brick_mesh, mask_brick_mesh = brick_meshes[i], mask_brick_meshes[i]
        brick_meshes[i] = transform_mesh(brick_mesh, transform)
        mask_brick_meshes[i] = transform_mesh(mask_brick_mesh, transform)

    scale_xyz = np.array([0.0024] * 3)

    def index_list(l, idxs):
        return [l[i] for i in sorted(list(idxs))]

    brick_ct = 0

    for i_str, op in d['operations'].items():

        b_step = d['operations'][i_str]['bricks']
        step_brick_ct = len(b_step)
        cur_brick_idxs = list(range(brick_ct + step_brick_ct))
        step_idxs = [brick_ct + j for j in range(step_brick_ct)]
        brick_ct += step_brick_ct
        if only_final and int(i_str) != len(d['operations']) - 1:
            continue

        azim = op['view_direction'][0]
        elev = op['view_direction'][1]

        cur_mask_brick_meshes = index_list(mask_brick_meshes, cur_brick_idxs)
        cur_mask_colors = index_list(mask_colors, cur_brick_idxs)

        cur_brick_meshes = index_list(brick_meshes, cur_brick_idxs)
        mesh = join_meshes_as_scene(cur_brick_meshes)
        R, T = look_at_view_transform(dist=2000, elev=elev, azim=azim, at=((0, 0, 0),))
        T = T.cuda()
        cameras = FoVOrthographicCameras(device=brick_meshes[0].device, R=R, T=T,
                                         scale_xyz=[scale_xyz])
        op['camera'] = {
            'T': list(map(float, T[0].detach().cpu().numpy())),
            'R': list(map(float, tr.quaternion_from_matrix(R[0].detach().cpu().numpy())))
        }

        masks_step, image_shadeless = get_brick_masks(cur_mask_brick_meshes, cur_mask_colors, step_idxs, cameras)

        image, depth_map = render_lego_scene(mesh, cameras)
        image[:, :, :, 3][image[:, :, :, 3] == 0] = 0
        image[:, :, :, 3][image[:, :, :, 3] > 0] = 1
        image_pil = Image.fromarray((image[0, :, :].detach().cpu().numpy() * 255).astype(np.uint8))
        image_pil = image_pil.resize((512, 512))
        image_pil = highlight_edge(image_pil, depth_map, image_shadeless)
        images.append(image_pil)

    if only_final:
        return images[0]
    else:
        return images
