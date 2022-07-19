import argparse
import json
import os

import numpy as np
import pytorch3d
import torch
import trimesh.transformations as tr
from pytorch3d.loss import chamfer_distance

from bricks.brick_info import Brick, dict_to_cbrick
from data_generation.utils import (
    bricks2meshes
)
from datasets.legokps_shape_cond_dataset import normalize_euler
from models.utils import Meters


def get_pc_from_mesh(mesh):
    N = 10000
    pc = pytorch3d.ops.sample_points_from_meshes(mesh.cuda(), N)[0]
    pc /= torch.Tensor([65 * 20, 65 * 8, 65 * 20]).cuda()
    return pc


from collections import defaultdict


def get_pc_from_dict(d):
    bricks_step = []
    cbrick_idxs = defaultdict(list)
    cbrick_poses = []
    for i_str, op in d['operations'].items():
        b_step = d['operations'][i_str]['bricks']
        bricks_num = len(b_step)
        bricks_this = []
        for j in range(bricks_num):
            rotation = b_step[j]['brick_transform']['rotation']
            position = b_step[j]['brick_transform']['position']
            rotation_euler = np.round(np.array(tr.euler_from_quaternion(rotation)) / np.pi * 180).astype(int)
            rotation_euler = list(map(int, rotation_euler))
            b_step[j]['brick_transform']['rotation_euler'] = rotation_euler
            if 'brick_type' in b_step[j]:
                brick_type = b_step[j]['brick_type']
                bricks_this.append(Brick(brick_type, position, rotation))
            else:
                cbrick = dict_to_cbrick(b_step[j], no_check=True)
                bricks_this.append(cbrick)
                cbrick_idxs[int(i_str)].append(j)
                cbrick_poses.append((tuple(position), normalize_euler(rotation_euler)))
        bricks_step.append(bricks_this)

    pc_step = []
    for bs in bricks_step:
        pc_step.append(list(map(get_pc_from_mesh, bricks2meshes(bs, [(255, 0, 0)] * len(bs)))))
    return pc_step, cbrick_idxs, cbrick_poses


@torch.no_grad()
def eval_d(pred_d, target_d, compute_final=False):
    pc_pred_list, _, cbrick_poses_pred = get_pc_from_dict(pred_d)
    pc_target_list, cbrick_idxs, cbrick_poses_gt = get_pc_from_dict(target_d)

    meters = Meters()

    if not compute_final:
        for i, (s_pred, s_target) in enumerate(zip(pc_pred_list, pc_target_list)):
            if len(s_pred) > 10:
                continue
            pc_pred = torch.cat(s_pred, dim=0)[None]
            pc_target = torch.cat(s_target, dim=0)[None]
            meters.update('stepwise_cd', chamfer_distance(pc_pred, pc_target)[0])
            if len(cbrick_idxs[i]) > 0:
                for j in cbrick_idxs[i]:
                    meters.update('brickwise_cd_cbrick', chamfer_distance(s_pred[j][None], s_target[j][None])[0])
            pc_pred = torch.stack(s_pred, dim=0)
            pc_target = torch.stack(s_target, dim=0)
            meters.update('brickwise_cd', chamfer_distance(pc_pred, pc_target)[0] * len(s_pred), n=len(s_pred))

        for i, (pose_pred, pose_gt) in enumerate(zip(cbrick_poses_pred, cbrick_poses_gt)):
            if len(s_pred) > 10:
                continue
            correct = int(np.allclose(np.array(pose_pred), np.array(pose_gt)))
            meters.update('pose_acc_cbrick', correct)


    else:
        def flatten(ls):
            res = []
            for l in ls:
                res.extend(l)
            return res

        pc_pred_list_flattened = flatten(pc_pred_list)
        pc_target_list_flattened = flatten(pc_target_list)
        pc_pred = torch.cat(pc_pred_list_flattened, dim=0)[None]
        pc_target = torch.cat(pc_target_list_flattened, dim=0)[None]

        meters.update('setwise_cd', chamfer_distance(pc_pred, pc_target)[0])

    return meters


subm_dependency_all = {
    'classics': {
        1: [0],
        4: [3],
        7: [6],
        12: [9, 10, 11],
        17: [16],
    },
    'architecture': {
        4: [0, 1, 2, 3],
        7: [6],
        15: [8, 9, 10, 11, 12, 13, 14],
    }
}

if __name__ == '__main__':
    meters = Meters()
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str)
    parser.add_argument('--compute_final', action='store_true')
    args = parser.parse_args()

    if 'architecture' in args.json_path:
        subm_dependency = subm_dependency_all['architecture']
    elif 'classics' in args.json_path:
        subm_dependency = subm_dependency_all['classics']
    else:
        subm_dependency = None

    subm_rev_map = {}
    if subm_dependency is not None:
        for k, subms in subm_dependency.items():
            for s in subms:
                subm_rev_map[s] = k

    for l in sorted(os.listdir(args.json_path)):
        json_path = os.path.join(args.json_path, l, 'info.json')
        if not os.path.isfile(json_path):
            continue
        base_path, _ = os.path.split(json_path)
        base_path2, step_id = os.path.split(base_path)
        step_id = int(step_id)
        if args.compute_final and step_id in subm_rev_map.keys():
            continue
        with open(json_path) as f:
            d_pred = json.load(f)
        with open(d_pred['gt_json_path']) as f:
            d_gt = json.load(f)
        meters.merge_from(eval_d(d_pred, d_gt, args.compute_final))

    print('Final result:')
    for k, v in meters.avg_dict().items():
        print(f'{k}:{v}')
    print()
