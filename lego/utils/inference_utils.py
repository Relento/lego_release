from typing import Tuple, Union, List

import numpy as np
import torch
import trimesh.transformations as tr
from scipy.optimize import linear_sum_assignment

from bricks.brick_info import (
    get_connection_offset, get_brick_class, add_cbrick_to_bricks_pc, BricksPC, CBrick, get_cbrick_keypoint,
    transform_points_round, Brick
)
from util.util import transform_points_screen


def dist_matching_greedy(pred, target):
    dists_all = ((target[:, None, :] - pred[None, :, :]) ** 2).sum(dim=-1)
    gt_candidates = list(range(target.shape[0]))
    pred_candidates = list(range(pred.shape[0]))
    idxs = [-1] * len(gt_candidates)
    for i in range(len(gt_candidates)):
        dists_row, pred_idxs = dists_all[np.meshgrid(gt_candidates, pred_candidates, indexing='ij')].min(dim=1)
        gt_idx_min = dists_row.argmin()
        idxs[gt_candidates[gt_idx_min]] = pred_candidates[pred_idxs[gt_idx_min]]
        gt_candidates.pop(gt_idx_min)
        pred_candidates.pop(pred_idxs[gt_idx_min])
    return idxs, dists_all[list(range(target.shape[0])), idxs].cpu().numpy()


def dist_matching_hungarian(pred, target):
    dists = ((target[:, None, :] - pred[None, :, :]) ** 2).sum(dim=-1).cpu().numpy()
    idxs_target, idxs_pred = linear_sum_assignment(dists)
    return idxs_pred, dists[idxs_target, idxs_pred]


def compute_2d_kp_to_3d_translation(bid: str, rot, op_type, kp: torch.Tensor,
                                    connections: Tuple[torch.Tensor, torch.Tensor],
                                    bricks_pc,
                                    scale, transforms, cameras, cbrick=None, keypoint_policy='brick',
                                    top_brick_ind=-1):
    """

    Args:
        bid:
        rot:
        op_type:
        kp: [2,], unbatched, for a single brick in this step
        connections: [?, 2], unbatched
        bricks_pc (BricksPC): unbatched
        scale:
        transforms:
        cameras:

    Returns:
        trans (np.array):
        candidates: List[np.ndarray]:

    """

    if cbrick is None:
        b_conns_canonical = torch.as_tensor(
            get_connection_offset(bid, rot, op_type),
            dtype=torch.float32)
        pos_kp_offset = torch.as_tensor([0, get_brick_class(bid).get_height(), 0])
    else:
        assert bid.startswith('CBrick')
        b_conns_canonical = torch.as_tensor(
            get_connection_offset(cbrick, rot, op_type),
            dtype=torch.float32)
        pos_kp_offset = get_cbrick_keypoint(cbrick, keypoint_policy, top_ind=top_brick_ind)[0] - cbrick.position
        pos_kp_offset = transform_points_round([pos_kp_offset], tr.quaternion_matrix(rot))[0]
        pos_kp_offset = torch.as_tensor(pos_kp_offset)

    b_conns_canonical[:] -= pos_kp_offset[None]

    # add center point for reference
    b_conns = torch.cat([b_conns_canonical, torch.as_tensor([[0, 0, 0]])], dim=0).cuda()
    b_conns_world = b_conns * scale[None]
    # [N, 2]
    b_conns_screen = transform_points_screen(cameras.get_full_projection_transform(),
                                             transforms.transform_points(b_conns_world[:, None, :].cuda()),
                                             ((512, 512),)).squeeze(dim=1)
    # we only care about the offsets of connection points relative to the center as we are adding them to the detected
    # keypoints
    b_conns_screen[:-1, :] -= b_conns_screen[[-1], :]
    b_conns_screen = b_conns_screen[:-1, :2]
    b_conns_screen = b_conns_screen.to(kp.device)
    b_conns_screen += kp[None, :]
    # [M, 2]
    conns_screen = connections[1][:, :]
    dists = ((conns_screen[:, None, :2] - b_conns_screen[None, :, :]) ** 2).sum(dim=-1)

    if bricks_pc is None:
        # skip 3d position validity check
        c_inds, b_inds = (dists == dists.min()).nonzero(as_tuple=True)
        # select the index with smallest depth
        ind = conns_screen[c_inds][:, 2].argmin()
        c_ind, b_ind = c_inds[ind], b_inds[ind]
        return connections[0][c_ind] - b_conns[b_ind].cpu().numpy(), []

    tol = 2.0
    thres = dists.min() + tol
    c_inds, b_inds = (dists <= thres).nonzero(as_tuple=True)
    candidates = []
    candidates_depth = {}

    if cbrick is not None:
        cbrick_bs = BricksPC.from_dict(cbrick._state_dict['bricks_pc'], no_check=True)

    for k in range(c_inds.shape[0]):
        c_ind, b_ind = c_inds[k], b_inds[k]
        trans = tuple(connections[0][c_ind].cpu().numpy() - b_conns[b_ind].cpu().tolist())
        position = np.array(trans) - pos_kp_offset.numpy()
        if cbrick is None:
            is_valid = bricks_pc.add_brick(bid, position,
                                           rot, op_type,
                                           only_check=True,
                                           canonical=False)
        else:
            cbrick_this = CBrick(cbrick_bs, position, rot)
            is_valid = add_cbrick_to_bricks_pc(bricks_pc, cbrick_this, op_type, only_check=True)
        if is_valid:
            candidates.append(trans)
            candidates_depth[trans] = conns_screen[c_ind][2]

    if not candidates:
        return np.array(trans), candidates, pos_kp_offset.numpy()
    else:
        from collections import Counter
        candidates_ct = Counter(candidates)
        candidates.sort(key=lambda x: -candidates_ct[x] + candidates_depth[x])
        return np.array(candidates[0]), candidates, pos_kp_offset.numpy()


from functools import partial
from data_generation.utils import brick2p3dmesh, transform_mesh, get_brick_masks
from pytorch3d.structures import join_meshes_as_scene
from lego.utils.camera_utils import get_cameras


class RenderBricks:
    BASE_MASK_COLOR = (255, 0, 0)
    CANDIDATE_MASK_COLOR = (0, 255, 0)

    def __init__(self, bricks_pc: BricksPC, obj_transform):
        '''
        For conveniently render the scenario where we have a base bricks_pc and
        want to render multiple candidate bricks.
        :param bricks_pc:
        :param obj_transform: assume fixed across the whole episode.
        '''
        self.bricks_pc = bricks_pc
        self.obj_transform = obj_transform
        self.candidate_brick = None
        self.candidate_op_type = 0
        self.candidate_mesh = None
        self._set_bricks_pc_mesh()

    def _get_mesh_from_bricks(self, bricks: List[Brick], color: Tuple[int, int, int]):
        # We don't care about individual bricks in the original bricks_pc so assign them the same color.
        mask_colors = [color] * len(bricks)

        brick2p3dmesh_to_cuda = partial(brick2p3dmesh, cuda=True)
        element_meshes = list(map(
            lambda p: brick2p3dmesh_to_cuda(p[0], p[1])
            , zip(bricks, mask_colors)))
        transform_mesh_fn = partial(transform_mesh, transform=self.obj_transform)
        element_meshes = list(map(transform_mesh_fn, element_meshes))
        return join_meshes_as_scene(element_meshes).cuda()

    def _set_bricks_pc_mesh(self):
        elements = []
        for i, b in enumerate(self.bricks_pc.bricks):
            if isinstance(b, Brick) or isinstance(b, CBrick):
                if isinstance(b, Brick):
                    elements.append(b)
                else:
                    elements.extend(b.bricks)

        self.bricks_pc_mesh = self._get_mesh_from_bricks(elements, RenderBricks.BASE_MASK_COLOR)

    def update_candidate(self, brick: Union[Brick, CBrick], op_type=0):
        elements = []
        if isinstance(brick, Brick):
            if not self.bricks_pc.add_brick(brick.brick_type, brick.position, brick.rotation, op_type=op_type,
                                            only_check=True):
                return False
            self.candidate_brick = brick
            self.candidate_op_type = op_type
            elements.append(brick)
        else:
            assert isinstance(brick, CBrick)
            if not add_cbrick_to_bricks_pc(self.bricks_pc, brick, op_type, only_check=True):
                return False
            self.candidate_brick = brick
            self.candidate_op_type = op_type
            elements.extend(brick.bricks)

        self.candidate_mesh = self._get_mesh_from_bricks(elements, RenderBricks.CANDIDATE_MASK_COLOR)
        return True

    def get_candidate_mask(self, azim=None, elev=None, cameras=None):
        if azim is not None:
            assert elev is not None and cameras is None
            cameras = get_cameras(azim, elev)
        else:
            assert cameras is not None
        assert self.candidate_mesh is not None
        meshes = [self.bricks_pc_mesh, self.candidate_mesh]
        mask_colors = [RenderBricks.BASE_MASK_COLOR, RenderBricks.CANDIDATE_MASK_COLOR]
        masks, _ = get_brick_masks(meshes, mask_colors, [1], cameras, render_size=512)
        return masks[0]


def compute_mask_2d_kp_to_3d_translation(bid: str, op_type, kp: torch.Tensor,
                                         connections: Tuple[torch.Tensor, torch.Tensor],
                                         bricks_pc,
                                         scale, transforms, cameras, rot=None, cbrick=None):
    pass


def compute_candidates_iou(bricks: List[Brick], bricks_pc, transform, cameras, mask_ref):
    rb = RenderBricks(bricks_pc, transform)
    iou_list = []

    def compute_iou(mask_a, mask_b):
        mask_intersect = np.minimum(mask_a, mask_b).sum()
        mask_union = np.maximum(mask_a, mask_b).sum()
        return mask_intersect / mask_union

    for brick in bricks:
        if not rb.update_candidate(brick):
            iou_list.append(-1)
            continue
        mask = rb.get_candidate_mask(cameras=cameras)
        iou_list.append(compute_iou(mask_ref, mask))

    return iou_list


def compute_xz_to_3d_translation(bid: str, rot, op_type, xs, zs, bricks_pc: BricksPC, xz_type='pos', cbrick=None,
                                 keypoint_policy='brick',
                                 top_brick_ind=-1):
    '''
    :param bid:
    :param rot:
    :param op_type:
    :param xs: list
    :param zs: list
    :param xz_type: 'pos' or 'top_center', showing whether x and z correspond to pos or top_center
    :param cbrick:
    :param keypoint_policy:
    :param top_brick_ind:
    :return:
    '''

    if xz_type == 'pos':
        pos_kp_offset = np.array([0, 0, 0])
    else:
        if cbrick is None:
            pos_kp_offset = np.array([0, get_brick_class(bid).get_height(), 0])
        else:
            assert bid.startswith('CBrick')
            pos_kp_offset = get_cbrick_keypoint(cbrick, keypoint_policy, top_ind=top_brick_ind)[0] - cbrick.position
            pos_kp_offset = transform_points_round([pos_kp_offset], tr.quaternion_matrix(rot))[0]
            pos_kp_offset = np.array(pos_kp_offset)

    if cbrick is not None:
        cbrick_bs = BricksPC.from_dict(cbrick._state_dict['bricks_pc'])

    xs_pos = list(np.array(xs) - pos_kp_offset[0])
    zs_pos = list(np.array(zs) - pos_kp_offset[2])

    candidates = []
    if op_type == 2:
        positions = [[x_pos, 0, z_pos] for x_pos, z_pos in zip(xs_pos, zs_pos)]
        for position in positions:
            if cbrick is None:
                is_valid = bricks_pc.add_brick(bid, position,
                                               rot, op_type,
                                               only_check=True,
                                               canonical=False)
            else:
                cbrick_this = CBrick(cbrick_bs, position, rot)
                is_valid = add_cbrick_to_bricks_pc(bricks_pc, cbrick_this, op_type, only_check=True)
            if is_valid:
                trans = position + pos_kp_offset
                candidates.append(trans)
    else:
        position_ys = list(set(map(lambda x: x[1], bricks_pc.get_stud_positions())))
        position_ys.sort(reverse=True)
        assert op_type == 0
        xzs_pos = list(zip(xs_pos, zs_pos))
        for (x, z) in xzs_pos:
            for y in position_ys:
                position = (x, y, z)
                if cbrick is None:
                    is_valid = bricks_pc.add_brick(bid, position,
                                                   rot, op_type,
                                                   only_check=True,
                                                   canonical=False)
                else:
                    cbrick_this = CBrick(cbrick_bs, position, rot)
                    is_valid = add_cbrick_to_bricks_pc(bricks_pc, cbrick_this, op_type, only_check=True)
                if is_valid:
                    candidates.append(position + pos_kp_offset)

    return candidates, pos_kp_offset


def search_valid_3d_translation(bid: str, rot, op_type, trans_list, bricks_pc: BricksPC, trans_type='pos', cbrick=None,
                                keypoint_policy='brick',
                                top_brick_ind=-1):
    '''
    :param bid:
    :param rot:
    :param op_type:
    :param xs: list
    :param zs: list
    :param xz_type: 'pos' or 'top_center', showing whether x and z correspond to pos or top_center
    :param cbrick:
    :param keypoint_policy:
    :param top_brick_ind:
    :return:
    '''

    if trans_type == 'pos':
        pos_kp_offset = np.array([0, 0, 0])
    else:
        if cbrick is None:
            pos_kp_offset = np.array([0, get_brick_class(bid).get_height(), 0])
        else:
            assert bid.startswith('CBrick')
            pos_kp_offset = get_cbrick_keypoint(cbrick, keypoint_policy, top_ind=top_brick_ind)[0] - cbrick.position
            pos_kp_offset = transform_points_round([pos_kp_offset], tr.quaternion_matrix(rot))[0]
            pos_kp_offset = np.array(pos_kp_offset)

    if cbrick is not None:
        cbrick_bs = BricksPC.from_dict(cbrick._state_dict['bricks_pc'], no_check=True)

    positions = list(np.array(trans_list) - pos_kp_offset[None])

    candidates = []
    if op_type == 2:
        positions_new = [[x_pos, 0, z_pos] for x_pos, _, z_pos in positions]
        for position in positions_new:
            if cbrick is None:
                is_valid = bricks_pc.add_brick(bid, position,
                                               rot, op_type,
                                               only_check=True,
                                               canonical=False)
            else:
                cbrick_this = CBrick(cbrick_bs, position, rot)
                is_valid = add_cbrick_to_bricks_pc(bricks_pc, cbrick_this, op_type, only_check=True)
            if is_valid:
                trans = position + pos_kp_offset
                candidates.append(trans)
                break
    else:
        for position in positions:
            if cbrick is None:
                is_valid = bricks_pc.add_brick(bid, position,
                                               rot, op_type,
                                               only_check=True,
                                               canonical=False)
            else:
                cbrick_this = CBrick(cbrick_bs, position, rot)
                is_valid = add_cbrick_to_bricks_pc(bricks_pc, cbrick_this, op_type, only_check=True)
            if is_valid:
                candidates.append(position + pos_kp_offset)
                break

    return candidates, pos_kp_offset


def recompute_conns(bs, op_type, transforms, cameras, scale):
    if op_type == 2:
        # y positions must be 0
        conns = ()
    else:
        if op_type == 0:
            connections = bs.get_stud_positions()
        else:
            connections = bs.get_astud_positions()
        connections = torch.as_tensor(list(connections), dtype=torch.float32).cuda()
        conns_world = connections * scale[None]
        conns_world = transforms.transform_points(conns_world[:, None, :].cuda())
        conns_screen = transform_points_screen(cameras.get_full_projection_transform(),
                                               conns_world,
                                               ((512, 512),)).squeeze(dim=1)
        conns = connections.detach(), conns_screen.detach()

    return conns
