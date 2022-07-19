import collections
import json
import os
import pickle

import cv2
import h5py as h5
import numpy as np
import torch
import torchvision.transforms.functional as TF
import trimesh.transformations as tr
from PIL import Image
from pycocotools import mask as mask_utils

from bricks.brick_info import (
    get_all_brick_ids,
    get_brick_class,
    get_brick_enc_voxel,
    get_cbrick_enc_voxel,
    dict_to_cbrick,
    add_cbrick_to_bricks_pc,
    get_cbrick_keypoint,
    get_cbrick_rotations
)
from datasets.base_dataset import BaseDataset
from models.heatmap.utils.image import draw_umich_gaussian, draw_msra_gaussian
from .definition import DatasetDefinitionBase, set_global_definition, gdef

rot_list = [
    (0, 0, 0),
    (0, 90, 0),
    (0, -90, 0),
    (180, 0, 180),
    (0, 0, 180),
    (180, 0, 0),
    (180, -90, 0),
    (-180, 90, 0),
]

rot_map = {
    (-180, -90, 0): (180, -90, 0),
    (180, 90, 0): (-180, 90, 0),
    (0, 0, -180): (0, 0, 180),
    (180, -90, 180): (0, -90, 0),
    (180, 90, 180): (0, 90, 0),
    (-180, 0, 0): (180, 0, 0),
    (0, -90, -180): (180, -90, 0),
    (0, -90, 180): (180, -90, 0),
    (-180, 0, -180): (180, 0, 180),
}


# Convert euler angles of possible brick rotations to id
def euler2id(e, bid=None, valid_rots=None, ignore_symmetry=False,
             symmetry_aware_label=False):
    e_new, rots = canonize_euler(e, bid, valid_rots, ignore_symmetry, return_rots=True)
    idx = rot_list.index(e_new)
    if not symmetry_aware_label:
        return idx
    if len(rots) == 1:
        assert idx == 0
        return 0
    elif len(rots) == 2:
        assert idx < 2
        return idx + 1
    else:
        assert len(rots) == 4 and idx < 4
        return idx + 3


def id2euler(idx, symmetry_aware_label=False):
    if symmetry_aware_label:
        if idx < 3 and idx > 0:
            idx = idx - 1
        elif idx >= 3:
            assert idx <= 6
            idx = idx - 3
        else:
            assert idx == 0
    return rot_list[idx]


def normalize_euler(euler):
    candidates = np.array([-180, -90, 0, 90, 180])
    fixed = list(euler)
    for i, e in enumerate(fixed):
        dist = (candidates - e) ** 2
        fixed[i] = candidates[dist.argmin()]
    fixed = tuple(fixed)
    if fixed in rot_map:
        fixed = rot_map[fixed]
    return fixed


def canonize_euler(euler, brick_type=None, rots=None, ignore_symmetry=False, return_rots=False):
    euler = normalize_euler(euler)  # first assign euler angle to one of axis-aligned rotations
    if rots is None:
        assert brick_type is not None
        rots = get_brick_class(brick_type).get_valid_rotations()
    if len(rots) != 4 and not ignore_symmetry:
        euler = canonize_euler_helper(rots, euler)
    if return_rots:
        return euler, rots
    else:
        return euler


def canonize_euler_helper(rots, euler):
    if len(rots) == 2:
        'axis symmetric'
        if euler in [(0, 0, 0), (0, 90, 0), (180, 0, 0), (180, -90, 0)]:
            return euler
        if euler == (0, -90, 0):
            euler_new = (0, 90, 0)
        elif euler == (180, 0, 180):
            euler_new = (0, 0, 0)
        elif euler == (0, 0, 180):
            euler_new = (180, 0, 0)
        else:
            assert euler == (-180, 90, 0)
            euler_new = (180, -90, 0)
        return euler_new

    elif len(rots) == 1:
        if euler in [(0, 0, 0), (0, 90, 0), (0, -90, 0), (180, 0, 180)]:
            return (0, 0, 0)
        else:
            return (180, 0, 0)
    else:
        assert "Wrong get_brick_rotations return!"


def canonize_brick_type(brick_type):
    '''As some brick types are visually indistinguiable, map them to a uniform brick_type.'''
    return {
        '3070b': '3070'
    }.get(brick_type, brick_type)


class LegoKpsShapeCondDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--load_valid_pos', action='store_true',
                            help='Load the valid positions of current brick and rotation')
        parser.add_argument('--img_type', default='rgb', choices=['rgb', 'gray_scale', 'laplacian'], type=str.lower)
        parser.add_argument('--occs_h5', action='store_true', help='use hdf5 file to load occs')
        parser.add_argument('--top_center', action='store_true', help='treat center of the top face as the center')
        parser.add_argument('--max_objs', type=int, default=50,
                            help='max number of detect objects, will filter out images with more objects')
        parser.add_argument('--max_brick_types', type=int, default=10, help='max number of brick types in a step')
        parser.add_argument('--load_bbox', action='store_true', help='load_bbox')
        parser.add_argument('--load_mask', action='store_true', help='load mask')
        parser.add_argument('--load_conns', action='store_true', help='load_bbox')
        parser.add_argument('--load_bricks', action='store_true', help='load_bbox')
        parser.add_argument('--allow_invisible', action='store_true', help='allow invisible bricks')
        parser.add_argument('--camera_jitter', type=int, default=0, help='add noise to the azim and elev')
        parser.add_argument('--lap_random_k', action='store_true', help='randomly change the kernel size of k')
        parser.add_argument('--load_lpub', action='store_true', help='load lpub3d rendered images')
        parser.add_argument('--disallow_regular_brick', action='store_true', help='skip single-element bricks')
        parser.add_argument('--kp_sup', action='store_true', help='Provide the position of keypoint in the voxel.')
        parser.add_argument('--only_load_max_objs', action='store_true',
                            help='only load steps with maximum number of objs')
        parser.add_argument('--crop_brick_occs', action='store_true', help='Crop brick occs')
        parser.add_argument('--cbrick_brick_kp', action='store_true', help='Load brick kp for cbricks.')
        parser.add_argument('--ignore_rotation_symmetry', action='store_true',
                            help='Don\'t canonize rotaiton by symmetry.')
        parser.add_argument('--symmetry_aware_rotation_label', action='store_true')
        parser.add_argument('--min_mask_pixels', type=int, default=100, help='Minimum pixels for a brick mask.')
        parser.add_argument('--assoc_emb_n_samples', type=int, default=100,
                            help='Number of sampled points for each mask.')
        parser.add_argument('--start_idx', type=int, default=0, help='Skip some entries in the dataset')
        parser.add_argument('--predict_trans', action='store_true')
        parser.add_argument('--normalize_trans', action='store_true')
        parser.add_argument('--eval_mode', action='store_true', help='Load some extra things for evaluation.')
        parser.add_argument('--load_set', type=str, default='', help='if not empty, only load a specific set.')
        parser.add_argument('--load_lpub_old', action='store_true')
        parser.set_defaults(input_nc=3, output_nc=3)
        return parser

    def __init__(self, opt, val):
        super().__init__(opt, val)
        set_global_definition(LegoKPSDefinition())

        if not val:
            data_dir = os.path.abspath(opt.dataroot)
        else:
            data_dir = os.path.abspath(opt.val_dataroot)
        self.data_dir = data_dir

        if opt.load_set:
            idxs = [opt.load_set]
        else:
            idxs = list(sorted(os.listdir(data_dir)))
            if 'metadata' in idxs:
                idxs.remove('metadata')
            if 'occs' in idxs:
                idxs.remove('occs')
        if opt.occs_h5:
            self.h5_dataset = None
            with open(os.path.join(data_dir, 'occs', 'occ_idxs.pkl'), 'rb') as f:
                self.occs_idxs = pickle.load(f)

        self.prev_obj_occ = []
        self.obj_q = []
        self.img_paths = []
        self.ts = []  # translations
        self.rs = []  # rotations
        self.rs_id = []  # classification id of the rotation
        self.obj_rs = []
        self.btypes = []  # brick types in the form of brick id or a BricksPC dict specifying the CBrick
        self.bboxes = []
        self.masks = []
        self.areas = []
        self.img_ids = []
        self.op_types = []
        if opt.load_valid_pos:
            self.valid_positions = []
        self.keypoints = []
        self.view_directions = []  # (azim, elev)
        self.obj_transforms = []  # (scale, t_x, t_y, t_z)

        self.occ_size = np.array([130, 130, 130])

        if opt.load_conns:
            self.conns_list = []

        if opt.load_bricks:
            self.bricks_list = []

        self.indices = None
        if opt.max_objs > 0:
            self.indices = []

        ct = 0
        filtered_ct = 0
        for fname in idxs:
            if (not val and ct >= opt.max_dataset_size) or (val and ct >= opt.max_val_dataset_size):
                break
            path = os.path.join(data_dir, fname)
            occ_filename = 'occs.pkl'
            with open(os.path.join(path, 'info.json')) as f:
                d = json.load(f)

            op_len = len(d['operations'])
            if not opt.occs_h5:
                with open(os.path.join(path, occ_filename), 'rb') as f:
                    obj_occs = pickle.load(f)

            if opt.load_conns:
                conns_path = os.path.join(path, 'conns.pkl')
                with open(conns_path, 'rb') as f:
                    conns_list = pickle.load(f)

            if self.opt.load_bricks:
                bricks_list = []
                from bricks.brick_info import BricksPC
                from copy import deepcopy
                bs = BricksPC(np.array(d['grid_size']))
                for i_str, op in d['operations'].items():
                    i = int(i_str)
                    # if i > 0:
                    bricks_list.append(deepcopy(bs))

                    bs.apply_transform(tr.quaternion_matrix(op['obj_rotation_quat']))
                    b_step = d['operations'][i_str]['bricks']
                    for j in range(len(b_step)):
                        if 'brick_type' in b_step[j]:
                            if not bs.add_brick(b_step[j]['brick_type'], b_step[j]['brick_transform']['position'],
                                                b_step[j]['brick_transform']['rotation'], b_step[j]['op_type'],
                                                canonical=False):
                                print('Cannot add brick at #', i)
                                bs.add_brick(b_step[j]['brick_type'], b_step[j]['brick_transform']['position'],
                                             b_step[j]['brick_transform']['rotation'], b_step[j]['op_type'],
                                             canonical=False, debug=True)
                        else:
                            cbrick = dict_to_cbrick(b_step[j])
                            if not add_cbrick_to_bricks_pc(bs, cbrick, b_step[j]['op_type']):
                                print('Cannot add brick at #', i)
                                import ipdb;
                                ipdb.set_trace()

            for i in range(0, op_len):
                if (not val and ct >= opt.max_dataset_size) or (val and ct >= opt.max_val_dataset_size):
                    break

                # Load step level information
                if opt.disallow_regular_brick:
                    is_regular = True
                    if len(d['stats']['parent_brick_clss'][i][0]) == 2:
                        # legacy code
                        for parent_brick_cls, _ in d['stats']['parent_brick_clss'][i]:
                            if parent_brick_cls == 'CBrick':
                                is_regular = False
                                break
                    else:
                        for parent_brick_cls, _, rel_e_idx in d['stats']['parent_brick_clss'][i]:
                            if parent_brick_cls == 'CBrick':
                                is_regular = False
                                break
                    if is_regular:
                        continue

                if opt.occs_h5:
                    occ_path = os.path.join(path, 'occs.pkl')
                    self.prev_obj_occ.append((occ_path, i - 1))
                else:
                    if i == 0:
                        self.prev_obj_occ.append(None)
                    else:
                        self.prev_obj_occ.append(obj_occs[i - 1])

                if not opt.load_lpub:
                    self.img_paths.append(os.path.join(path, str(i).zfill(3) + '.png'))
                else:
                    p1, p2 = os.path.split(data_dir)
                    if opt.load_lpub_old:
                        self.img_paths.append(os.path.join(path, str(i).zfill(3) + '_lpub.png'))
                    else:
                        self.img_paths.append(os.path.join(p1, p2 + '_lpub3d', fname, str(i).zfill(3) + '_lpub.png'))

                self.img_ids.append(f'{fname}_{i}')

                self.obj_rs.append(d['operations'][str(i)]['obj_rotation_quat'])
                obj_q = np.array(d['operations'][str(i)]['obj_rotation_quat'])
                if obj_q[0] < 0:
                    obj_q *= -1
                self.obj_q.append(obj_q)

                # Load brick level information
                bs = d['operations'][str(i)]['bricks']
                ts_step = []
                rs_step = []
                rs_id_step = []
                btypes_step = []
                bboxes_step = []
                masks_step = []
                areas_step = []
                keypoints_step = []
                op_type_step = []
                cbrick_dict_step = []

                has_invisible_kps = False
                has_invisible_box = False
                for j in range(len(bs)):
                    cbrick = None
                    if 'brick_type' in bs[j]:
                        bs[j]['brick_type'] = canonize_brick_type(bs[j]['brick_type'])

                    rs_step.append(bs[j]['brick_transform']['rotation'])
                    if 'brick_type' not in bs[j]:
                        brick_type = None
                        cbrick = dict_to_cbrick(bs[j])
                        cbrick_dict = dict(bs[j]['canonical_state'])
                        del cbrick_dict['position']
                        del cbrick_dict['rotation']
                        if cbrick_dict not in cbrick_dict_step:
                            cbrick_dict_step.append(cbrick_dict)
                        if opt.ignore_rotation_symmetry:
                            rots = [0, 1, 2, 3]
                        else:
                            if 'valid_rotations' in bs[j]:
                                rots = bs[j]['valid_rotations']
                            else:
                                rots = get_cbrick_rotations(cbrick)

                    else:
                        brick_type = bs[j]['brick_type']
                        rots = None
                    r_id = euler2id(bs[j]['brick_transform']['rotation_euler'], brick_type, rots,
                                    ignore_symmetry=opt.ignore_rotation_symmetry,
                                    symmetry_aware_label=opt.symmetry_aware_rotation_label)
                    assert r_id >= 0

                    rs_id_step.append(r_id)
                    ts_step.append(bs[j]['brick_transform']['position'])
                    if opt.top_center:
                        if 'brick_type' not in bs[j]:
                            if opt.cbrick_brick_kp:
                                '''Keypoint of a brick is used.'''
                                if cbrick is None:
                                    cbrick = dict_to_cbrick(bs[j])
                                ts_step[-1] = get_cbrick_keypoint(cbrick, 'brick')[0]
                            else:
                                h = cbrick.get_height()
                                ts_step[-1][1] += h
                        else:
                            ts_step[-1][1] += get_brick_class(bs[j]['brick_type']).get_height()
                    if 'brick_type' not in bs[j]:  # CBrick
                        btypes_step.append(bs[j])
                    else:
                        btypes_step.append(bs[j]['brick_type'])
                    if bs[j]['bbox']:
                        bboxes_step.append(bs[j]['bbox'][0] + bs[j]['bbox'][1])
                        areas_step.append((bboxes_step[-1][3] - bboxes_step[-1][1]) * \
                                          (bboxes_step[-1][2] - bboxes_step[-1][0]))
                        if areas_step[-1] <= 0:
                            has_invisible_box = True
                    else:
                        has_invisible_box = True
                        # print('no bbox: ', self.img_ids[-1], self.indices[-1])
                        bboxes_step.append([ts_step[-1][0], ts_step[-1][2], ts_step[-1][0], ts_step[-1][
                            2]])  # need to signify invalidity if bbox is actually used by the model
                        areas_step.append(0)
                    masks_step.append(bs[j]['mask'])
                    if 'mask_pixel_count' in bs[j] and bs[j]['mask_pixel_count'] < opt.min_mask_pixels:
                        has_invisible_box = True
                    if opt.cbrick_brick_kp:
                        keypoints_step.append(bs[j]['brick_transform']['keypoint_brick'][:2])
                    else:
                        keypoints_step.append(bs[j]['brick_transform']['keypoint'][:2])
                    if min(keypoints_step[-1]) < 0 or max(keypoints_step[-1]) > 511:
                        has_invisible_kps = True
                    op_type_step.append(bs[j]['op_type'])

                if opt.load_conns:
                    self.conns_list.append(conns_list[i])
                if opt.load_bricks:
                    self.bricks_list.append(bricks_list[i])

                self.ts.append(ts_step)
                self.rs.append(rs_step)
                self.rs_id.append(rs_id_step)
                self.btypes.append(btypes_step)
                self.bboxes.append(bboxes_step)
                self.masks.append(masks_step)
                self.areas.append(areas_step)
                self.keypoints.append(keypoints_step)
                self.op_types.append(op_type_step)
                self.obj_transforms.append((d['obj_scale'], *d['obj_center']))

                self.view_directions.append(d['operations'][str(i)]['view_direction'])

                num_brick_types = len(set(b for b in btypes_step if isinstance(b, str))) + len(cbrick_dict_step)
                if (opt.max_objs > 0 and len(ts_step) <= opt.max_objs) and \
                        not has_invisible_kps and \
                        (num_brick_types <= opt.max_brick_types) and \
                        (opt.allow_invisible or not has_invisible_box):
                    if not opt.only_load_max_objs or len(ts_step) == opt.max_objs:
                        self.indices.append(len(self.ts) - 1)
                    else:
                        filtered_ct += 1
                else:
                    filtered_ct += 1

                ct += 1

        print(f'Filtered {filtered_ct} steps.')
        if self.indices is not None:
            self.indices = self.indices[self.opt.start_idx:]

    def _transform(self, img, random_ksize=False):
        if self.opt.img_type == 'laplacian':
            ksize = None
            ksize = 3
            if random_ksize:
                ksize = np.random.choice([3, 5, 7, 9])
            img = np.array(cv2.Laplacian(np.array(img)[..., 0], cv2.CV_32F, ksize=ksize))
            img = np.repeat(np.clip(img, 0, 255)[..., None], 3, axis=2)
            img /= 255.0
        img = TF.to_tensor(img)
        # For ResNet pretrained backbone
        # img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.opt.img_type == 'rgb':
            # white bg
            img = TF.normalize(img, mean=[0.92500746, 0.92472865, 0.92534608], std=[0.0467119, 0.04713448, 0.04659452])
        else:
            if self.opt.img_type == 'gray_scale':
                img = TF.normalize(img, mean=[0.92488375, 0.92488375, 0.92488375],
                                   std=[0.04443009, 0.04443009, 0.04443009])
            elif self.opt.img_type == 'laplacian':
                img = TF.normalize(img, mean=[0.00576893, 0.00576893, 0.00576893],
                                   std=[0.00208021, 0.00208021, 0.00208021])

        return img

    def __getitem__(self, index):
        if self.indices is not None:
            index = self.indices[index]

        img_path = self.img_paths[index]
        if self.opt.img_type == 'rgb':
            img = Image.open(img_path).convert('RGB')
        elif self.opt.img_type in ['gray_scale', 'laplacian']:
            img = Image.open(img_path).convert('L').convert('RGB')

        img = self._transform(img, random_ksize=self.opt.lap_random_k)

        obj_occ_prev_voxel = np.zeros(self.occ_size, dtype=bool)
        if self.opt.occs_h5:
            if self.h5_dataset is None:
                self.h5_dataset = h5.File(os.path.join(self.data_dir, 'occs', 'occs.h5'), 'r')['dataset']
            prev_occ_key = self.prev_obj_occ[index]
            if prev_occ_key[1] >= 0:
                start_idx, end_idx = self.occs_idxs[prev_occ_key]
                prev_occ = self.h5_dataset[start_idx:end_idx]
            else:
                prev_occ = None
        else:
            if self.prev_obj_occ[index] is not None:
                prev_occ = np.array(self.prev_obj_occ[index])
            else:
                prev_occ = None
        if prev_occ is not None:
            obj_occ_prev_voxel[prev_occ[:, 0], prev_occ[:, 1], prev_occ[:, 2]] = True

        b_ids = []
        cbrick_dict_list = []
        cbrick_list = []

        # bids of cbricks starts from -1 and is decremented by 1 for each new cbrick type
        for btype_or_dict in self.btypes[index]:
            if isinstance(btype_or_dict, str):
                b_ids.append(get_all_brick_ids().index(btype_or_dict))
            else:
                assert isinstance(btype_or_dict, dict)
                cbrick_canonical_dict = dict(btype_or_dict['canonical_state'])
                del cbrick_canonical_dict['position']
                del cbrick_canonical_dict['rotation']
                if not cbrick_dict_list:
                    cbrick_dict_list.append(cbrick_canonical_dict)
                    cbrick_list.append(dict_to_cbrick(btype_or_dict, reset_pose=True))
                    b_id = -1
                else:
                    found = False
                    for i, cbrick_d in enumerate(cbrick_dict_list):
                        if cbrick_canonical_dict == cbrick_d:
                            b_id = - i - 1
                            found = True
                    if not found:
                        b_id = -len(cbrick_dict_list) - 1
                        cbrick_dict_list.append(cbrick_canonical_dict)
                        cbrick_list.append(dict_to_cbrick(btype_or_dict, reset_pose=True))

                b_ids.append(b_id)

        trans = self.ts[index]
        if self.opt.predict_trans:
            if self.opt.normalize_trans:
                trans_pred = (torch.as_tensor(trans) - gdef.translation_mean) / gdef.translation_std
                assert (trans_pred <= 1).all() and (trans_pred >= -1).all()
            else:
                trans_pred = torch.as_tensor(trans)
        azim_jitter, elev_jitter = 0, 0
        if self.opt.camera_jitter > 0:
            azim_jitter += np.random.randint(-self.opt.camera_jitter, self.opt.camera_jitter + 1)
            elev_jitter += np.random.randint(-self.opt.camera_jitter, self.opt.camera_jitter + 1)

        feed_dict = {
            'img': img, 'img_path': img_path,
            'obj_occ_prev': obj_occ_prev_voxel,
            'obj_r': np.array(self.obj_rs[index]),
            'obj_q': np.array(self.obj_q[index]),
            'btype': self.btypes[index],
            'azim': self.view_directions[index][0] + azim_jitter,
            'elev': self.view_directions[index][1] + elev_jitter,
            'obj_scale': self.obj_transforms[index][0],
            'obj_center': np.array(self.obj_transforms[index][1:]),

            'target': {
                'trans': trans,
                'op_type': torch.as_tensor(self.op_types[index]),
            }

        }

        if self.opt.load_bbox:
            feed_dict['target']['boxes'] = torch.as_tensor(self.bboxes[index])
        if self.opt.load_mask:
            masks_orig = torch.as_tensor(mask_utils.decode(self.masks[index]))
            feed_dict['target']['masks'] = masks_orig
        if self.opt.load_conns:
            feed_dict['target']['conns'] = self.conns_list[index]
        if self.opt.load_bricks:
            feed_dict['bricks'] = self.bricks_list[index]

        if self.opt.eval_mode:
            # order ground truth by brick types to facilitate evaluation.

            def get_brick_name(bid):
                if bid < 0:
                    return 'CBrick #' + str(-bid - 1)
                else:
                    return get_all_brick_ids()[bid]

            idxs = np.argsort(np.array(b_ids))
            b_ids_reordered = np.array(b_ids)[idxs]
            bid_ct = {}
            for b in b_ids_reordered:
                if b not in bid_ct:
                    bid_ct[b] = 1
                else:
                    bid_ct[b] += 1

            feed_dict['target']['ordered'] = {
                'trans': [trans[idx] for idx in idxs],
                'op_type': feed_dict['target']['op_type'][idxs],
                'bid': b_ids_reordered,
                'rot_decoded': [id2euler(self.rs_id[index][idx],
                                         symmetry_aware_label=self.opt.symmetry_aware_rotation_label)
                                for idx in idxs],
                'bid_decoded': [get_brick_name(b) for b in b_ids_reordered],
                'bid_ct': list(bid_ct.items()),
                'reverse_idxs': np.argsort(idxs),
            }

        output_h = gdef.input_h // gdef.down_ratio
        output_w = gdef.input_w // gdef.down_ratio
        num_classes = len(get_all_brick_ids())

        num_objs = len(self.ts[index])
        b_counter = collections.Counter(b_ids)
        num_distinct_bricks = len(b_counter.keys())
        max_num_bricks = max(b_counter.values())

        # For each step, we group bricks by types and store their information in each group
        hm = np.zeros((num_distinct_bricks, output_h, output_w), dtype=np.float32)
        reg = np.zeros((num_distinct_bricks, max_num_bricks, 2), dtype=np.float32)
        ind = np.zeros((num_distinct_bricks, max_num_bricks), dtype=np.int64)
        reg_mask = np.zeros((num_distinct_bricks, max_num_bricks), dtype=np.uint8)
        b_ids_ = np.zeros((num_distinct_bricks), dtype=np.int32)
        rots = np.ones((num_distinct_bricks, max_num_bricks), dtype=np.int32) * -100  # -100 for ignored entry
        kps = np.zeros((num_distinct_bricks, max_num_bricks, 2), dtype=np.int32)
        bboxes = np.zeros((num_distinct_bricks, max_num_bricks, 4), dtype=np.int32)
        if self.opt.predict_trans:
            trans_list = np.zeros((num_distinct_bricks, max_num_bricks, 3), dtype=np.float32)

        if self.opt.load_mask:
            masks_resized = torch.nn.functional.interpolate(masks_orig.permute((2, 0, 1))[None], (128, 128))[0].numpy()
            # + 1 for backgrounds
            mask_all = np.zeros((128, 128), dtype=np.uint8)
            if self.opt.assoc_emb:
                mask_instance = np.zeros((num_distinct_bricks, 128, 128), dtype=np.uint8)
                n_samples = self.opt.assoc_emb_n_samples
                assoc_emb_sample_inds = np.zeros((num_distinct_bricks, max_num_bricks, n_samples, 2), dtype=np.uint8)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        # record the number of instances for each brick type
        b_counter_cur = collections.defaultdict(int)
        # indexing each brick type
        bid2ind = {}
        reorder_map = {}
        for k in range(num_objs):
            visible = False
            bbox = self.bboxes[index][k]
            if bbox is not None:
                bbox = (np.array(bbox) / gdef.down_ratio).round()
                bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
                bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h >= 0 and w >= 0:
                    visible = True
            # note that some objects may be small than 4x4
            if visible or self.opt.allow_invisible:
                cls_id = int(b_ids[k])
                if cls_id not in bid2ind:
                    bid2ind[cls_id] = len(bid2ind)
                first_ind = bid2ind[cls_id]
                second_ind = b_counter_cur[cls_id]
                reorder_map[(cls_id, second_ind)] = k
                b_counter_cur[cls_id] += 1

                radius = gdef.hm_gauss
                ct = np.array(
                    self.keypoints[index][k], dtype=np.float32) / gdef.down_ratio
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[first_ind], ct_int, radius)
                # wh[k] = 1. * w, 1. * h
                ind[first_ind][second_ind] = ct_int[1] * output_w + ct_int[0]
                reg[first_ind][second_ind] = ct - ct_int
                reg_mask[first_ind][second_ind] = 1
                b_ids_[first_ind] = b_ids[k]
                rots[first_ind][second_ind] = self.rs_id[index][k]
                kps[first_ind][second_ind] = self.keypoints[index][k]
                if self.opt.predict_trans:
                    trans_list[first_ind][second_ind] = trans_pred[k]
                if self.opt.load_bbox:
                    bboxes[first_ind][second_ind] = self.bboxes[index][k]
                if self.opt.load_mask:
                    nonzero_inds = masks_resized[k].nonzero()
                    mask_all[nonzero_inds] = first_ind + 1
                    if self.opt.assoc_emb:
                        mask_instance[first_ind, nonzero_inds[0], nonzero_inds[1]] = second_ind + 1  # 0 for background
                        n = nonzero_inds[0].shape[0]
                        if n == 0:
                            assoc_emb_sample_inds[first_ind, second_ind, :, 0] = 0
                            assoc_emb_sample_inds[first_ind, second_ind, :, 1] = 0
                        else:
                            sample_inds = np.random.choice(range(n), n_samples)
                            assoc_emb_sample_inds[first_ind, second_ind, :, 0] = nonzero_inds[0][sample_inds]
                            assoc_emb_sample_inds[first_ind, second_ind, :, 1] = nonzero_inds[1][sample_inds]

            else:
                raise ValueError('Invalid bounding box! at ' + img_path + 'brick #' + str(k))

        ret = {'hm': hm, 'reg': reg, 'reg_mask': reg_mask, 'ind': ind,  # 'wh': wh,
               'bid': b_ids_, 'rot': rots, 'kp': kps,

               'bid_counter': list(sorted(list(b_counter_cur.items()),
                                          key=lambda p: bid2ind[p[0]])),  # for recovering each individual brick
               'reorder_map': reorder_map,
               'cbrick': cbrick_list
               }
        if self.opt.load_bbox:
            ret['bbox'] = bboxes
        if self.opt.load_mask:
            ret['mask'] = mask_all
            if self.opt.assoc_emb:
                ret['mask_instance'] = mask_instance
                ret['assoc_emb_sample_inds'] = assoc_emb_sample_inds

        feed_dict.update(ret)

        brick_occs = []
        # Get occ voxel for each brick type
        for bid in b_counter_cur.keys():
            extra_point = None
            if self.opt.brick_voxel_embedding_dim > 0:
                extra_value = 1
            else:
                extra_value = 2
            if bid < 0:
                if self.opt.kp_sup:
                    extra_point = [0, cbrick_list[-bid - 1].get_height(), 0]
                brick_occs.append(torch.as_tensor(get_cbrick_enc_voxel(cbrick_list[-bid - 1],
                                                                       extra_point=extra_point,
                                                                       extra_point_value=extra_value
                                                                       )))
            else:
                b_type = get_all_brick_ids()[bid]
                if self.opt.kp_sup:
                    extra_point = [0, get_brick_class(b_type).get_height(), 0]
                brick_occs.append(torch.as_tensor(get_brick_enc_voxel(b_type, extra_point=extra_point,
                                                                      extra_point_value=extra_value)))

        brick_occs = torch.stack(brick_occs).unsqueeze(1)
        if self.opt.brick_voxel_embedding_dim == 0:
            brick_occs = brick_occs.float()
        else:
            brick_occs = brick_occs.long()
        if self.opt.crop_brick_occs:
            grid_size = 65
            min_xyz = [32, 16, 32]
            max_xyz = [d + grid_size for d in min_xyz]
            brick_occs = brick_occs[:, :, min_xyz[0]:max_xyz[0], min_xyz[1]:max_xyz[1], min_xyz[2]:max_xyz[2]]

        feed_dict['brick_occs'] = brick_occs

        if self.opt.num_bricks_single_forward > 1:
            # flattened brick information for model supporting multiple bricks
            reg_flat = np.zeros((self.opt.max_objs, 2), dtype=np.float32)
            ind_flat = np.zeros((self.opt.max_objs,), dtype=np.int64)
            reg_mask_flat = np.zeros((self.opt.max_objs,), dtype=np.uint8)
            bid_flat = np.zeros((self.opt.max_objs,), dtype=np.int32)
            rot_flat = np.ones((self.opt.max_objs,), dtype=np.int32) * -100  # -100 for ignored entry
            if self.opt.predict_trans:
                trans_flat = np.ones((self.opt.max_objs, 3), dtype=np.float32)

            # [num_distinct_brick_types, max_num_bricks_of_one_type, ...]
            num_brick_total = 0
            for j in range(reg_mask.shape[0]):
                num_brick_this = int(reg_mask[j].sum())
                reg_flat[num_brick_total:num_brick_total + num_brick_this] = reg[j, :num_brick_this]
                ind_flat[num_brick_total:num_brick_total + num_brick_this] = ind[j, :num_brick_this]
                rot_flat[num_brick_total:num_brick_total + num_brick_this] = rots[j, :num_brick_this]
                if self.opt.predict_trans:
                    trans_flat[num_brick_total:num_brick_total + num_brick_this] = trans_list[j, :num_brick_this]
                bid_flat[num_brick_total:num_brick_total + num_brick_this] = b_ids_[j]
                num_brick_total += num_brick_this
            reg_mask_flat[:num_brick_total] = 1
            feed_dict.update({
                'reg_flat': reg_flat, 'reg_mask_flat': reg_mask_flat, 'ind_flat': ind_flat,
                'bid_flat': bid_flat, 'rot_flat': rot_flat,
            })
            if self.opt.predict_trans:
                feed_dict['trans_flat'] = trans_flat

        return feed_dict

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        else:
            return len(self.img_paths)


def collate_fn(batch):
    concat_names = ['brick_pt_init', 'brick_pt', 'trans_flat_cat', 'rot_flat_cat', 'bid_flat_cat', 'mask_flat_cat']
    list_tensor_names = [
        'hm', 'reg', 'reg_mask', 'ind', 'bid', 'rot', 'kp', 'brick_occs', 'bbox',
        'mask_instance', 'assoc_emb_sample_inds'
    ]
    d = {}
    targets = []
    for k in batch[0].keys():
        if k in concat_names:
            if isinstance(batch[0][k], list):
                d[k + '_length'] = torch.as_tensor([len(b[k]) for b in batch])
            else:
                d[k + '_length'] = torch.as_tensor([b[k].shape[0] for b in batch])
            d[k] = torch.cat([torch.as_tensor(b[k]) for b in batch])
        elif k in list_tensor_names:
            d[k] = [torch.as_tensor(b[k]) for b in batch]
        elif isinstance(batch[0][k], str):
            d[k] = [b[k] for b in batch]
        elif isinstance(batch[0][k], np.ndarray):
            d[k] = torch.stack([torch.as_tensor(b[k]) for b in batch], 0)
        elif isinstance(batch[0][k], float):
            d[k] = torch.tensor([b[k] for b in batch], dtype=torch.float64)
        elif isinstance(batch[0][k], int):
            d[k] = torch.tensor([b[k] for b in batch])
        elif isinstance(batch[0][k], torch.Tensor):
            d[k] = torch.stack([b[k] for b in batch], 0)
        elif k == 'target':
            targets = tuple(b[k] for b in batch)
        else:
            d[k] = [b[k] for b in batch]
            # raise NotImplementedError('Key {} has unsupported data type {} in batch!'.format(k, type(batch[0][k])))

    return d, targets


class LegoKPSDefinition(DatasetDefinitionBase):
    translation_mean = torch.as_tensor([0, 24.5, 0])
    translation_std = torch.as_tensor([32.5, 32.5, 32.5])
    occ_translation = torch.as_tensor([-65, -65 // 4, -65])
    down_ratio = 4
    input_h = 512
    input_w = 512
    hm_gauss = 1
    grid_size = np.array([65, 65, 65])
    occ_size = np.array([130, 130, 130])
