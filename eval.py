import copy
import json
import os
import shutil
import warnings

import numpy as np
import pytorch3d.transforms as pt
import torch
import trimesh.transformations as tr
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from bricks.brick_info import (
    get_brick_class, get_cbrick_keypoint, dict_to_cbrick, add_cbrick_to_bricks_pc, CBrick
)
from datasets import create_dataset
from debug.utils import render_dict_simple
from lego.utils.camera_utils import get_cameras, get_scale
from lego.utils.inference_utils import recompute_conns
from models import create_model
from models.utils import Meters
from options.test_options import TestOptions
from util.util import mkdirs

warnings.filterwarnings("ignore", category=UserWarning)


def replace_poses(opt, d, pose_list, replace_per_step=True, subm_cbricks=None):
    n_steps = len(d['operations'])
    b_steps_orig = [copy.deepcopy(d['operations'][str(i)]['bricks']) for i in range(n_steps)]
    subm_ct = 0
    for ex in pose_list:
        b_step = d['operations'][str(ex['step_id'])]['bricks']
        bricks_pred = ex['bricks_pred']
        has_subm = False
        for i, b in enumerate(b_step):
            rot_euler = bricks_pred[i]['rot_decoded']
            rot_quat = tr.quaternion_from_euler(*list(map(lambda x: x * np.pi / 180, rot_euler)))
            position = np.array(bricks_pred[i]['trans'])
            if 'brick_type' in b:
                brick_type = b['brick_type']
                kp_offset = [0, get_brick_class(brick_type).get_height(), 0]
            else:
                if subm_cbricks is not None:
                    cbrick = subm_cbricks[subm_ct]
                    has_subm = True
                    if cbrick is not None:
                        b['canonical_state'] = cbrick.to_dict()
                    else:
                        cbrick = dict_to_cbrick(b)
                else:
                    cbrick = dict_to_cbrick(b)
                kp_offset = get_cbrick_keypoint(cbrick, policy='brick' if opt.cbrick_brick_kp else 'simple')[
                                0] - cbrick.position

            if opt.top_center:
                position -= kp_offset
            b['brick_transform']['position'] = list(map(float, position))
            b['brick_transform']['rotation'] = list(map(float, rot_quat))
        if has_subm:
            subm_ct += 1

    ks = list(d.keys())
    ks.remove('operations')
    d_template = {k: d[k] for k in ks}
    ds = []
    if replace_per_step:
        for i in range(n_steps):
            d_this = copy.deepcopy(d_template)
            if i > 0:
                d['operations'][str(i - 1)]['bricks'] = b_steps_orig[i - 1]
            d_this['operations'] = copy.deepcopy({str(idx): d['operations'][str(idx)] for idx in range(i + 1)})
            ds.append(d_this)
        return ds
    else:
        return d


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

from bricks.brick_info import get_cbrick_enc_voxel


def get_brick_occ(opt, cbrick):
    extra_point = None
    if opt.brick_voxel_embedding_dim > 0:
        extra_value = 1
    else:
        extra_value = 2
        extra_point = [0, cbrick.get_height(), 0]
    brick_occ = torch.as_tensor(get_cbrick_enc_voxel(cbrick, extra_point=extra_point, extra_point_value=extra_value))

    if opt.brick_voxel_embedding_dim == 0:
        brick_occ = brick_occ.float()
    else:
        brick_occ = brick_occ.long()
    if opt.crop_brick_occs:
        grid_size = 65
        min_xyz = [32, 16, 32]
        max_xyz = [d + grid_size for d in min_xyz]
        brick_occ = brick_occ[min_xyz[0]:max_xyz[0], min_xyz[1]:max_xyz[1], min_xyz[2]:max_xyz[2]]
    return brick_occ


from bricks.brick_info import BricksPC


def evaluate_set(opt, model, set_path, meters, is_subm=False, subm_cbricks=None, subm_meters=None):
    result_list = []
    opt.load_set = set_path
    opt.serial_batches = True
    autoregressive = opt.autoregressive_inference
    if autoregressive:
        opt.batch_size = 1
        current_bricks_pc = BricksPC(grid_size=(65, 65, 65), record_parents=False)
        if subm_cbricks is not None:
            subm_ct = 0

    meters_this = Meters()
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    print(f'Loading from {set_path}, number of steps {len(dataset)}')

    if is_subm:
        oracle_ct = 0  # disable oracle in subm
    else:
        oracle_ct = len(dataset) * opt.oracle_percentage
    for i, data in tqdm(enumerate(dataset), total=len(dataset) // opt.batch_size):
        if i > oracle_ct and autoregressive:
            data[0]['obj_occ_prev'] = torch.as_tensor(current_bricks_pc.get_occ_with_rotation()[0])[None]

            data[0]['bricks'] = [copy.deepcopy(current_bricks_pc)]
            transforms = pt.Transform3d().translate(-data[0]['obj_center']).scale(data[0]['obj_scale'][0]).cuda()
            cameras = get_cameras(azim=data[0]['azim'][0], elev=data[0]['elev'][0])
            data[1][0]['conns'] = recompute_conns(
                current_bricks_pc, op_type=0, transforms=transforms, cameras=cameras, scale=get_scale())

            if subm_cbricks is not None:
                # Use the predicted submodules as input
                for j, bid_counter in enumerate((data[0]['bid_counter'])):
                    has_subm = False
                    for k, (bid, ct) in enumerate(bid_counter):
                        if bid < 0:
                            has_subm = True
                            data[0]['brick_occs'][j][k] = get_brick_occ(opt, subm_cbricks[subm_ct])
                            data[0]['cbrick'][j][-bid - 1] = subm_cbricks[subm_ct]

                    if has_subm:
                        subm_ct += 1

        model.set_input(data)
        model.test()
        losses = model.get_current_losses()
        data, targets = data
        for name, v in losses.items():
            obj_sum = sum(data['reg_mask'][i].sum().item() for i in range(len(data['reg_mask'])))
            meters.update(name, v * obj_sum, obj_sum)

        def zip_keys(exs, features=['trans', 'rot_decoded', 'bid', 'bid_decoded'], n=-1):
            res = []
            if n == -1:
                n = len(exs[features[0]])
            for idx in range(n):
                ex = {}
                for k in features:
                    if k in exs:
                        ex[k] = exs[k][idx]
                        if isinstance(ex[k], torch.Tensor):
                            ex[k] = ex[k].cpu().numpy()
                res.append(ex)
            return res

        for j in range(data['img'].shape[0]):
            use_gt = False

            target = targets[j]['ordered']
            num_bricks = len(target['bid'])
            bricks_gt = zip_keys(target, n=num_bricks)
            if not use_gt:
                detection = model.detections[j]
                idxs = detection['bid'].argsort().cpu().numpy()
                for k, v in detection.items():
                    if isinstance(v, (torch.Tensor, np.ndarray)):
                        detection[k] = detection[k][idxs]
                    else:
                        detection[k] = [detection[k][idx] for idx in idxs]
                bid_ct = target['bid_ct']

                bricks_pred = zip_keys(detection, n=num_bricks)
                bricks_pred_matched = []
                brick_ct = 0
                # Perform Hungarian matching
                for _, ct in bid_ct:
                    trans_gt = np.array(target['trans'])[brick_ct:brick_ct + ct, None]
                    trans_pred = np.array(detection['trans'])[None, brick_ct:brick_ct + ct]
                    trans_mat = ((trans_gt - trans_pred) ** 2).sum(axis=-1)
                    trans_mat = (trans_mat < 0.1).astype(np.int8)
                    _, pred_idxs = linear_sum_assignment(-trans_mat)
                    for idx in pred_idxs:
                        bricks_pred_matched.append(bricks_pred[brick_ct + idx])
                    brick_ct += ct

            else:
                bricks_pred = bricks_gt
                for b in bricks_pred:
                    b['trans'] = np.array(b['trans'])
                bricks_pred_matched = bricks_pred

            brickwise_correct_ct = 0
            all_correct = True
            for b_gt, b_pred in zip(bricks_gt, bricks_pred_matched):
                b_pred['bid_decoded'] = b_gt['bid_decoded']
                if b_gt['rot_decoded'] == b_pred['rot_decoded'] and (b_gt['trans'] == b_pred['trans']).all():
                    brickwise_correct_ct += 1
                    b_pred['correct'] = True
                else:
                    all_correct = False
                    b_pred['correct'] = False

            meters_this.update('brickwise_acc', brickwise_correct_ct, num_bricks)
            meters_this.update('stepwise_acc', int(all_correct), 1)

            img_path = data['img_path'][j]
            img_fname = os.path.basename(img_path)
            if '_' in img_fname:
                step_id = int(img_fname[:-9])
            else:
                step_id = int(img_fname[:-4])

            set_id = img_path.split('/')[-2:]
            result_list.append({
                'set_id': set_id,
                'step_id': step_id,
                'bricks_pred': [
                    bricks_pred_matched[idx] for idx in target['reverse_idxs']
                ] if i > oracle_ct or (not autoregressive) else [
                    bricks_gt[idx] for idx in target['reverse_idxs']
                ]
            })
            if autoregressive:
                bs = result_list[-1]['bricks_pred']
                for i, b in enumerate(bs):
                    rot_quat = tr.quaternion_from_euler(*list(map(lambda x: x * np.pi / 180, b['rot_decoded'])))
                    if b['bid'] >= 0:
                        kp_offset = np.array([0, get_brick_class(b['bid_decoded']).get_height(), 0])
                        current_bricks_pc.add_brick(b['bid_decoded'], b['trans'] - kp_offset, rot_quat,
                                                    op_type=targets[0]['op_type'][i].cpu().numpy(), no_check=True)
                    else:
                        cbrick_canonical = data['cbrick'][0][int(-b['bid']) - 1]
                        kp_offset = \
                            get_cbrick_keypoint(cbrick_canonical, policy='brick' if opt.cbrick_brick_kp else 'simple')[
                                0] - cbrick_canonical.position
                        cbrick_this = copy.deepcopy(cbrick_canonical)
                        cbrick_this.position = b['trans'] - kp_offset
                        cbrick_this.rotaiton = rot_quat
                        assert add_cbrick_to_bricks_pc(current_bricks_pc, cbrick_this,
                                                       op_type=targets[0]['op_type'], no_check=True)

    n_steps = meters_this.n_d['stepwise_acc']
    if subm_cbricks is not None:
        n_steps += subm_meters.n_d['stepwise_acc']

    mtc = n_steps - meters_this.sum_d['stepwise_acc']
    if subm_cbricks is not None:
        mtc -= subm_meters.sum_d['stepwise_acc']

    meters_this.update('mtc_raw', mtc)
    meters_this.update('mtc_norm', mtc / n_steps)
    meters_this.update('setwise_acc', int(mtc == 0))

    if not is_subm:
        meters.merge_from(meters_this)
    else:
        meters.merge_from(meters_this, ks=['brickwise_acc', 'stepwise_acc'])

    # print('Submodule:', is_subm, ' Set:', set_path, 'correct:', mtc == 0)
    # for k, v in meters_this.avg_dict().items():
    #     print('\t', k, ':', v)
    if is_subm:
        if autoregressive:
            cbrick = CBrick(current_bricks_pc, [0, 0, 0], [1, 0, 0, 0])
        else:
            cbrick = None
        return result_list, cbrick, meters_this
    return result_list


class EvalOptions(TestOptions):
    def initialize(self, parser):
        parser = TestOptions.initialize(self, parser)
        parser.add_argument('--n_set', type=int, default=1, help='Number of sets to be evaluated.')
        parser.add_argument('--output_pred_json', action='store_true')
        parser.add_argument('--output_lpub3d_ldr', action='store_true')
        parser.add_argument('--render_pred_json', action='store_true')
        parser.add_argument('--single_json', action='store_true')
        parser.add_argument('--autoregressive_inference', action='store_true')
        parser.add_argument('--oracle_percentage', type=float, default=0,
                            help='Percentage of steps that are assumed to be true.')
        return parser


if __name__ == '__main__':
    opt = EvalOptions().parse()  # get training options
    opt.serial_batches = True
    opt.eval_mode = True
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    model.eval()

    result_dir = os.path.join(opt.results_dir, opt.name, opt.dataset_alias)
    mkdirs(result_dir)
    result_list = []
    meters = Meters()
    fres = open(os.path.join(result_dir, 'eval_result.txt'), 'w')
    if opt.output_pred_json:
        output_dir = os.path.join(result_dir, 'output')

    set_paths = []
    data_dir = os.path.abspath(opt.dataroot)
    set_paths = list(sorted(os.listdir(data_dir)))
    for n in ['occs', 'metadata']:
        if n in set_paths:
            set_paths.remove(n)

    if opt.n_set >= 0:
        set_paths = set_paths[:opt.n_set]

    from collections import defaultdict

    subm_cbricks_d = defaultdict(list)
    subm_meters_d = defaultdict(Meters)
    if 'architecture' in opt.dataset_alias:
        subm_dependency = subm_dependency_all['architecture']
    elif 'classics' in opt.dataset_alias:
        subm_dependency = subm_dependency_all['classics']
    else:
        subm_dependency = None
        subm_cbricks_d = None
        subm_meters_d = None

    subm_rev_map = {}
    if subm_dependency is not None:
        for k, subms in subm_dependency.items():
            for s in subms:
                subm_rev_map[s] = k

    for s in tqdm(set_paths):
        base, dir = os.path.split(s)
        set_id = int(dir)
        if subm_dependency is not None:
            if set_id in subm_rev_map.keys():
                main_id = subm_rev_map[set_id]
                result_list_this, subm_cbrick, subm_meters = evaluate_set(opt, model, s, meters, is_subm=True)
                # This set corresponds to a submodule and will be leveraged in other sets.
                subm_cbricks_d[main_id].append(subm_cbrick)
                # only in this set a submodule is used in multiple steps
                if 'architecture' in opt.dataset_alias and main_id == 7:
                    subm_cbricks_d[main_id].append(subm_cbrick)
                subm_meters_d[main_id].merge_from(subm_meters)
            else:
                if set_id in subm_dependency.keys():
                    result_list_this = evaluate_set(opt, model, s, meters, subm_cbricks=subm_cbricks_d[set_id],
                                                    subm_meters=subm_meters_d[set_id])
                else:
                    result_list_this = evaluate_set(opt, model, s, meters)
        else:
            result_list_this = evaluate_set(opt, model, s, meters)
        result_list.append(result_list_this)

        if opt.output_pred_json:
            gt_json_path = os.path.join(data_dir, s, 'info.json')
            with open(gt_json_path) as f:
                d = json.load(f)

            json_dir = os.path.join(output_dir, s)
            mkdirs(json_dir)

            if opt.autoregressive_inference or opt.single_json:
                subm_cbricks = None if subm_dependency is None else subm_cbricks_d[set_id]
                d_new = replace_poses(opt, d, result_list_this, replace_per_step=False, subm_cbricks=subm_cbricks)
                json_path = os.path.join(json_dir, 'info.json')
                d_new['gt_json_path'] = gt_json_path
                with open(json_path, 'w') as f:
                    json.dump(d_new, f, indent=4)

                if opt.render_pred_json:
                    imgs = render_dict_simple(d_new, no_check=True)
            else:
                imgs = []
                d_news = replace_poses(opt, d, result_list_this, replace_per_step=True)
                for i, d_new in enumerate(d_news):
                    d_new['gt_json_path'] = gt_json_path
                    json_path = os.path.join(json_dir, f'info_{str(i).zfill(3)}.json')
                    with open(json_path, 'w') as f:
                        json.dump(d_new, f, indent=4)
                    if opt.render_pred_json:
                        img = render_dict_simple(d_new, only_final=True)
                        imgs.append(img)

            if opt.render_pred_json:
                for i, img in enumerate(imgs):
                    img = img.resize((512, 512))
                    img_fname = f'{i:03d}.png'
                    shutil.copy(os.path.join(data_dir, s, img_fname), os.path.join(json_dir, img_fname))
                    img_fname = img_fname.replace('.png', '_predict.png')
                    img.save(os.path.join(json_dir, img_fname))

    print_keys = ['brickwise_acc', 'stepwise_acc', 'mtc_norm']
    for name, v in meters.avg_dict().items():
        if name in print_keys:
            print('test/' + name, ':', v)
        print('test/' + name, ':', v, file=fres)

    fres.close()
