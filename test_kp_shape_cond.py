import collections
import os
import time

import matplotlib;
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from datasets import create_dataset
from models import create_model
from options.test_options import TestOptions
from util.html_table import HTMLTableVisualizer, HTMLTableColumnDesc
from util.util import mkdirs

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coco_related import visualize
from bricks.brick_info import get_all_brick_ids, get_brick_annotation
from datasets.legokps_shape_cond_dataset import id2euler
from debug.utils import visualize_bricks
import pprint

if __name__ == '__main__':
    opt = TestOptions().parse()  # get training options
    opt.display_id = -1
    opt.display_port = -1
    vis_ct = 0
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.

    print('The number of testing examples = %d' % dataset_size)

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    total_iters = 0  # the total number of training iterations

    print('Dataset size:', len(dataset))
    model.eval()

    epoch_iter = 0

    losses_sum = collections.defaultdict(float)
    losses_count = collections.defaultdict(int)
    losses_test = collections.defaultdict(float)

    result_dir = os.path.join(opt.results_dir, opt.name, opt.dataset_alias)
    mkdirs(result_dir)
    vis_dir = os.path.join(result_dir, 'vis')
    mkdirs(vis_dir)
    fres = open(os.path.join(result_dir, 'result.txt'), 'w')
    res_all = []

    res_d = collections.defaultdict(list)

    if opt.n_vis > 0:
        img_axis = Image.open('data/axis.png')
        vis = HTMLTableVisualizer(os.path.join(result_dir, 'visualization'), 'Lego Visualization')
        vis.begin_html()
    from collections import defaultdict, Counter

    wrong_ct = defaultdict(Counter)

    for i, data in tqdm(enumerate(dataset), total=len(dataset) // opt.batch_size):  # inner loop within one epoch

        iter_start_time = time.time()  # timer for computation per iteration
        total_iters += opt.batch_size
        epoch_iter += opt.batch_size
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.test()

        losses = model.get_current_losses()
        data, targets = data
        for name, v in losses.items():
            obj_sum = sum(data['reg_mask'][i].sum().item() for i in range(len(data['reg_mask'])))
            losses_sum[name] += v * obj_sum
            losses_count[name] += obj_sum

        if vis_ct < opt.n_vis:
            # Compute 2d position of precited points

            model.compute_visuals(opt.n_vis - vis_ct)
            for i in range(data['img'].shape[0]):
                with vis.table('Visualize #{}'.format(vis_ct), [
                    HTMLTableColumnDesc('lego_id', 'Lego ID', 'text', {'width': '50px'}),
                    HTMLTableColumnDesc('img_id', 'Image ID', 'text', {'width': '50px'}),
                    HTMLTableColumnDesc('img_box', 'Image', 'figure', {'width': '768'}),
                    HTMLTableColumnDesc('img_hm_gt', 'GT heatmap', 'image', {'width': '768'}),
                    HTMLTableColumnDesc('img_hm_pred', 'Predicted heatmap', 'image', {'width': '768'}),
                    HTMLTableColumnDesc('img', 'Image', 'image', {'width': '768'}),
                    HTMLTableColumnDesc('img_prev', 'Previous Image', 'image', {'width': '512px'}),
                    HTMLTableColumnDesc('img_axis', 'Axis', 'image', {'width': '200px'}),
                ]):
                    img_path = data['img_path'][i]
                    img_fname = os.path.basename(img_path)
                    img_dir = os.path.dirname(img_path)
                    if '_' in img_fname:
                        img_id = int(img_fname[:-9])
                    else:
                        img_id = int(img_fname[:-4])
                    img_prev_fname = '{}.png'.format(str(img_id - 1).zfill(3))
                    if opt.load_lpub:
                        img_prev_fname = img_prev_fname.replace('.png', '_lpub.png')
                    img_prev_path = os.path.join(img_dir, img_prev_fname)
                    from PIL import Image

                    if img_id == 0:
                        img_prev = Image.new('RGB', (1, 1))
                    else:
                        img_prev = Image.open(img_prev_path)

                    obj_id = img_path.split('/')[-2:]

                    kps_target_list = []
                    bid_list = []
                    rot_list = []
                    bbox_list = []
                    reorder_idxs_ = []  # reorder the list according to the original brick order

                    for j, (bid, ct) in enumerate(model.bid_counter[i]):
                        kps_target_list.append(model.kps_target[i][j, :ct])
                        bid_list += [bid] * ct
                        rot_list.append(model.rots[i][j, :ct])
                        bbox_list.append(data['bbox'][i][j, :ct])
                        reorder_idxs_ += [data['reorder_map'][i][(bid, k)] for k in range(ct)]

                    reorder_idxs = [0] * len(reorder_idxs_)
                    for m, n in enumerate(reorder_idxs_):
                        reorder_idxs[n] = m

                    kps_target_list = torch.cat(kps_target_list, dim=0)[reorder_idxs]
                    bid_list = torch.as_tensor(bid_list)[reorder_idxs]
                    rot_list = torch.cat(rot_list, dim=0)[reorder_idxs]
                    bbox_list = torch.cat(bbox_list, dim=0)[reorder_idxs]

                    img = Image.fromarray(model.vis_dict['img'][i])
                    img_hm_pred = Image.fromarray(model.vis_dict['hm_pred'][i])
                    img_hm_gt = Image.fromarray(model.vis_dict['hm_gt'][i])

                    gt_boxes = bbox_list.cpu().numpy()
                    gt_boxes = gt_boxes[:, [1, 0, 3, 2]]  # visualize script adopts [y0, x0, y1, x1]
                    fig_box, ax = plt.subplots(figsize=(20, 20))

                    if opt.load_mask:
                        masks = targets[i]['masks'].cpu().numpy()
                    visualize.draw_boxes(np.array(img), refined_boxes=gt_boxes,
                                         captions=list(map(str, range(gt_boxes.shape[0]))), ax=ax)

                    vis.row(lego_id=obj_id[0], img_id=obj_id[1],
                            img_prev=img_prev, img=img, img_hm_pred=img_hm_pred, img_hm_gt=img_hm_gt, img_axis=img_axis,
                            img_box=fig_box)
                    plt.close(fig='all')

                if opt.load_mask:
                    fig_mask_gt, ax = plt.subplots(figsize=(20, 20))
                    gt_masks = targets[i]['masks']
                    gt_masks = gt_masks.cpu().numpy()
                    visualize.draw_boxes(np.array(img), masks=gt_masks,
                                         captions=list(map(str, range(gt_boxes.shape[0]))), ax=ax)

                    from models.utils import expand_mask

                    fig_mask, ax = plt.subplots(figsize=(20, 20))
                    num_ins = model.masks_inst_gt[i].max()
                    masks = expand_mask(model.masks_inst[i], num_classes=num_ins + 1)[1:]
                    masks = torch.nn.functional.interpolate(masks[None].float(), (512, 512),
                                                            mode='bilinear')
                    masks = masks[0].permute((1, 2, 0)).long()
                    masks = masks.cpu().numpy()
                    visualize.draw_boxes(np.array(img), refined_boxes=gt_boxes, masks=masks,
                                         captions=list(map(str, range(gt_boxes.shape[0]))), ax=ax)

                    with vis.table(f'Masks #{j}', [
                        HTMLTableColumnDesc('masks', 'Brick Masks', 'figure', {'width': '512px'}),
                    ]):
                        vis.row(masks=fig_mask_gt)
                        vis.row(masks=fig_mask)

                num_objs = model.detections[i]['kp'].shape[0]
                gt_kps_all = kps_target_list
                pred_kps_all = model.detections[i]["kp"].long().cpu()
                dists = ((gt_kps_all[:num_objs, None, :] - pred_kps_all[None, :, :]) ** 2).sum(dim=-1)
                closest_gt_idxs = dists.argmin(dim=0).numpy()

                with vis.table(f'Prediction heatmap info #{j}', [
                    HTMLTableColumnDesc('hm', 'Invidividual Heatmap', 'image', {'width': '512px'}),
                ]
                               ):
                    for hm in model.vis_dict['hm_pred_sep'][i]:
                        vis.row(hm=Image.fromarray(hm))

                if model.cbricks[i]:
                    with vis.table(f'CBricks Info', [
                        HTMLTableColumnDesc('cbrick_id', 'ID', 'code', {}),
                        HTMLTableColumnDesc('cbrick_img', 'CBrick Image', 'image', {'width': '300px'}),
                        HTMLTableColumnDesc('bricks', 'Bricks', 'code', {}),
                    ]):
                        for cbrick_id, cbrick in enumerate(model.cbricks[i]):
                            vis.row(cbrick_id=cbrick_id,
                                    cbrick_img=visualize_bricks(cbrick.bricks_raw),
                                    bricks=pprint.pformat(cbrick.bricks_raw))

                for j in range(model.detections[i]['kp'].shape[0]):
                    with vis.table(f'GT/Pred bricks info #{j}', [
                        HTMLTableColumnDesc('brick_img', 'Brick Image', 'image', {'width': '200px'}),
                        HTMLTableColumnDesc('kp_img', 'GT/Pred keypoints Zoom', 'image', {'width': '200px'}),
                        HTMLTableColumnDesc('kp_all_img', 'GT/Pred keypoints', 'image', {'width': '200px'}),
                        HTMLTableColumnDesc('kp_closest_img', 'Pred and closest GT keypoint', 'image',
                                            {'width': '200px'}),
                        HTMLTableColumnDesc('info', 'Info', 'code', {}),
                    ]
                                   ):

                        bbox = bbox_list[j].cpu().numpy()
                        img_box = img.copy()
                        draw = ImageDraw.Draw(img_box)
                        draw.rectangle(list(bbox))
                        bbox += [-20, -20, 20, 20]
                        bbox = bbox.clip(0, 512 - 1)

                        img_kp = img.copy()
                        draw = ImageDraw.Draw(img_kp)
                        point_size = np.array([3, 3])

                        gt_kps = gt_kps_all[j].cpu().tolist()
                        pred_kps = model.detections[i]["kp"][j].long().cpu().tolist()
                        draw.pieslice(
                            (pred_kps[0] - point_size[0], pred_kps[1] - point_size[1], pred_kps[0] + point_size[0],
                             pred_kps[1] + point_size[1]),
                            0, 360, fill='#ff0000', width=2)
                        draw.pieslice(
                            (gt_kps[0] - point_size[0], gt_kps[1] - point_size[1], gt_kps[0] + point_size[0],
                             gt_kps[1] + point_size[1]),
                            0, 360, fill='#00ff00', width=2)
                        brick_img = img_box.crop(bbox)
                        kp_img = img_kp.crop(bbox)
                        kp_all_bbox = np.array([min(gt_kps[0], pred_kps[0]), min(gt_kps[1], pred_kps[1]),
                                                max(gt_kps[0], pred_kps[0]), max(gt_kps[1], pred_kps[1]),
                                                ])
                        kp_all_bbox += [-40, -40, 40, 40]
                        kp_all_bbox = kp_all_bbox.clip(0, 512 - 1)
                        kp_all_img = img_kp.crop(kp_all_bbox)

                        gt_closest_kps = gt_kps_all[closest_gt_idxs[j]].cpu().tolist()
                        img_kp_closest = img.copy()
                        draw = ImageDraw.Draw(img_kp_closest)
                        draw.pieslice(
                            (pred_kps[0] - point_size[0], pred_kps[1] - point_size[1], pred_kps[0] + point_size[0],
                             pred_kps[1] + point_size[1]),
                            0, 360, fill='#ff0000', width=2)
                        draw.pieslice(
                            (gt_closest_kps[0] - point_size[0], gt_closest_kps[1] - point_size[1],
                             gt_closest_kps[0] + point_size[0], gt_closest_kps[1] + point_size[1]),
                            0, 360, fill='#0000ff', width=2)
                        kp_closest_bbox = np.array(
                            [min(gt_closest_kps[0], pred_kps[0]), min(gt_closest_kps[1], pred_kps[1]),
                             max(gt_closest_kps[0], pred_kps[0]), max(gt_closest_kps[1], pred_kps[1]),
                             ])
                        kp_closest_bbox += [-40, -40, 40, 40]
                        kp_closest_bbox = kp_closest_bbox.clip(0, 512 - 1)
                        kp_closest_img = img_kp_closest.crop(kp_closest_bbox)

                        info = ''
                        info += f'GT Keypoint Position: {gt_kps}\n'
                        info += f'Pred Keypoint Position: {pred_kps}\n'
                        info += f'Correct Keypoint Position: {(gt_kps == pred_kps)}\n'
                        kps_dist = ((np.array(gt_kps) - np.array(pred_kps)) ** 2).sum()
                        info += f'Pixel L2 Dist <= 2: {kps_dist <= 4}\n'
                        info += f'Pixel L2 Dist <= 4: {kps_dist <= 16}\n'
                        info += f'Pixel L2 Dist <= 8: {kps_dist <= 64}\n'


                        def get_brick_name(bid):
                            if bid < 0:
                                return 'CBrick #' + str(-bid - 1)
                            else:
                                return get_all_brick_ids()[bid]


                        gt_btype = get_brick_name(bid_list[j].long().item())
                        pred_btype = get_brick_name(model.detections[i]["bid"][j].long().item())
                        info += f'GT Brick type: {gt_btype} ({get_brick_annotation(gt_btype)})\n'
                        info += f'Pred Brick type: {pred_btype} ({get_brick_annotation(pred_btype)})\n'
                        info += f'Correct Brick type: {pred_btype == gt_btype}\n'
                        gt_rot = id2euler(rot_list[j].item(), symmetry_aware_label=opt.symmetry_aware_rotation_label)
                        pred_rot = id2euler(model.detections[i]['rot'][j].long().item(),
                                            symmetry_aware_label=opt.symmetry_aware_rotation_label)
                        info += f'GT Rotation: {gt_rot}\n'
                        info += f'Pred Rotation: {pred_rot}\n'
                        if 'rot_orig' in model.detections[i]:
                            pred_rot_orig = id2euler(model.detections[i]['rot_orig'][j].long().item(),
                                                     symmetry_aware_label=opt.symmetry_aware_rotation_label)
                            info += f'Original Pred Rotation: {pred_rot_orig}\n'
                        info += f'Correct Rotation: {pred_rot == gt_rot}\n'
                        gt_trans = targets[i]['trans'][j]
                        pred_trans = model.detections[i]['trans'][j]
                        trans_dist = ((np.array(gt_trans) - np.array(pred_trans)) ** 2).sum()
                        info += f'GT 3d Translation: {gt_trans}\n'
                        info += f'Pred 3d Translation: {pred_trans}\n'
                        if 'trans_orig' in model.detections[i]:
                            pred_trans_orig = model.detections[i]['trans_orig'][j]
                            info += f'Original Pred 3d Translation: {pred_trans_orig}\n'
                        info += f'3d Trans L2 Dist <= 1: {trans_dist <= 1}\n'
                        info += f'3d Trans L2 Dist <= 2: {trans_dist <= 4}\n'
                        info += f'Correct 3d Translation: {np.allclose(pred_trans, gt_trans)}'

                        info = info.replace('False', '<b>False</b>')
                        vis.row(brick_img=brick_img,
                                kp_img=kp_img, kp_all_img=kp_all_img, kp_closest_img=kp_closest_img,
                                info=info)

                vis_ct += 1
                if vis_ct >= opt.n_vis:
                    break

    if opt.n_vis > 0:
        with vis.table('Summary', [HTMLTableColumnDesc('item', 'Item', 'code', {})]):
            for name, v in losses_sum.items():
                vis.row(item='test/' + name + ':' + str(v / losses_count[name]))

        vis.end_html()

    res_all.sort(key=lambda x: -x[-1])
    for r in res_all:
        fres.write('\t'.join(map(str, r)) + '\n')

    for name, v in losses_sum.items():
        print('test/' + name, ':', v / losses_count[name])
        print('test/' + name, ':', v / losses_count[name], file=fres)

    fres.close()
