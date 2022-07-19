import copy
import itertools
import os
from collections import OrderedDict, defaultdict
from itertools import chain

import numpy as np
import torch
import trimesh.transformations as tr
from PIL import Image
from torch import nn, optim

from bricks.brick_info import (
    get_all_brick_ids, BricksPC, CBrick, add_cbrick_to_bricks_pc, get_brick_enc_voxel_info
)
from datasets.legokps_shape_cond_dataset import id2euler
from lego.utils.inference_utils import dist_matching_greedy, compute_xz_to_3d_translation
from lego.utils.inference_utils import search_valid_3d_translation
from tu.ddp import get_distributed_model, master_only
from .base_model import BaseModel
from .heatmap.models.decode import lego_decode_bid
from .heatmap.utils.vis import gen_colormap, blend_img
from .utils import AccumGrad

DEBUG_FMAP = bool(os.getenv("DEBUG_FMAP", 0) == '1')


def remove_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = nn.Identity()
    for name, child in module.named_children():
        module_output.add_module(name, remove_batchnorm(child))
    del module
    return module_output


class HourglassTransModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.set_defaults(batch_size=64, lr=1e-4, display_ncols=7, niter_decay=0,
        #                     dataset_mode='clevr', niter=int(64e6 // 7e4))

        parser.set_defaults(batch_size=2, lr=1e-3, niter_decay=0,
                            dataset_mode='thermometer', niter=200, lr_policy='step', lr_decay_iters=3)

        parser.add_argument('--pretrain_encoder', action='store_true', help='Use pretrained encoder')
        parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder')

        parser.add_argument('--hidden_dim', type=int, default=64, help='Dim of hidden layer')
        parser.add_argument('--resnet_layer2', action='store_true', help='Use layer2 of ResNet in the encoder')
        parser.add_argument('--grid_size', type=int, default=21, help='Grid size')
        parser.add_argument('--lbd_t', type=float, default=1, help='Weight of the translation loss')
        parser.add_argument('--lbd_r', type=float, default=1, help='Weight of the rotaiton loss')
        parser.add_argument('--lbd_q', type=float, default=1, help='Weight of the object quaternion loss')
        parser.add_argument('--lbd_h', type=float, default=1, help='Weight of the heatmap loss')
        parser.add_argument('--lbd_o', type=float, default=1, help='Weight of the offset loss')

        parser.add_argument('--occ_color', action='store_true', help='Use colored voxel')
        parser.add_argument('--voxel_brick', action='store_true', help='Use voxel for brick representation')
        parser.add_argument('--gt_q', action='store_true', help='Use ground truth object quaternion')
        parser.add_argument('--train_obj_pose', action='store_true', help='Train object pose')
        parser.add_argument('--train_brick_pose', action='store_true', help='Train brick pose')
        parser.add_argument('--pretrain_obj_pose', default='', type=str, help='Use pretrained obj posedecoder ckpt')
        parser.add_argument('--num_features', type=int, default=64, help='Feature dims')
        parser.add_argument('--normalize_t', action='store_true', help='Normalize translation coords')
        parser.add_argument('--backbone_fmap_size', type=int, default=64, help='Feature dims')
        parser.add_argument('--occ_out_channels', type=int, default=8, help='Feature dims')
        parser.add_argument('--box_detections_per_img', type=int, default=5, help='Feature dims')
        parser.add_argument('--occ_fmap_size', type=int, default=64, help='occupancy projection dim')
        parser.add_argument('--brick_emb_dim', type=int, default=64, help='Feature dims of the brick embedding')
        parser.add_argument('--one_hot_brick_emb', action='store_true',
                            help='Use one hot embedding to encode the bricks')
        parser.add_argument('--num_bricks_single_forward', type=int, default=1,
                            help='Number of brick voxels in a single forward.')
        parser.add_argument('--brick_voxel_embedding_dim', type=int, default=0,
                            help='If nonzero, use learnable embedding for voxel rather than single values')
        parser.add_argument('--projection_brick_encoder', action='store_true', help='use projection type brick encoder')

        parser.add_argument('--mse_loss', action='store_true', help='mse loss rather than focal loss')
        parser.add_argument('--num_stacks', type=int, default=1, help='number of stacks in hourglass')
        parser.add_argument('--reg_offset', action='store_true', help='regression for the offset')

        parser.add_argument('--gt_rot', action='store_true', help='Use gt rotation instead')

        parser.add_argument('--grad_ckpt', action='store_true', help='Gradient Checkpointing')
        parser.add_argument('--freeze_networks', action='store_true')
        parser.add_argument('--eval_during_training', action='store_true')

        parser.set_defaults(predict_trans=True)
        parser.add_argument('--predict_trans_separate', action='store_true')
        parser.add_argument('--search_xz', action='store_true')
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel

        self.model_names = [
            'HG'
        ]

        self.loss_names = ['sum', 'hm', 'ce_rot', 'l1_off', 'l1_trans']

        if opt.brick_voxel_embedding_dim > 0:
            brick_voxel_embedding_num = len(get_brick_enc_voxel_info()[2]) + 2  # plus keypoint and empty voxel
            brick_voxel_embedding_dim = opt.brick_voxel_embedding_dim
        else:
            brick_voxel_embedding_num = 0
            brick_voxel_embedding_dim = 0

        from .heatmap.lego_hg import LegoHourGlass
        self.netHG = LegoHourGlass(
            opt=opt,
            num_classes=len(get_all_brick_ids()),
            occ_out_channels=opt.occ_out_channels,
            occ_fmap_size=opt.occ_fmap_size,
            shape_condition=True,
            brick_emb=opt.brick_emb_dim,
            one_hot_brick_emb=opt.one_hot_brick_emb,
            num_bricks_single_forward=opt.num_bricks_single_forward,
            brick_voxel_embedding_num=brick_voxel_embedding_num,
            brick_voxel_embedding_dim=brick_voxel_embedding_dim,
            projection_brick_encoder=opt.projection_brick_encoder,
            num_rots=7 if opt.symmetry_aware_rotation_label else 4,
            predict_masks=False,
            predict_trans=True,
            predict_trans_seperate=opt.predict_trans_separate,
        )

        if len(opt.gpu_ids) > 0:
            assert (torch.cuda.is_available())
            assert len(opt.gpu_ids) == 1, opt.gpu_ids
            if torch.distributed.is_initialized():
                assert opt.gpu_ids[0] == torch.distributed.get_rank() == opt.local_rank
                self.netHG.to(opt.local_rank)
                #     self.netHG = torch.nn.DataParallel(self.netHG, opt.gpu_ids)  # multi-GPUs
                self.netHG = nn.SyncBatchNorm.convert_sync_batchnorm(self.netHG)
                print('Converting model to ddp.')
                self.netHG = get_distributed_model(self.netHG, rank=opt.local_rank)
                print('Converted model to ddp.')
            else:
                print('torch ddp not initialized')
                # self.netHG = remove_batchnorm(self.netHG)
                self.netHG.to(opt.gpu_ids[0])

        if opt.freeze_networks:
            for p in self.netHG.parameters():
                p.requires_grad = False
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        if self.isTrain:  # only defined during training time
            parameters = []
            for name in self.model_names:
                net = getattr(self, 'net' + name)
                parameters.append(net.parameters())
            # self.optimizer = optim.SGD(chain(*parameters), lr=opt.lr, momentum=0.9, weight_decay=0.0005)
            self.optimizer = optim.AdamW(chain(*parameters), lr=opt.lr)
            if opt.acc_grad > 1:
                self.optimizer = AccumGrad(self.optimizer, opt.acc_grad)
            self.optimizers = [self.optimizer]

        self.opt = opt

    def setup(self, opt):
        super().setup(opt)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        input, targets = input  # extract mask r-cnn related input
        self.img = input['img'].to(self.device)
        self.img_path = input['img_path']
        self.occ_prev = input['obj_occ_prev'].to(self.device).float().unsqueeze(dim=1)
        self.obj_q = input['obj_q'].to(self.device).float()
        self.obj_r = input['obj_r'].to(self.device).float()
        # self.b_id = input['b_id'].to(self.device)
        self.azims = torch.as_tensor(input['azim']).to(self.device).reshape(-1).float()
        self.elevs = torch.as_tensor(input['elev']).to(self.device).reshape(-1).float()
        self.hms = [hm.to(self.device) for hm in input['hm']]

        self.offset_regs = [reg.to(self.device) for reg in input['reg']]
        self.offset_regs_mask = [m.to(self.device) for m in input['reg_mask']]
        self.bids = [b.to(self.device) for b in input['bid']]
        self.rots = [r.to(self.device).long() for r in input['rot']]
        self.inds = [i.to(self.device).long() for i in input['ind']]

        self.offset_regs = [reg.to(self.device) for reg in input['reg']]
        self.offset_regs_mask = [m.to(self.device) for m in input['reg_mask']]
        self.bids = [b.to(self.device) for b in input['bid']]
        self.rots = [r.to(self.device).long() for r in input['rot']]
        self.inds = [i.to(self.device).long() for i in input['ind']]

        self.offset_regs_flat = input['reg_flat'].to(self.device)
        self.offset_regs_mask_flat = input['reg_mask_flat'].to(self.device)
        self.bids_flat = input['bid_flat'].to(self.device)
        self.rots_flat = input['rot_flat'].to(self.device).long()
        self.inds_flat = input['ind_flat'].to(self.device).long()
        self.trans_flat = input['trans_flat'].to(self.device)

        self.kps_target = input['kp']
        self.trans = [targets[i]['trans'] for i in range(len(targets))]
        if self.opt.load_bricks:
            self.op_types = [targets[i]['op_type'] for i in range(len(targets))]
            self.bricks = input['bricks']
        self.obj_scales = input['obj_scale'].to(self.device)
        self.obj_centers = input['obj_center'].to(self.device)
        self.brick_occs = [b.to(self.device) for b in input['brick_occs']]
        self.bid_counter = input['bid_counter']
        self.reorder_map = input['reorder_map']
        self.cbricks = input['cbrick']

    def forward(self):
        target = {
            'hm': self.hms,
            'reg': self.offset_regs_flat,
            'reg_mask': self.offset_regs_mask_flat,
            'rot': self.rots_flat,
            'ind': self.inds_flat,
            'trans': self.trans_flat,
        }

        if self.opt.grad_ckpt:
            def create_forward(module, outputs_list):
                def forward(*inputs):
                    outputs, loss_sum, loss_dict = module(*inputs[:-1], target, self.brick_occs, self.bids)
                    outputs_list.extend([outputs, loss_dict])
                    return loss_sum

                return forward

            outputs_list = []
            dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
            self.loss_sum = torch.utils.checkpoint.checkpoint(create_forward(self.netHG, outputs_list),
                                                              self.img, self.occ_prev,
                                                              self.azims, self.elevs, self.obj_q, self.obj_scales,
                                                              self.obj_centers, dummy_tensor
                                                              )
            self.outputs = outputs_list[0]
            loss_dict = outputs_list[1]

        else:
            self.outputs, self.loss_sum, loss_dict = self.netHG(self.img, self.occ_prev,
                                                                self.azims, self.elevs, self.obj_q, self.obj_scales,
                                                                self.obj_centers,
                                                                target, self.brick_occs, self.bids)

        for k, v in loss_dict.items():
            if k == 'loss_sum':
                continue
            setattr(self, k, v)

        self.masks_inst = None

        if not self.netHG.training or self.opt.eval_during_training:
            self.detections = []
            with torch.no_grad():
                bids_tmp = torch.zeros_like(self.bids_flat)
                for i in range(self.bids_flat.shape[0]):
                    ct = 0
                    for j in range(self.bids_flat.shape[1]):
                        if self.offset_regs_mask_flat[i][j] == 0:
                            break
                        if j > 0 and self.bids_flat[i][j] != self.bids_flat[i][j - 1]:
                            ct += 1
                        bids_tmp[i][j] = ct

                detections = lego_decode_bid(self.outputs[-1]['hm'],
                                             self.outputs[-1]['rot'],
                                             bids_tmp,
                                             self.offset_regs_mask_flat,
                                             self.outputs[-1]['reg'],
                                             self.outputs[-1]['trans'])
                for i in range(len(detections)):
                    # map back to the bid
                    for j in range(detections[i].shape[0]):
                        detections[i][j, 3] = self.bid_counter[i][int(detections[i][j, 3].item())][0]

                sum_dict = defaultdict(float)
                obj_sum = 0
                rot_lists = []
                for i in range(len(self.hms)):
                    num_objs = self.offset_regs_mask[i].sum().item()
                    obj_sum += num_objs
                    kps_pred = detections[i][:num_objs, :2]

                    kps_target_list = []
                    bid_list = []
                    rot_list = []
                    reorder_idxs_ = []  # reorder the list according to the original brick order
                    for j, (bid, ct) in enumerate(self.bid_counter[i]):
                        kps_target_list.append(self.kps_target[i][j, :ct])
                        bid_list += [bid] * ct
                        rot_list.append(self.rots[i][j, :ct])
                        reorder_idxs_ += [self.reorder_map[i][(bid, k)] for k in range(ct)]

                    reorder_idxs = [0] * len(reorder_idxs_)
                    for m, n in enumerate(reorder_idxs_):
                        reorder_idxs[n] = m

                    kps_target_list = torch.cat(kps_target_list, dim=0).to(kps_pred.device)[reorder_idxs]
                    bid_list = torch.as_tensor(bid_list).to(kps_pred.device)[reorder_idxs]
                    rot_list = torch.cat(rot_list, dim=0).to(kps_pred.device)[reorder_idxs]
                    rot_lists.append(rot_list)

                    idxs_pred, dists_matched = dist_matching_greedy(kps_pred, kps_target_list)
                    detection_matched = detections[i][idxs_pred]

                    sum_dict['kp_mse'] += dists_matched.sum()
                    sum_dict['kp_acc'] += (dists_matched == 0).sum()
                    sum_dict['kp_acc@tol2'] += (dists_matched <= 4).sum()
                    sum_dict['kp_acc@tol4'] += (dists_matched <= 16).sum()
                    sum_dict['kp_acc@tol8'] += (dists_matched <= 64).sum()
                    sum_dict['cls_acc'] += (detection_matched[:, 3] == bid_list).sum().item()
                    sum_dict['rot_acc'] += (detection_matched[:, 4] == rot_list).sum().item()

                    from datasets.definition import gdef
                    if self.opt.normalize_trans:
                        step_trans = detection_matched[:, 5:] * gdef.translation_std.to(
                            self.device) + gdef.translation_mean.to(self.device)
                    else:
                        step_trans = detection_matched[:, 5:]
                    step_trans = (step_trans * 2).round() / 2
                    self.detections.append({
                        'kp': detection_matched[:, :2],
                        'bid': detection_matched[:, 3],
                        'rot': detection_matched[:, 4],
                        'trans': step_trans.cpu().numpy(),
                    })

                for i in range(len(self.hms)):
                    step_trans = self.detections[i]['trans']
                    self.detections[i]['trans_orig'] = np.array(step_trans)
                    if self.opt.load_bricks:
                        num_objs = self.offset_regs_mask[i].sum().item()
                        bs = copy.deepcopy(self.bricks[i])
                        for j in range(num_objs):
                            bid_this = self.detections[i]["bid"][j].long().item()

                            if bid_this < 0:
                                bid_decoded = 'CBrick#' + str(-bid_this - 1)
                                cbrick = self.cbricks[i][-bid_this - 1]
                            else:
                                all_brick_ids = get_all_brick_ids()
                                bid_decoded = all_brick_ids[bid_this]
                                cbrick = None

                            if not self.opt.gt_rot:
                                rot_decoded = id2euler(self.detections[i]['rot'][j].long().item(),
                                                       symmetry_aware_label=self.opt.symmetry_aware_rotation_label)
                            else:
                                rot_decoded = id2euler(rot_lists[i][j],
                                                       symmetry_aware_label=self.opt.symmetry_aware_rotation_label)

                            rot_decoded = tr.quaternion_from_euler(*(np.array(rot_decoded) * np.pi / 180))
                            if self.opt.search_xz:
                                search_coord_delta = [0, 0.5, -0.5]
                                x = step_trans[j][0]
                                z = step_trans[j][2]
                                xzs = []
                                for m in range(len(search_coord_delta)):
                                    for n in range(len(search_coord_delta)):
                                        xzs.append(
                                            (x + search_coord_delta[m], z + search_coord_delta[n]))

                                # Make sure candidates are sorted according to the deviation from the original prediction
                                xzs.sort(key=lambda p: (p[0] - x) ** 2 + (p[1] - z) ** 2)
                                xs, zs = zip(*xzs)

                                candidates, pos_kp_offset = compute_xz_to_3d_translation(
                                    bid=bid_decoded,
                                    rot=rot_decoded,
                                    op_type=self.op_types[i][j],
                                    xs=xs,
                                    zs=zs,
                                    bricks_pc=bs,
                                    xz_type='pos' if not self.opt.top_center else 'top_center',
                                    cbrick=cbrick)

                            else:
                                search_coord_delta = [0, 0.5, -0.5, 1, -1]
                                trans_list = []
                                delta_list = list(itertools.product(*([search_coord_delta] * 3)))
                                delta_list.sort(key=lambda p: p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
                                trans = step_trans[j]
                                for m, n, l in delta_list:
                                    trans_list.append([trans[0] + m, trans[1] + n, trans[2] + l])

                                candidates, pos_kp_offset = search_valid_3d_translation(
                                    bid=bid_decoded,
                                    rot=rot_decoded,
                                    op_type=self.op_types[i][j],
                                    trans_list=trans_list,
                                    bricks_pc=bs,
                                    trans_type='pos' if not self.opt.top_center else 'top_center',
                                    cbrick=cbrick)

                            if candidates:
                                trans = candidates[0]
                                step_trans[j] = trans
                                brick_position = trans - pos_kp_offset
                                if bid_this >= 0:
                                    assert bs.add_brick(
                                        bid_decoded,
                                        brick_position,
                                        rot_decoded, self.op_types[i][j], canonical=False)
                                else:
                                    cbrick_bs = BricksPC.from_dict(cbrick._state_dict['bricks_pc'], no_check=True)
                                    cbrick_this = CBrick(cbrick_bs, brick_position, rot_decoded)
                                    assert add_cbrick_to_bricks_pc(bs, cbrick_this, self.op_types[i][j])

                    dists_trans = ((np.array(self.trans[i]) - np.array(step_trans)) ** 2).sum(axis=1)
                    dists_trans_xz = ((np.array(self.trans[i])[:, [0, 2]] - np.array(step_trans[:, [0, 2]])) ** 2).sum(
                        axis=1)
                    sum_dict['trans_mse'] += dists_trans.sum()
                    sum_dict['trans_acc'] += (dists_trans <= 1e-6).sum()
                    sum_dict['trans_acc@tol1'] += (dists_trans <= 1).sum()
                    sum_dict['trans_acc@tol2'] += (dists_trans <= 4).sum()
                    sum_dict['trans_mse_xz'] += dists_trans.sum()
                    sum_dict['trans_acc_xz'] += (dists_trans_xz <= 1e-6).sum()
                    sum_dict['trans_acc@tol1_xz'] += (dists_trans_xz <= 1).sum()
                    sum_dict['trans_acc@tol2_xz'] += (dists_trans_xz <= 4).sum()

                    num_objs = self.offset_regs_mask[i].sum().item()
                    self.detections[i]['rot_decoded'] = [None] * num_objs
                    for j in range(num_objs):
                        rot_id = self.detections[i]['rot'][j].long().item()
                        self.detections[i]['rot_decoded'][j] = id2euler(rot_id,
                                                                        symmetry_aware_label=self.opt.symmetry_aware_rotation_label)
                self.eval_outputs = {}
                for k, v in sum_dict.items():
                    self.eval_outputs[k] = v / obj_sum

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        if not self.netHG.training or self.opt.eval_during_training:
            errors_ret.update(self.eval_outputs)
        return errors_ret

    @torch.no_grad()
    def compute_visuals(self, n_vis=5):
        stacks_to_show = [-1]
        n_vis = min(n_vis, self.img.shape[0])
        self.vis_dict = defaultdict(list)
        theme = 'black'
        for s in stacks_to_show:
            output = self.outputs[s]
            for i in range(n_vis):
                img = np.array(Image.open(self.img_path[i]).convert('RGB'))
                pred = gen_colormap(output['hm'][i].squeeze(1).detach().cpu().numpy(), theme=theme)
                gt = gen_colormap(self.hms[i].detach().cpu().numpy(), theme=theme)
                self.vis_dict['img'].append(img)
                self.vis_dict['hm_pred'].append(blend_img(img, pred, theme=theme))
                self.vis_dict['hm_gt'].append(blend_img(img, gt, theme=theme))
                if not self.netHG.training:
                    hm_pred_sep = []
                    for j in range(self.hms[i].shape[0]):
                        pred_sep = gen_colormap(output['hm'][i][[j]].squeeze(1).detach().cpu().numpy(), theme=theme)
                        hm_pred_sep.append(blend_img(img, pred_sep))
                    self.vis_dict['hm_pred_sep'].append(hm_pred_sep)

    def get_current_visuals(self):
        return self.vis_dict

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        loss = self.loss_sum
        if not self.opt.freeze_networks:
            loss.backward()

    def optimize_parameters(self, sync=True):
        """Update network weights; it will be called in every training iteration."""
        if not torch.distributed.is_initialized() or self.opt.acc_grad == 1 or sync:
            self.forward()  # first call forward to calculate intermediate results
            self.backward()  # calculate gradients for network G
        else:
            # Only sync the gradients at the last step when using gradient accumulation
            # Assuming there is only one model!
            with getattr(self, 'net' + self.model_names[0]).no_sync():
                self.forward()  # first call forward to calculate intermediate results
                self.backward()  # calculate gradients for network G
        if self.opt.clip > 0:
            for name in self.model_names:
                if isinstance(name, str):
                    net = getattr(self, 'net' + name)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), self.opt.clip)
        self.optimizer.step()  # update gradients for network G
        self.optimizer.zero_grad()  # clear network G's existing gradients

    @master_only
    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        # super().save_networks(epoch)  # hangs with ddp + >1 gpus

        for name in self.model_names:
            save_filename = '%s_net_%s.pth' % (epoch, name)
            save_path = os.path.join(self.save_dir, save_filename)
            net = getattr(self, 'net' + name)
            torch.save(net.state_dict(), save_path)

        save_filename = '%s_optimizer.pth' % epoch
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(self.optimizers[0].state_dict(), save_path)

        save_filename = '%s_lr_scheduler.pth' % epoch
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(self.schedulers[0].state_dict(), save_path)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in he file name '%s_net_%s.pth' % (epoch, name)
        """
        # super().load_networks(epoch)
        for name in self.model_names:
            load_filename = '%s_net_%s.pth' % (epoch, name)
            load_path = os.path.join(self.save_dir, load_filename)
            net = getattr(self, 'net' + name)
            if isinstance(net, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                net = net.module
            state_dict = torch.load(load_path)
            consume_prefix_in_state_dict_if_present(state_dict, 'module.')
            print('loading net %s from %s, mode: %s' % (name, load_path, 'strict'))
            net.load_state_dict(state_dict, strict=True)

        if self.isTrain and (not self.opt.ignore_optimizer):
            load_filename = '%s_optimizer.pth' % epoch
            load_path = os.path.join(self.save_dir, load_filename)
            print('loading the optimizer from %s' % load_path)
            state_dict = torch.load(load_path, map_location=str(self.device))
            self.optimizer.load_state_dict(state_dict)

        if self.isTrain and (not self.opt.ignore_lr_scheduler):
            load_filename = '%s_lr_scheduler.pth' % epoch
            load_path = os.path.join(self.save_dir, load_filename)
            print('loading the lr scheduler from %s' % load_path)
            state_dict = torch.load(load_path, map_location=str(self.device))
            self.schedulers[0].load_state_dict(state_dict)


# taken from newer pytorch
def consume_prefix_in_state_dict_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix):]
            state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix):]
            metadata[newkey] = metadata.pop(key)
