import copy
import os
from collections import OrderedDict, defaultdict
from itertools import chain

import numpy as np
import pytorch3d.transforms as pt
import torch
import trimesh.transformations as tr
from PIL import Image
from torch import nn, optim

from bricks.brick_info import (
    get_brick_class, get_all_brick_ids, BricksPC, CBrick, add_cbrick_to_bricks_pc, get_brick_enc_voxel_info, Brick
)
from datasets.legokps_shape_cond_dataset import id2euler
from lego.utils.camera_utils import get_cameras, get_scale
from lego.utils.inference_utils import compute_2d_kp_to_3d_translation, compute_candidates_iou, recompute_conns
from lego.utils.inference_utils import dist_matching_greedy
from tu.ddp import get_distributed_model, master_only
from util.util import transform_points_screen
from .base_model import BaseModel
from .heatmap.models.decode import lego_decode_bid_shape_cond, lego_decode_bid
from .heatmap.utils.vis import gen_colormap, blend_img
from .utils import AccumGrad, assign_masks, expand_mask, compute_iou, matching_inst_mask_idxs, Meters

DEBUG_FMAP = bool(os.getenv("DEBUG_FMAP", 0) == '1')


def remove_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = nn.Identity()
    for name, child in module.named_children():
        module_output.add_module(name, remove_batchnorm(child))
    del module
    return module_output


class HourglassShapeCondModel(BaseModel):

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
        parser.add_argument('--predict_masks', action='store_true')
        parser.set_defaults(assoc_emb=True)
        parser.add_argument('--assoc_emb_lbd', type=float, default=0.1)
        parser.add_argument('--search_ref_mask', action='store_true')
        parser.add_argument('--no_coordconv', action='store_true')

        parser.add_argument('--mse_loss', action='store_true', help='mse loss rather than focal loss')
        parser.add_argument('--num_stacks', type=int, default=1, help='number of stacks in hourglass')
        parser.add_argument('--reg_offset', action='store_true', help='regression for the offset')

        parser.add_argument('--gt_rot', action='store_true', help='Use gt rotation instead')

        parser.add_argument('--grad_ckpt', action='store_true', help='Gradient Checkpointing')
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

        self.loss_names = ['sum', 'hm', 'ce_rot', 'l1_off']
        if opt.predict_masks:
            self.loss_names.extend(['mask', 'pull', 'push'])

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
            predict_masks=self.opt.predict_masks,
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

        if self.opt.num_bricks_single_forward > 1:
            self.offset_regs_flat = input['reg_flat'].to(self.device)
            self.offset_regs_mask_flat = input['reg_mask_flat'].to(self.device)
            self.bids_flat = input['bid_flat'].to(self.device)
            self.rots_flat = input['rot_flat'].to(self.device).long()
            self.inds_flat = input['ind_flat'].to(self.device).long()

        self.kps_target = input['kp']
        if self.opt.load_conns:
            self.conns = [targets[i]['conns'] for i in range(len(targets))]
            self.trans = [targets[i]['trans'] for i in range(len(targets))]
            self.op_types = [targets[i]['op_type'] for i in range(len(targets))]
        if self.opt.load_bricks:
            self.bricks = input['bricks']
        self.obj_scales = input['obj_scale'].to(self.device)
        self.obj_centers = input['obj_center'].to(self.device)
        self.brick_occs = [b.to(self.device) for b in input['brick_occs']]
        self.bid_counter = input['bid_counter']
        self.reorder_map = input['reorder_map']
        self.cbricks = input['cbrick']

        if self.opt.predict_masks:
            self.masks = torch.stack([m.long() for m in input['mask']]).to(self.device)
            self.masks_instance = [m.long().to(self.device) for m in input['mask_instance']]
            self.assoc_emb_sample_inds = [m.long().to(self.device) for m in input['assoc_emb_sample_inds']]

    def compute_instance_masks(self, kmeans_init_kps=None, reorder_idxs_list=None):
        self.masks_pred = self.outputs[-1]['mask'].argmax(1)
        self.masks_inst = torch.zeros_like(self.masks).cpu()

        for i in range(self.masks_pred.shape[0]):
            num_brick_types = self.masks[i].max()
            num_bricks = 0
            if kmeans_init_kps is not None:
                kps = kmeans_init_kps[i]
            for j in range(num_brick_types):
                n_ins = self.masks_instance[i][j].max().item()
                if kmeans_init_kps is not None:
                    kps_this = np.array(kps[num_bricks:num_bricks + n_ins].cpu())
                else:
                    kps_this = None
                masks_inst_this = assign_masks(self.outputs[-1]['assoc_emb'][i][j],
                                               (self.masks_pred[i] == j + 1).cpu().numpy(),
                                               n_ins, kps_this)
                masks_inst_this[masks_inst_this.nonzero()] += num_bricks
                num_bricks += n_ins
                self.masks_inst[i][masks_inst_this.nonzero()] = \
                    torch.as_tensor(masks_inst_this[masks_inst_this.nonzero()]).cpu().long()

        self.masks_inst_gt = torch.zeros_like(self.masks).cpu()
        for i in range(self.masks_inst_gt.shape[0]):
            masks_inst_gt = self.masks_instance[i].clone()
            num_instances = masks_inst_gt[0].max()
            for j in range(1, masks_inst_gt.shape[0]):
                masks_inst_gt[0][masks_inst_gt[j].nonzero(as_tuple=True)] = \
                    masks_inst_gt[j][masks_inst_gt[j].nonzero(as_tuple=True)] + num_instances
                num_instances += masks_inst_gt[j].max()
            self.masks_inst_gt[i] = masks_inst_gt[0]

        if self.opt.search_ref_mask:
            self.masks_inst_exp = []
            for i in range(self.masks_inst_gt.shape[0]):
                num_ins = len(reorder_idxs_list[i])
                masks = expand_mask(self.masks_inst[i], num_classes=num_ins + 1)[1:]
                masks = torch.nn.functional.interpolate(masks[None].float(), (512, 512),
                                                        mode='bilinear')
                masks = masks[0].permute((1, 2, 0)).long()
                masks = masks[:, :, reorder_idxs_list[i]]
                self.masks_inst_exp.append(masks.cpu().numpy())

    def forward(self):
        if self.opt.num_bricks_single_forward == 1:
            target = {
                'hm': self.hms,
                'reg': self.offset_regs,
                'reg_mask': self.offset_regs_mask,
                'rot': self.rots,
                'ind': self.inds,
            }
        else:
            target = {
                'hm': self.hms,
                'reg': self.offset_regs_flat,
                'reg_mask': self.offset_regs_mask_flat,
                'rot': self.rots_flat,
                'ind': self.inds_flat,
            }

        if self.opt.predict_masks:
            target['mask'] = self.masks
            target['mask_instance'] = self.masks_instance
            target['assoc_emb_sample_inds'] = self.assoc_emb_sample_inds

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

        if not self.netHG.training:
            self.detections = []
            with torch.no_grad():
                if self.opt.num_bricks_single_forward == 1:
                    detections = lego_decode_bid_shape_cond(self.outputs[-1]['hm'],
                                                            self.outputs[-1]['rot'],
                                                            self.bid_counter,
                                                            self.outputs[-1]['reg'],
                                                            )
                else:
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
                                                 self.outputs[-1]['reg'])
                    for i in range(len(detections)):
                        # map back to the bid
                        for j in range(detections[i].shape[0]):
                            detections[i][j, 3] = self.bid_counter[i][int(detections[i][j, 3].item())][0]

                sum_dict = defaultdict(float)
                obj_sum = 0
                rot_lists = []
                kps_pred_list = []
                reorder_idxs_list = []
                for i in range(len(self.hms)):
                    num_objs = self.offset_regs_mask[i].sum().item()
                    obj_sum += num_objs
                    kps_pred = detections[i][:num_objs, :2]
                    kps_pred_list.append(kps_pred / 4)  # downsample to 128x128

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

                    # Perform Hungarian matching between predicted and ground-truth keypoints
                    idxs_pred, dists_matched = dist_matching_greedy(kps_pred, kps_target_list)
                    detection_matched = detections[i][idxs_pred]
                    reorder_idxs_list.append(idxs_pred)

                    sum_dict['kp_mse'] += dists_matched.sum()
                    sum_dict['kp_acc'] += (dists_matched == 0).sum()
                    sum_dict['kp_acc@tol2'] += (dists_matched <= 4).sum()
                    sum_dict['kp_acc@tol4'] += (dists_matched <= 16).sum()
                    sum_dict['kp_acc@tol8'] += (dists_matched <= 64).sum()
                    sum_dict['cls_acc'] += (detection_matched[:, 3] == bid_list).sum().item()
                    sum_dict['rot_acc'] += (detection_matched[:, 4] == rot_list).sum().item()
                    # import ipdb; ipdb .set_trace()
                    self.detections.append({
                        'kp': detection_matched[:, :2],
                        'bid': detection_matched[:, 3],
                        'rot': detection_matched[:, 4],
                    })

                # 3D Pose inference
                if self.opt.load_conns and self.opt.load_bricks:
                    if not self.masks_inst and self.opt.search_ref_mask:
                        self.compute_instance_masks(kps_pred_list, reorder_idxs_list)
                    scale = get_scale()
                    for i in range(len(self.detections)):
                        # i is batch index
                        num_objs = self.offset_regs_mask[i].sum().item()
                        transforms = pt.Transform3d().translate(-self.obj_centers[None, i]).scale(
                            self.obj_scales[i]).cuda()
                        cameras = get_cameras(azim=self.azims[i], elev=self.elevs[i])
                        bs = copy.deepcopy(self.bricks[i])
                        bs.apply_transform(tr.quaternion_matrix(self.obj_r[i].cpu()))

                        step_trans = []
                        rot_mask_ref_list = []
                        if self.opt.search_ref_mask:
                            self.detections[i]['rot_orig'] = torch.Tensor(self.detections[i]['rot'].cpu())

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
                            if self.op_types[i][j].item() == 2:
                                l = torch.linspace(-32.5, 32.5, 131, device='cuda')
                                mg_x, mg_y = torch.meshgrid(l, l)
                                if bid_this > 0:
                                    brick_height = get_brick_class(bid_decoded).get_height()
                                else:
                                    brick_height = cbrick.get_height()
                                mg = torch.stack([mg_x, torch.ones_like(mg_x) * brick_height, mg_y], dim=-1)
                                mg = mg.reshape(-1, 1, 3)
                                mg_transformed = mg * scale[None, None, :]
                                mg_transformed = transforms.transform_points(mg_transformed)
                                kps_screen = transform_points_screen(cameras.get_full_projection_transform(),
                                                                     mg_transformed,
                                                                     ((512, 512),)).squeeze(dim=1).round()[:, :2]
                                kps_screen = kps_screen.cuda()
                                dists = ((kps_screen - self.detections[i]['kp'][j][None, :]) ** 2).sum(dim=-1)
                                idx = dists.argmin().item()
                                trans = mg[idx, 0].cpu().numpy()
                                brick_position = np.array(trans)
                                brick_position[1] = 0
                                rot_mask_ref_list.append(self.detections[i]['rot'][j].long().item())
                                if bid_this > 0:
                                    valid_trans = bs.add_brick(
                                        bid_decoded,
                                        brick_position,
                                        rot_decoded, self.op_types[i][j], canonical=False, only_check=True)
                                else:
                                    cbrick_bs = BricksPC.from_dict(cbrick._state_dict['bricks_pc'], no_check=True)
                                    cbrick_this = CBrick(cbrick_bs, brick_position, rot_decoded)
                                    valid_trans = add_cbrick_to_bricks_pc(bs, cbrick_this, self.op_types[i][j],
                                                                          only_check=True)
                            else:
                                conns = recompute_conns(bs, self.op_types[i][j], transforms, cameras, scale)

                                # Only perform a-of-s rotation prediction for submodules
                                if not self.opt.predict_masks or not self.opt.search_ref_mask or cbrick is None:
                                    keypoint_policy = 'brick' if self.opt.cbrick_brick_kp else 'simple'
                                    trans, candidates, pos_kp_offset = compute_2d_kp_to_3d_translation(bid_decoded,
                                                                                                       rot_decoded,
                                                                                                       op_type=
                                                                                                       self.op_types[i][
                                                                                                           j],
                                                                                                       kp=
                                                                                                       self.detections[
                                                                                                           i]['kp'][j],
                                                                                                       connections=conns,
                                                                                                       bricks_pc=bs,
                                                                                                       scale=scale,
                                                                                                       transforms=transforms,
                                                                                                       cameras=cameras,
                                                                                                       cbrick=cbrick,
                                                                                                       keypoint_policy=keypoint_policy
                                                                                                       )
                                    brick_position = np.array(trans) - pos_kp_offset
                                    if candidates:
                                        valid_trans = True
                                    else:
                                        valid_trans = False
                                    if self.opt.search_ref_mask:
                                        rot_mask_ref_list.append(self.detections[i]['rot'][j].long().item())
                                else:
                                    brick_candidates = []
                                    trans_candidates = []
                                    rot_decoded_candidates = []
                                    rot_id_candidates = []
                                    if bid_this >= 0:
                                        rots = get_brick_class(get_all_brick_ids()[bid_this]).get_valid_rotations()
                                    else:
                                        rots = [0, 1, 2, 3]

                                    from bricks.brick_info import cbrick_highest_brick
                                    import itertools
                                    if bid_this < 0:
                                        top_brick_num = len(cbrick_highest_brick(cbrick, allow_multiple=True)[0])
                                        top_brick_inds = list(range(top_brick_num))
                                    else:
                                        top_brick_inds = [-1]

                                    for rot_id, top_brick_ind in itertools.product(rots, top_brick_inds):
                                        # rot_decoded = id2euler(rot_id,
                                        #                    symmetry_aware_label=self.opt.symmetry_aware_rotation_label)
                                        rot_decoded = id2euler(rot_id, symmetry_aware_label=False)
                                        rot_decoded = tr.quaternion_from_euler(*(np.array(rot_decoded) * np.pi / 180))
                                        keypoint_policy = 'brick' if self.opt.cbrick_brick_kp else 'simple'
                                        trans, candidates, pos_kp_offset = \
                                            compute_2d_kp_to_3d_translation(bid_decoded,
                                                                            rot_decoded,
                                                                            op_type=
                                                                            self.op_types[i][
                                                                                j],
                                                                            kp=
                                                                            self.detections[
                                                                                i]['kp'][j],
                                                                            connections=conns,
                                                                            bricks_pc=bs,
                                                                            scale=scale,
                                                                            transforms=transforms,
                                                                            cameras=cameras,
                                                                            cbrick=cbrick,
                                                                            keypoint_policy=keypoint_policy,
                                                                            top_brick_ind=top_brick_ind
                                                                            )
                                        position = trans - pos_kp_offset
                                        if top_brick_ind == 0:
                                            original_pos_kp_offset = pos_kp_offset
                                        elif top_brick_ind > 0:
                                            trans = position + original_pos_kp_offset
                                        trans_candidates.append(trans)
                                        rot_decoded_candidates.append(rot_decoded)
                                        rot_id_candidates.append(rot_id)
                                        if bid_this >= 0:
                                            brick_candidates.append(Brick(bid_decoded, position, rot_decoded))
                                        else:
                                            cbrick_bs = BricksPC.from_dict(cbrick._state_dict['bricks_pc'],
                                                                           no_check=True)
                                            cbrick_this = CBrick(cbrick_bs, position, rot_decoded)
                                            brick_candidates.append(cbrick_this)

                                    iou_list = compute_candidates_iou(brick_candidates, bs, transforms, cameras,
                                                                      self.masks_inst_exp[i][:, :, j])
                                    ind = iou_list.index(max(iou_list))
                                    if max(iou_list) == -1:
                                        valid_trans = False
                                    else:
                                        valid_trans = True
                                    trans = trans_candidates[ind]
                                    rot_decoded = rot_decoded_candidates[ind]
                                    brick_position = brick_candidates[ind].position
                                    rot_id = rot_id_candidates[ind]
                                    # make rot_id symmetry aware if needed:
                                    if self.opt.symmetry_aware_rotation_label:
                                        rot_gt = rot_lists[i][j]
                                        if 0 < rot_gt < 3:
                                            rot_id += 1
                                        elif rot_gt >= 3:
                                            rot_id += 3
                                        else:
                                            rot_id = 0
                                    rot_mask_ref_list.append(rot_id)
                                    self.detections[i]['rot'][j] = rot_id

                            if valid_trans:
                                if bid_this >= 0:
                                    assert bs.add_brick(
                                        bid_decoded,
                                        brick_position,
                                        rot_decoded, self.op_types[i][j], canonical=False)
                                else:
                                    cbrick_bs = BricksPC.from_dict(cbrick._state_dict['bricks_pc'], no_check=True)
                                    cbrick_this = CBrick(cbrick_bs, brick_position, rot_decoded)
                                    assert add_cbrick_to_bricks_pc(bs, cbrick_this, self.op_types[i][j])

                                # otherwise cannot add the brick because there are no valid 3d positions

                            step_trans.append(trans)
                        self.detections[i]['trans'] = step_trans
                        dists_trans = ((np.array(self.trans[i]) - np.array(step_trans)) ** 2).sum(axis=1)
                        if self.opt.predict_masks and self.opt.search_ref_mask:
                            sum_dict['rot_mask_ref_acc'] += (
                                    torch.as_tensor(rot_mask_ref_list).cuda() == rot_lists[i]).sum().item()
                        sum_dict['trans_mse'] += dists_trans.sum()
                        sum_dict['trans_acc'] += (dists_trans <= 1e-6).sum()
                        sum_dict['trans_acc@tol1'] += (dists_trans <= 1).sum()
                        sum_dict['trans_acc@tol2'] += (dists_trans <= 4).sum()

                        num_objs = self.offset_regs_mask[i].sum().item()
                        self.detections[i]['rot_decoded'] = [None] * num_objs
                        for j in range(num_objs):
                            rot_id = self.detections[i]['rot'][j].long().item()
                            self.detections[i]['rot_decoded'][j] = id2euler(rot_id,
                                                                            symmetry_aware_label=self.opt.symmetry_aware_rotation_label)

                self.eval_outputs = {}
                for k, v in sum_dict.items():
                    self.eval_outputs[k] = v / obj_sum

            if not self.netHG.training and self.opt.predict_masks:
                if self.masks_inst is None:
                    self.compute_instance_masks()
                with torch.no_grad():
                    masks_pred = self.outputs[-1]['mask']

                    meters = Meters()
                    obj_sum = 0

                    result_dict_list = []
                    for i in range(len(self.masks)):
                        mask_pred = masks_pred[i]
                        mask_pred_decode = mask_pred.argmax(0)
                        mask_gt = self.masks[i]
                        brick_types_num = mask_gt.max()
                        obj_sum += brick_types_num
                        # Ignore background
                        result_dict = defaultdict(list)
                        for j in range(1, brick_types_num + 1):
                            mask_pred_this = mask_pred_decode == j
                            mask_gt_this = mask_gt == j
                            iou = compute_iou(mask_pred_this, mask_gt_this)
                            meters.update('iou', iou)
                            result_dict['iou'].append(iou)
                            for thres in [0.5, 0.75, 0.85, 0.95]:
                                meters.update(f'iou@{thres}', int(iou > thres))
                                result_dict[f'iou@{thres}'].append((iou > thres))

                        bricks_num = self.masks_inst_gt[i].max().item()
                        self.masks_inst[i], iou_mat = matching_inst_mask_idxs(self.masks_inst_gt[i],
                                                                              self.masks_inst[i])
                        for j in range(1, bricks_num + 1):
                            iou = iou_mat[j - 1]
                            meters.update('iou_instance', iou)
                            result_dict['iou_instance'].append(iou)
                            for thres in [0.5, 0.75, 0.85, 0.95]:
                                meters.update(f'iou_instance@{thres}', int(iou > thres))
                                result_dict[f'iou_instance@{thres}'].append((iou > thres))

                        result_dict_list.append(result_dict)

                    self.result_dict_list = result_dict_list
                    self.eval_outputs.update(meters.avg_dict())

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        if not self.netHG.training:
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

                pred_mask = expand_mask(output['mask'][i].squeeze(1).detach().argmax(0)).cpu().numpy()
                pred = gen_colormap(pred_mask, theme=theme)
                gt = gen_colormap(expand_mask(self.masks[i].detach()).cpu().numpy(), theme=theme)
                self.vis_dict['mask_pred'].append(blend_img(img, pred, theme=theme))
                self.vis_dict['mask_gt'].append(blend_img(img, gt, theme=theme))

                if self.opt.predict_masks:
                    if self.masks_inst is None:
                        self.compute_instance_masks()
                    instance = gen_colormap(expand_mask(self.masks_inst[i]).cpu().numpy(), theme=theme)
                    self.vis_dict[f'mask_pred_instance'].append(blend_img(img, instance, theme=theme))
                    instance_gt = gen_colormap(expand_mask(self.masks_inst_gt[i]).cpu().numpy(), theme=theme)
                    self.vis_dict[f'mask_gt_instance'].append(blend_img(img, instance_gt, theme=theme))

    def get_current_visuals(self):
        return self.vis_dict

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        loss = self.loss_sum
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
