from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from .models.losses import FocalLoss, L1Loss, RotLoss, TagLoss
from .models.utils import _sigmoid


class KPLoss(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = L1Loss()
        self.crit_rot = RotLoss()
        if opt.predict_trans:
            self.crit_trans = L1Loss()
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt

        hm_loss, rot_loss, off_loss = 0, 0, 0
        if 'bid' in batch:
            bid_loss = 0

        if opt.predict_trans:
            trans_loss = 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.lbd_r > 0 and 'rot' in output:
                rot_loss += self.crit_rot(output['rot'], batch['ind'], batch['rot']
                                          ) / opt.num_stacks
            if opt.lbd_r > 0 and 'bid' in output:
                bid_loss += self.crit_rot(output['bid'], batch['ind'], batch['bid']
                                          ) / opt.num_stacks

            if opt.reg_offset and opt.lbd_o > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.predict_trans and opt.lbd_t > 0:
                trans_loss += self.crit_trans(output['trans'], batch['reg_mask'],
                                              batch['ind'], batch['trans']) / opt.num_stacks

        loss = opt.lbd_h * hm_loss + \
               opt.lbd_r * rot_loss + \
               opt.lbd_o * off_loss

        if 'bid' in outputs[0]:
            loss += opt.lbd_r * bid_loss

        if opt.predict_trans:
            loss += opt.lbd_t * trans_loss

        loss_stats = {'loss_sum': loss, 'loss_hm': hm_loss,
                      'loss_ce_rot': rot_loss,
                      'loss_l1_off': off_loss}
        if opt.predict_trans:
            loss_stats['loss_l1_trans'] = trans_loss
        if 'rot' not in outputs[0]:
            del loss_stats['loss_ce_rot']
        if 'bid' in outputs[0]:
            loss_stats['loss_ce_bid'] = bid_loss

        return loss, loss_stats


class MaskLoss(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.crit = torch.nn.CrossEntropyLoss()
        self.opt = opt
        if opt.assoc_emb:
            self.crit_assoc_emb = TagLoss(opt)

    def forward(self, outputs, batch):
        opt = self.opt

        mask_loss = 0
        if self.opt.assoc_emb:
            pull_loss = 0
            push_loss = 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            mask_loss += self.crit(output['mask'], batch['mask']) / opt.num_stacks
            if self.opt.assoc_emb:
                losses = self.crit_assoc_emb(output['assoc_emb'], batch['mask_instance'],
                                             batch['assoc_emb_sample_inds'])
                pull_loss += losses[0] / opt.num_stacks
                push_loss += losses[1] / opt.num_stacks

        loss = 0
        loss += mask_loss
        if self.opt.assoc_emb:
            loss += self.opt.assoc_emb_lbd * pull_loss + self.opt.assoc_emb_lbd * push_loss

        loss_stats = {'loss_sum': loss, 'loss_mask': mask_loss}
        if self.opt.assoc_emb:
            loss_stats.update({'loss_pull': pull_loss, 'loss_push': push_loss})
        return loss, loss_stats

# class CtdetTrainer(BaseTrainer):
#     def __init__(self, opt, model, optimizer=None):
#         super(CtdetTrainer, self).__init__(opt, model, optimizer=optimizer)
#
#     def _get_losses(self, opt):
#         loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
#         loss = CtdetLoss(opt)
#         return loss_states, loss
#
#     def debug(self, batch, output, iter_id):
#         opt = self.opt
#         reg = output['reg'] if opt.reg_offset else None
#         dets = ctdet_decode(
#             output['hm'], output['wh'], reg=reg,
#             cat_spec_wh=opt.cat_spec_wh, K=opt.K)
#         dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
#         dets[:, :, :4] *= opt.down_ratio
#         dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
#         dets_gt[:, :, :4] *= opt.down_ratio
#         for i in range(1):
#             debugger = Debugger(
#                 dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
#             img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
#             img = np.clip(((
#                                    img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
#             pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
#             gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
#             debugger.add_blend_img(img, pred, 'pred_hm')
#             debugger.add_blend_img(img, gt, 'gt_hm')
#             debugger.add_img(img, img_id='out_pred')
#             for k in range(len(dets[i])):
#                 if dets[i, k, 4] > opt.center_thresh:
#                     debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
#                                            dets[i, k, 4], img_id='out_pred')
#
#             debugger.add_img(img, img_id='out_gt')
#             for k in range(len(dets_gt[i])):
#                 if dets_gt[i, k, 4] > opt.center_thresh:
#                     debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
#                                            dets_gt[i, k, 4], img_id='out_gt')
#
#             if opt.debug == 4:
#                 debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
#             else:
#                 debugger.show_all_imgs(pause=True)
#
#     def save_result(self, output, batch, results):
#         reg = output['reg'] if self.opt.reg_offset else None
#         dets = ctdet_decode(
#             output['hm'], output['wh'], reg=reg,
#             cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
#         dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
#         dets_out = ctdet_post_process(
#             dets.copy(), batch['meta']['c'].cpu().numpy(),
#             batch['meta']['s'].cpu().numpy(),
#             output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
#         results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
