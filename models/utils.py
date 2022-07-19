import numpy as np
import torch
from torch.optim.optimizer import Optimizer


class AccumGrad(Optimizer):
    def __init__(self, base_optimizer, nr_acc):
        self._base_optimizer = base_optimizer
        self._nr_acc = nr_acc
        self._current = 0

    @property
    def state(self):
        return self._base_optimizer.state

    @property
    def param_groups(self):
        return self._base_optimizer.param_groups

    def state_dict(self):
        return {
            'base_optimizer': self._base_optimizer.state_dict(),
            'current': self._current
        }

    def load_state_dict(self, state_dict):
        self._current = state_dict['current']
        return self._base_optimizer.load_state_dict(state_dict['base_optimizer'])

    def zero_grad(self):
        return self._base_optimizer.zero_grad()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self._current += 1

        for group in self._base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self._base_optimizer.state[p]

                if 'grad_buffer' not in param_state:
                    buf = param_state['grad_buffer'] = d_p.clone()
                else:
                    buf = param_state['grad_buffer']
                    buf.add_(d_p)

                if self._current >= self._nr_acc:
                    buf.mul_(1. / self._current)
                    p.grad.data.copy_(buf)
                    buf.zero_()
                    del param_state['grad_buffer']

        if self._current >= self._nr_acc:
            self._base_optimizer.step()
            self._current = 0

        return loss


def assign_masks(emb, semantic_mask, n_ins, init_kps=None):
    '''
    :param emb: [H, W]
    :param semantic_mask: [H, W]
    :return:
    '''
    from sklearn.cluster import KMeans
    inds = semantic_mask.nonzero()
    n_ins = min(n_ins, inds[0].size)
    if n_ins == 0:
        return torch.zeros(emb.shape)
    if init_kps is not None:
        # use the position of keypoints to set the center of clusters
        inds_pts = np.array(list(zip(*inds)))
        dists = ((init_kps[:, None, [1, 0]] - inds_pts[None, :]) ** 2).sum(axis=-1)
        nearest_inds_pts_inds = dists.argmin(axis=1)
        inds_pts_selected = inds_pts[nearest_inds_pts_inds]
        init_pts = emb[inds_pts_selected[:, 0], inds_pts_selected[:, 1]].cpu().numpy()
        if len(init_pts.shape) == 1:
            init_pts = init_pts[:, None]

    pts = emb[inds][:, None]
    if init_kps is not None:
        labels = KMeans(n_clusters=n_ins, init=init_pts[:n_ins], n_init=1).fit(pts.cpu().numpy()).labels_
    else:
        labels = KMeans(n_clusters=n_ins).fit(pts.cpu().numpy()).labels_
    instance_masks = np.zeros(semantic_mask.shape, dtype=np.int32)
    instance_masks[inds] = labels + 1
    return instance_masks


def expand_mask(m, num_classes=-1):
    return torch.nn.functional.one_hot(m.long(), num_classes=num_classes).permute((2, 0, 1))


from scipy.optimize import linear_sum_assignment


def matching_inst_mask_idxs(mask_gt, mask_pred):
    '''
    :param mask_gt: [H, W]
    :param mask_pred: [H, W]
    :return:
    mask_pred with reordered indices to that match gt mask
    '''
    N = mask_gt.max().item()  # number of instances.
    mask_gt_exp = expand_mask(mask_gt, num_classes=N + 1)[1:]  # remove background mask.
    mask_pred_exp = expand_mask(mask_pred, num_classes=N + 1)[1:]

    mat_intersection = torch.minimum(mask_gt_exp[:, None], mask_pred_exp[None]).float().reshape(N, N, -1).sum(-1)
    mat_union = torch.maximum(mask_gt_exp[:, None], mask_pred_exp[None]).float().reshape(N, N, -1).sum(-1)
    mat_iou = (mat_intersection / (mat_union + 1e-6)).cpu().numpy()
    idxs_gt, idxs_pred = linear_sum_assignment(-mat_iou)
    idxs_gt = list(idxs_gt)
    idxs_pred = list(idxs_pred)

    def remap_idxs(i):
        if i == 0:
            return 0
        return idxs_pred.index(i - 1) + 1

    mask_pred_reordered = np.vectorize(remap_idxs)(mask_pred.cpu().numpy())
    return torch.as_tensor(mask_pred_reordered, device=mask_pred.device), mat_iou[idxs_gt, idxs_pred]


def compute_iou(mask_a, mask_b):
    if isinstance(mask_a, torch.Tensor):
        intersect = (mask_a & mask_b).float().sum()
        union = (mask_a | mask_b).float().sum()
        iou = (intersect / union).item()
    else:
        assert isinstance(mask_a, np.ndarray)
        intersect = (mask_a & mask_b).astype(np.float).sum()
        union = (mask_a | mask_b).astype(np.float).sum()
        iou = intersect / union
    return iou


from collections import defaultdict


class Meters:
    def __init__(self):
        self.sum_d = defaultdict(int)
        self.n_d = defaultdict(int)

    def update(self, name, v, n=1):
        self.sum_d[name] += v
        self.n_d[name] += n

    def avg(self, name):
        return self.sum_d[name] / self.n_d[name]

    def avg_dict(self):
        d = {}
        for k in self.sum_d.keys():
            d[k] = self.sum_d[k] / self.n_d[k]
        return d

    def merge_from(self, m, ks=None):
        if ks is None:
            ks = m.sum_d.keys()
        for k in ks:
            self.sum_d[k] += m.sum_d[k]
            self.n_d[k] += m.n_d[k]
