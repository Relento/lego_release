import functools

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=opt.lr_decay_rate)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.001,
                                                   patience=opt.lr_patience)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.lr_decay_iters, eta_min=opt.min_lr)
    elif opt.lr_policy == 'cosine_restart':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt.lr_decay_iters, eta_min=opt.min_lr)
    elif opt.lr_policy == '1cycle':
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=opt.lr_decay_iters,
                                            cycle_momentum=False,
                                            div_factor=1e6)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1 or classname.find(
                'BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    if init_type == 'none':
        print('do not initialize(pretrained model)')
        return
    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


import torchvision.models as models
from .coordconv import AddCoords3D
import os

DEBUG_FMAP = bool(os.getenv("DEBUG_FMAP", 0) == '1')
debug_dir = 'lego_debug/'


class SimpleResNetEncoder(nn.Module):
    def __init__(self, pretrain=False, use_layer2=False):
        super().__init__()
        self.resnet = models.resnet18(pretrained=pretrain)
        self.use_layer2 = use_layer2

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        if self.use_layer2:
            x = self.resnet.layer2(x)
        return x


class SimpleResNetEncoder2(nn.Module):
    def __init__(self, pretrain=False, use_layer2=False):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrain)
        nets = []
        net_names = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1']
        if use_layer2:
            net_names.append('layer2')
        for net in net_names:
            nets.append(getattr(resnet, net))
        self.resnet = nn.Sequential(*nets)

    def forward(self, x):
        x = self.resnet(x)
        return x


class SimpleResNetEncoder3(nn.Module):
    def __init__(self, pretrain=False, use_layer2=False):
        super().__init__()
        resnet = models.resnet34(pretrained=pretrain)
        nets = []
        net_names = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1']
        if use_layer2:
            net_names.append('layer2')
        for net in net_names:
            nets.append(getattr(resnet, net))
        self.resnet = nn.Sequential(*nets)

    def forward(self, x):
        x = self.resnet(x)
        return x


from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class SimpleResNetEncoder4(nn.Module):
    def __init__(self, pretrain=False):
        super().__init__()
        self.resnet_fpn = resnet_fpn_backbone('resnet34', pretrained=pretrain,
                                              norm_layer=None, returned_layers=[1, 2],
                                              trainable_layers=5)

    def forward(self, x):
        x = self.resnet_fpn(x)
        return x['0']


def conv2d(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
):
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=padding,
        stride=stride,
        bias=False,
    )
    bn = nn.BatchNorm2d(out_channels)
    relu = nn.ReLU()
    return [conv, bn, relu]


def conv3d(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
):
    conv = nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size,
        padding=padding,
        stride=stride,
        bias=False,
    )
    bn = nn.BatchNorm3d(out_channels)
    relu = nn.ReLU()
    return [conv, bn, relu]


def upconv3d(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        output_padding: int = 0,
        final_layer: bool = False,
):
    upconv = nn.ConvTranspose3d(
        in_channels,
        out_channels,
        kernel_size,
        padding=padding,
        stride=stride,
        output_padding=output_padding,
        bias=final_layer,
    )
    if not final_layer:
        bn = nn.BatchNorm3d(out_channels)
        relu = nn.ReLU()
        return [upconv, bn, relu]
    else:
        return [upconv]


def simple_voxel_encoder1(num_features, num_input_features=1, addconv=True):
    if addconv:
        layers = [AddCoords3D()]
        layers.extend(conv3d(num_input_features + 3, num_features))
    else:
        layers = conv3d(num_input_features, num_features)
    layers.extend(conv3d(num_features, num_features, stride=2))
    layers.extend(conv3d(num_features, num_features, stride=2))
    layers.extend(conv3d(num_features, num_features, stride=2))
    return nn.Sequential(*layers)


def simple_voxel_encoder2(num_features, num_input_features=3, addconv=True):
    if addconv:
        layers = [AddCoords3D()]
        layers.extend(conv3d(num_input_features + 3, num_features))
    else:
        layers = conv3d(num_input_features, num_features)
    layers.extend(conv3d(num_features, num_features, stride=2))
    return nn.Sequential(*layers)


# output meshgrid of the camera space
# sizes [..., H, W, D]
def ndc_meshgrid(size):
    x = torch.linspace(1, -1, size[-3])
    y = torch.linspace(1, -1, size[-2])
    # set according training data's depth range
    z = torch.linspace(0.025, 0.055, size[-1])
    coords = torch.meshgrid(x, y, z)
    coords = torch.stack(coords, dim=-1)
    coords = coords.unsqueeze(0).expand(*size, 3)
    return coords


def argmax_first(input):
    # only supports argmax over the last dimension
    indices = torch.arange(input.shape[-1], device=input.device).repeat(input.shape[:-1] + (1,))
    max_values, _ = input.max(dim=-1)
    indices[input != max_values.to(input.device).unsqueeze(-1)] = input.shape[-1]
    return indices.min(dim=-1)[0]


def occ_inds2fmap(occ_inds, overlap=True):
    imgs = occ_inds.float()
    imgs_unique = imgs.unique()
    imgs -= imgs_unique[1]
    imgs /= (imgs_unique[-1] - imgs_unique[1])
    imgs[imgs < 0] = 0
    # imgs = (imgs * 255).long().transpose(-3, -2)
    imgs = (imgs * 255).long()
    from PIL import Image
    import numpy as np
    for i in range(imgs.shape[0]):
        if imgs[i].shape[0] > 1:
            img_np = (imgs[i].squeeze().detach().cpu().numpy()).astype(np.uint8)
            for j in range(img_np.shape[0]):
                img = Image.fromarray(img_np[j])
                img.save(debug_dir + f'{str(i).zfill(3)}_rot{j}.png')
        else:
            img = (imgs[i].squeeze().detach().cpu().numpy()).astype(np.uint8)
            img = Image.fromarray(img)
            img.save(debug_dir + f'{str(i).zfill(3)}.png')
        if overlap:
            img_img = Image.open(debug_dir + f'{str(i).zfill(3)}_img.png')
            img = img.resize(img_img.size).convert('RGB')
            img_overlap = np.array(img).astype(np.float) * 0.7 + np.array(img_img).astype(np.float) * 0.3
            img_overlap = Image.fromarray(img_overlap.astype(np.uint8))
            img_overlap.save(debug_dir + f'{str(i).zfill(3)}_overlap.png')
    import ipdb;
    ipdb.set_trace()
