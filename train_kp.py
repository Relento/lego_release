"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import argparse
import collections
import time

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from datasets import CustomDatasetDataLoader
from datasets import create_dataset
from models import create_model
from options.train_options import TrainOptions
from tu.ddp import init_ddp_with_launch, get_distributed_dataloader, master_only_print, get_rank, get_world_size
from tu.loggers.visualizer import HTMLVisualizer as Visualizer
from tu.train_setup import set_seed


def create_distributed_dataset(opt, val=False):
    data_loader = DistributedCustomDatasetDataloader(opt, val=val)
    dataset = data_loader.load_data()
    return dataset


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            v = input_dict[k]
            values.append(v if isinstance(input_dict[k], torch.Tensor) else torch.as_tensor(v).to(get_rank()))
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class DistributedCustomDatasetDataloader(CustomDatasetDataLoader):
    def __init__(self, opt, val=False):
        super().__init__(opt, val)

        # overwrite dataloader with the distributed version
        d = self.dataloader
        self.dataloader = get_distributed_dataloader(d.dataset,
                                                     batch_size=d.batch_size // torch.distributed.get_world_size(),
                                                     shuffle=not (opt.serial_batches or val),
                                                     num_workers=d.num_workers, collate_fn=d.collate_fn,
                                                     seed=opt.seed)
        self._epoch = opt.epoch_count

    def __iter__(self):
        # print('setting sampler epoch', self._epoch)
        self.dataloader.sampler.set_epoch(self._epoch)
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

        self._epoch += 1


class DistributedTrainOptions(TrainOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = TrainOptions.initialize(self, parser)
        parser.add_argument('--local_rank', type=int, required=True)
        parser.add_argument('--use_ddp', action='store_true')
        return parser


def main():
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # ddp
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_ddp', action='store_true')
    parser.add_argument('--local_rank', type=int)
    namespace = parser.parse_known_args()[0]
    use_ddp = namespace.use_ddp
    if use_ddp:
        init_ddp_with_launch(namespace.local_rank)
        # init ddp before parsing so that master_only print works
        opt = DistributedTrainOptions().parse()  # get training options
        opt.gpu_ids = [opt.local_rank]
        print(torch.cuda.current_device())
    else:
        opt = TrainOptions().parse()

    if opt.wandb and get_rank() == 0:
        import wandb
        wandb.init(project='lego', name=opt.wandb_prefix + 'run-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S')),
                   settings=wandb.Settings(start_method='fork'))

    if use_ddp:
        dataset = create_distributed_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    else:
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    master_only_print('The number of training examles = %d' % dataset_size)

    if opt.val_dataroot:
        val_dataset = create_dataset(opt, val=True)  # create a dataset given opt.dataset_mode and other options
        master_only_print('The number of validation examples = %d' % len(val_dataset))

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations

    if opt.wandb and get_rank() == 0:
        wandb.config.update(opt)
        for net_name in model.model_names:
            wandb.watch(getattr(model, 'net' + net_name))

    # DEBUG
    if use_ddp:
        p = next(getattr(model, 'net' + model.model_names[0]).named_parameters())
        print(p[0], p[1].device, 'from rank', opt.local_rank, opt.gpu_ids)

    # hack: save logs under ${opt.checkpoints_dir}/${opt.name}/loss_log_${opt.local_rank}.txt
    # visualizer.log_name = os.path.join(opt.checkpoints_dir, opt.name, f'loss_log_{opt.local_rank}.txt')

    if opt.seed > 0:
        # Set random seed for this experiment
        set_seed(opt.seed)

    best_val = np.inf
    batch_lr_schedulers = ['1cycle', 'cosine', 'cosine_restart', 'step']
    for epoch in range(opt.epoch_count,
                       opt.niter + opt.niter_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

        master_only_print('Dataset size:', len(dataset))
        for i, data in enumerate(dataset):  # inner loop within one epoch
            model.train()
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size  # technically need drop_last = True in dataloader
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            sync = (epoch_iter // opt.batch_size) % opt.acc_grad == 0
            sync = True
            model.optimize_parameters(sync)  # calculate loss functions, get gradients, update network weights
            if opt.lr_policy in batch_lr_schedulers:
                model.update_learning_rate()

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                if opt.wandb:
                    monitor = dict(losses)
                    monitor = reduce_dict(monitor)
                    if get_rank() == 0:
                        monitor['epoch'] = epoch
                        lr = model.optimizers[0].param_groups[0]['lr']
                        monitor['lr'] = lr
                        wandb.log(monitor)

                if get_rank() == 0:
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if get_rank() == 0:
                if total_iters % opt.vis_freq == 0:
                    # model.compute_visuals(n_vis=1000)
                    model.compute_visuals()
                    if opt.wandb:
                        vis_dict = {}
                        for vis_name, l in model.vis_dict.items():
                            l_wandb = [wandb.Image(img, caption=str(i)) for i, img in enumerate(l)]
                            vis_dict[vis_name] = l_wandb
                        wandb.log(vis_dict)
                    else:
                        # one sample in the batch takes one row
                        layout = []
                        if model.get_current_visuals():
                            for i in range(len(next(iter(model.get_current_visuals().values())))):
                                layout.append([])
                                for k, v in model.get_current_visuals().items():
                                    layout[-1].append(dict(info=f"{k}_{i}", image=Image.fromarray(v[i])))
                            visualizer.display_current_results(layout, epoch, epoch_iter)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                if get_rank() == 0:
                    master_only_print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

                with torch.no_grad():
                    if opt.val_dataroot:
                        model.eval()
                        losses_sum = collections.defaultdict(float)
                        losses_count = collections.defaultdict(int)
                        losses_val = collections.defaultdict(float)
                        # if epoch >= 10:
                        #     import ipdb; ipdb.set_trace()
                        for i, data in enumerate(val_dataset):  # inner loop within one epoch
                            model.set_input(data)  # unpack data from dataset and apply preprocessing
                            model.test()
                            losses = model.get_current_losses()
                            data, targets = data
                            for name, v in losses.items():
                                losses_sum[name] += v * data['img'].shape[0]
                                losses_count[name] += data['img'].shape[0]

                        for name, v in losses_sum.items():
                            losses_val['val/' + name] = v / losses_count[name]
                        if use_ddp:
                            losses_val = reduce_dict(losses_val)
                        model.metric = losses_val['val/sum']

                        if get_rank() == 0:
                            if opt.wandb:
                                monitor = dict(losses_val)
                                monitor['epoch'] = epoch
                                wandb.log(monitor)
                            visualizer.print_current_losses(epoch, epoch_iter, losses_val, 0, 0)

                        if model.metric < best_val:
                            best_val = model.metric
                            master_only_print(
                                'Saving the best model at the end of epoch %d, iters %d' % (epoch, total_iters))
                            model.save_networks('best')

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            master_only_print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        if opt.lr_policy not in batch_lr_schedulers:
            model.update_learning_rate()  # update learning rates at the beginning of every step
        master_only_print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))


if __name__ == '__main__':
    main()
