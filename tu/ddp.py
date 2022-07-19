import torch
import torch.distributed as dist


def init_ddp(rank, world_size, port):
    assert not dist.is_initialized()
    dist.init_process_group(backend='nccl', init_method=f"tcp://localhost:{port}",
                            world_size=world_size, rank=rank)
    assert dist.is_available() and dist.is_initialized()

    torch.cuda.set_device(rank)

    print('init ddp from', dist.get_rank(), 'world size', dist.get_world_size())


def init_ddp_with_launch(local_rank, **kwargs):
    assert not dist.is_initialized()

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', **kwargs)

    assert dist.is_available() and dist.is_initialized()
    print('init ddp from', dist.get_rank(), 'world size', dist.get_world_size())


def get_distributed_model(model, rank, strict=False, find_unused_parameters=False):
    """
    .. warnings::
        User should call `model.to(rank)` before this function.

    Args:
        model:
        rank:
        strict:

    Returns:

    """

    if dist.is_available() and dist.is_initialized():
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=find_unused_parameters)

    if strict:
        raise RuntimeError('dist not available or initialized')

    return model


def get_distributed_dataloader(dataset, batch_size, strict=False, shuffle=True, num_workers=8, seed=0,
                               collate_fn=None) -> torch.utils.data.DataLoader:
    """

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example:
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         loader.sampler.set_epoch(epoch)
        ...     train(loader)

    Args:
        dataset:
        batch_size (int):
            From official doc: the batch size should be larger than the number of GPUs used locally.
        strict (bool):
        shuffle (bool): set to False for validation or test
        num_workers:
        seed:

    Returns:

    """
    if dist.is_available() and dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, seed=seed)
    elif strict:
        raise RuntimeError('dist not available or initialized')
    else:
        sampler = None
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and (sampler is None),
        sampler=sampler,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return data_loader


class InfDataLoader:
    def __init__(self, data_loader, init_epoch=0):
        self._epoch = init_epoch
        self.data_loader = data_loader

        self.iter = iter(self)

    def __iter__(self):
        # warning: calling next(iter(loader)) gives the same batch
        # because self.epoch += 1 is not reached
        while True:
            if self.data_loader.sampler is not None:
                self.data_loader.sampler.set_epoch(self._epoch)
            for batch in self.data_loader:
                yield batch
            self._epoch += 1

    def __next__(self):
        return next(self.iter)

    @property
    def epoch(self):
        return self._epoch


# from nvidia imaginaire
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import functools


def get_rank():
    r"""Get rank of the thread."""
    rank = 0
    if dist.is_available():
        if dist.is_initialized():
            rank = dist.get_rank()
    return rank


def get_world_size():
    r"""Get world size. How many GPUs are available in this job."""
    world_size = 1
    if dist.is_available():
        if dist.is_initialized():
            world_size = dist.get_world_size()
    return world_size


def master_only(func):
    r"""Apply this function only to the master GPU."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        r"""Simple function wrapper for the master function"""
        if get_rank() == 0:
            return func(*args, **kwargs)
        else:
            return None

    return wrapper


def is_master():
    r"""check if current process is the master"""
    return get_rank() == 0


@master_only
def master_only_print(*args):
    r"""master-only print"""
    print(*args)


def dist_reduce_tensor(tensor):
    r""" Reduce to rank 0 """
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if get_rank() == 0:
            tensor /= world_size
    return tensor


def dist_all_reduce_tensor(tensor):
    r""" Reduce to all ranks """
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.all_reduce(tensor)
        tensor.div_(world_size)
    return tensor


def dist_all_gather_tensor(tensor):
    r""" gather to all ranks """
    world_size = get_world_size()
    if world_size < 2:
        return [tensor]
    tensor_list = [
        torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    with torch.no_grad():
        dist.all_gather(tensor_list, tensor)
    return tensor_list
