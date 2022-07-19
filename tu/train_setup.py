import os
import random

import numpy as np
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def spawn_ddp(args, worker):
    """

    Args:
        worker: a function with argument rank, world_size, args_in
            example see test_ddp_spawn

    Returns:

    """
    assert torch.cuda.is_available()
    world_size = torch.cuda.device_count()

    torch.multiprocessing.spawn(worker, nprocs=world_size, args=(world_size, args), join=True)
