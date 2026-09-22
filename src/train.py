import os

import torch.distributed as dist 


def setup_distributed_process():
    '''
    Initializing the distributed process
    '''

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADD","localhost")
        os.environ.setdefault("MASTER_PORT","12355")

        dist.init_process_group(
            backend="nccl",
            world_size=1,
            rank=1
        )



def clean_up_distributed_process():
    '''
    Cleaning up the distributed process from the DDP
    '''

    if dist.is_initialized():
        dist.destroy_process_group()