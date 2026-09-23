import os
import time 
import torch
import torch.distributed as dist 
from model import Model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


def setup_distributed_process():
    '''
    Initializing the distributed process
    '''

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR","localhost")
        os.environ.setdefault("MASTER_PORT","12355")
        os.environ.setdefault("RANK","0")
        os.environ.setdefault("WORLD_SIZE","1")

        # rank/world_size come from the env vars torchrun sets
        dist.init_process_group(backend="nccl")



def clean_up_distributed_process():
    '''
    Cleaning up the distributed process from the DDP
    '''

    if dist.is_initialized():
        dist.destroy_process_group()

setup_distributed_process()

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

vocab_size = 10000
model = Model(vocab_size=vocab_size)
model = model.to(device=device)

dpp_model  = DDP(model, device_ids=[local_rank])


num_sample = 1024
seq_len = 32

X = torch.randint(0,vocab_size,(num_sample,seq_len))
y = torch.randint(0,vocab_size,(num_sample,seq_len))


dataset =  TensorDataset(X,y)


sampler = DistributedSampler(
    dataset=dataset,
    num_replicas=dist.get_world_size(),
    rank=dist.get_rank(),
    shuffle=True,
    seed=42,
)

data_loader = DataLoader(dataset=dataset,batch_size=32,sampler=sampler)

num_batches = len(data_loader)
sampler_per_rank = len(sampler)

print(f'No of the batchers {num_batches}')
print(f'No of the sample per rank {sampler_per_rank}')

clean_up_distributed_process()





