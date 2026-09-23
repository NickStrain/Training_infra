import os
import time 
import torch
import numpy as np
import torch.distributed as dist 
import torch.nn as nn
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


optimizer = torch.optim.Adam(dpp_model.parameters(),lr=1e-3)
criterion = nn.CrossEntropyLoss()

dpp_model.train()
total_loss = 0.0
num_steps = 0
step_times = []

for i in range(0,50):
    for batch_x, batch_y in data_loader:
        t0 = time.perf_counter()
        optimizer.zero_grad()

        logits = dpp_model(batch_x.to(device))
        loss = criterion(logits.view(-1,vocab_size),(batch_y.to(device)).view(-1))

        loss.backward()

        nn.utils.clip_grad_norm_(dpp_model.parameters(),max_norm=1.0)
        optimizer.step()

        step_time = time.perf_counter() - t0
        step_times.append(step_time)
        total_loss += loss.item()
        num_steps +=1

avg_loss = total_loss/num_steps
avg_step_ms = np.mean(step_times)*1000


print(f'Average loss: {avg_loss}')
print(f'Average step time: {avg_step_ms}')
clean_up_distributed_process()





