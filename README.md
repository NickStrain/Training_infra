# Training_infra

Sandbox for experimenting with data parallelism / DistributedDataParallel (DDP).

```
configs/ddp.yaml       # basic hyperparams
src/
  data.py              # toy dataset + DistributedSampler dataloader
  model.py             # toy MLP
  train.py             # DDP training loop (entrypoint: python -m src.train)
  utils/distributed.py # process group setup/teardown helpers
scripts/run_ddp.sh      # torchrun launcher, e.g. ./scripts/run_ddp.sh 2
```

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Note: this venv is on Python 3.14, which may be too new for the current
PyTorch wheels — if `pip install torch` fails, create a separate venv with
Python 3.11/3.12 for this experiment.

## Run

```
./scripts/run_ddp.sh <nproc>   # e.g. 2 to simulate 2 workers on the single GPU/CPU
```
