#!/usr/bin/env bash
set -euo pipefail

# Single node, N processes (one per GPU, or just N>1 on CPU with gloo to
# exercise the DDP mechanics on a single-GPU machine).
NPROC=${1:-1}

torchrun \
  --standalone \
  --nproc_per_node="$NPROC" \
  -m src.train
