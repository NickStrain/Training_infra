import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from utils.distributed import setup_distributed, cleanup_distributed, is_main_process



class _CopyToTensorParallelRegion(torch.autograd.Function):
    """Forward: identity. Backward: all-reduce the gradient.

    Used at the input of a ColumnParallelLinear: every rank sees the full
    input, so each rank's grad w.r.t. that input is only a partial sum.
    """

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)
        return grad_output


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    """Forward: all-reduce. Backward: identity.

    Used at the output of a RowParallelLinear: each rank produces a partial
    sum of the output, and the gradient of a sum is passed through unchanged.
    """

    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _GatherFromTensorParallelRegion(torch.autograd.Function):
    """Forward: all-gather along the last dim. Backward: keep this rank's slice."""

    @staticmethod
    def forward(ctx, x):
        world_size = dist.get_world_size()
        x = x.contiguous()
        chunks = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(chunks, x)
        return torch.cat(chunks, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        return grad_output.chunk(world_size, dim=-1)[rank].contiguous()


class _ScatterToTensorParallelRegion(torch.autograd.Function):
    """Forward: keep this rank's slice of the last dim. Backward: all-gather."""

    @staticmethod
    def forward(ctx, x):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        return x.chunk(world_size, dim=-1)[rank].contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        world_size = dist.get_world_size()
        grad_output = grad_output.contiguous()
        chunks = [torch.empty_like(grad_output) for _ in range(world_size)]
        dist.all_gather(chunks, grad_output)
        return torch.cat(chunks, dim=-1)


def copy_to_tp_region(x):
    return _CopyToTensorParallelRegion.apply(x)


def reduce_from_tp_region(x):
    return _ReduceFromTensorParallelRegion.apply(x)


def gather_from_tp_region(x):
    return _GatherFromTensorParallelRegion.apply(x)


def scatter_to_tp_region(x):
    return _ScatterToTensorParallelRegion.apply(x)


# ---------------------------------------------------------------------------
# Parallel layers
# ---------------------------------------------------------------------------

def _init_full_weight(out_features: int, in_features: int, seed: int) -> torch.Tensor:
    """Build the full (unsharded) weight with a fixed seed so every rank agrees
    on it, then each rank slices out its own shard."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(out_features, in_features, generator=gen) * (1.0 / in_features**0.5)


class ColumnParallelLinear(nn.Module):
    """Y = X A + b with A split by columns: A = [A_1, ..., A_p].

    In nn.Linear terms the weight is (out, in), so we shard along dim 0
    (out_features). Each rank computes Y_i = X A_i + b_i.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 rank: int,
                 world_size: int,
                 bias: bool = True,
                 gather_output: bool = False,
                 seed: int = 0,
                 ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.world_size = world_size
        self.gather_output = gather_output

        assert out_features % world_size == 0, (
            f"out_features {out_features} must be divisible by world_size {world_size}"
        )
        self.shard_size = out_features // world_size

        full_weight = _init_full_weight(out_features, in_features, seed)
        start = rank * self.shard_size
        self.weight = nn.Parameter(full_weight[start:start + self.shard_size].clone())

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.shard_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = copy_to_tp_region(x)
        output = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            output = gather_from_tp_region(output)
        return output


class RowParallelLinear(nn.Module):
    """Y = X A + b with A split by rows: A = [A_1; ...; A_p], X = [X_1, ..., X_p].

    In nn.Linear terms the weight is (out, in), so we shard along dim 1
    (in_features). Each rank computes X_i A_i, then the partial results are
    summed with an all-reduce. Bias is added once, after the reduction.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 rank: int,
                 world_size: int,
                 bias: bool = True,
                 input_is_parallel: bool = True,
                 seed: int = 0,
                 ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.world_size = world_size
        self.input_is_parallel = input_is_parallel

        assert in_features % world_size == 0, (
            f"in_features {in_features} must be divisible by world_size {world_size}"
        )
        self.shard_size = in_features // world_size

        full_weight = _init_full_weight(out_features, in_features, seed)
        start = rank * self.shard_size
        self.weight = nn.Parameter(full_weight[:, start:start + self.shard_size].clone())

        if bias:
            # Full-size bias, replicated on every rank.
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            x = scatter_to_tp_region(x)
        partial = F.linear(x, self.weight)
        output = reduce_from_tp_region(partial)
        if self.bias is not None:
            output = output + self.bias
        return output


class TensorParallelMLP(nn.Module):
    """Megatron-style MLP: Column -> activation -> Row.

    The intermediate activation stays sharded, so the whole block needs a
    single all-reduce in forward and a single all-reduce in backward.
    """

    def __init__(self, d_model: int, d_ff: int, rank: int, world_size: int, seed: int = 0) -> None:
        super().__init__()
        self.fc1 = ColumnParallelLinear(d_model, d_ff, rank, world_size, gather_output=False, seed=seed)
        self.fc2 = RowParallelLinear(d_ff, d_model, rank, world_size, input_is_parallel=True, seed=seed + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Correctness check against a plain single-device MLP.
# Run with: torchrun --standalone --nproc_per_node=2 src/tensor_parallelism.py
# ---------------------------------------------------------------------------

def _reference_mlp(d_model: int, d_ff: int, seed: int = 0) -> nn.Sequential:
    fc1 = nn.Linear(d_model, d_ff)
    fc2 = nn.Linear(d_ff, d_model)
    with torch.no_grad():
        fc1.weight.copy_(_init_full_weight(d_ff, d_model, seed))
        fc1.bias.zero_()
        fc2.weight.copy_(_init_full_weight(d_model, d_ff, seed + 1))
        fc2.bias.zero_()
    return nn.Sequential(fc1, nn.ReLU(), fc2)


def main() -> None:
    # One GPU per rank needs NCCL; otherwise fall back to gloo on CPU so we can
    # simulate several ranks on a single-GPU machine.
    n_gpus = torch.cuda.device_count()
    world_size_env = int(os.environ.get("WORLD_SIZE", 1))
    backend = "nccl" if n_gpus >= world_size_env else "gloo"
    rank, local_rank, world_size = setup_distributed(backend=backend)
    device = torch.device(f"cuda:{local_rank}") if backend == "nccl" else torch.device("cpu")

    d_model, d_ff, batch, seq = 64, 256, 4, 8

    tp_mlp = TensorParallelMLP(d_model, d_ff, rank, world_size).to(device)
    ref_mlp = _reference_mlp(d_model, d_ff).to(device)

    # Same input on every rank.
    torch.manual_seed(1234)
    x = torch.randn(batch, seq, d_model).to(device)
    x_tp = x.clone().requires_grad_(True)
    x_ref = x.clone().requires_grad_(True)

    out_tp = tp_mlp(x_tp)
    out_ref = ref_mlp(x_ref)
    out_tp.sum().backward()
    out_ref.sum().backward()

    shard1 = tp_mlp.fc1.shard_size
    shard2 = tp_mlp.fc2.shard_size
    checks = {
        "forward output": torch.allclose(out_tp, out_ref, atol=1e-5),
        "input grad": torch.allclose(x_tp.grad, x_ref.grad, atol=1e-5),
        "fc1 weight grad": torch.allclose(
            tp_mlp.fc1.weight.grad,
            ref_mlp[0].weight.grad[rank * shard1:(rank + 1) * shard1], atol=1e-5),
        "fc1 bias grad": torch.allclose(
            tp_mlp.fc1.bias.grad,
            ref_mlp[0].bias.grad[rank * shard1:(rank + 1) * shard1], atol=1e-5),
        "fc2 weight grad": torch.allclose(
            tp_mlp.fc2.weight.grad,
            ref_mlp[2].weight.grad[:, rank * shard2:(rank + 1) * shard2], atol=1e-5),
        "fc2 bias grad": torch.allclose(tp_mlp.fc2.bias.grad, ref_mlp[2].bias.grad, atol=1e-5),
    }

    for name, ok in checks.items():
        print(f"[rank {rank}] {name}: {'OK' if ok else 'MISMATCH'}")

    if is_main_process():
        n_local = sum(p.numel() for p in tp_mlp.parameters())
        n_full = sum(p.numel() for p in ref_mlp.parameters())
        print(f"world_size={world_size} backend={backend} "
              f"params per rank={n_local} vs full={n_full}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
