import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
from torch import optim
import torch.multiprocessing as mp
import os


class f(torch.autograd.Function):
    """f-function from Megatron paper. g-function is not needed since
    master implements all_gather using torch.cat(futures)"""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, gradient):
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)


class MLP(nn.Module):
    """Non-distributed generalized matrix multiply"""

    def __init__(self, in_dim, hidden_dim, dropout):
        super(MLP, self).__init__()
        self.in_dim = in_dim  # Will just be equal to hidden dim since MLP is after MSA
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(self.in_dim, self.hidden_dim)
        # Output dimension is the same as input dimension since all GEMM are
        # all-reduced, not gathered
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.nonlin = F.gelu

    def forward(self, x):
        xa = self.fc1(x)

        y = self.nonlin(xa)

        z = self.fc2(y)

        return z


class MLPShard(nn.Module):
    def __init__(self, device, in_dim, hidden_dim, dropout):
        super(MLPShard, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.net = MLP(self.in_dim, self.hidden_dim, self.dropout).to(self.device)

    def parameter_rrefs(self):
        return [dist.rpc.RRef(p) for p in self.parameters()]

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        out = self.net(x)
        return out.cpu()


class SelfAttention(nn.Module):
    """Non-distributed self attention module"""

    # TODO: This module can be split before the final projection matmul
    #       (fc2) for distribtued pipeline parallel if needed. However, it
    #       may be better to split along layer boundaries instead.
    # TODO: Convert this from single head attention to MSA

    def __init__(self, input_dim, hidden_dim, dropout):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim  # actually total hidden_dim // 2
        self.hidden_dim = hidden_dim  # actually total hidden_dim // num_heads

        # hidden dim for this layer is 2 * model dim
        self.fc1 = nn.Linear(input_dim, hidden_dim * 3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        qkv = self.fc1(x)
        q, k, v = torch.split(qkv, self.hidden_dim, dim=-1)

        qk = torch.einsum("bse,bSe->bsS", q, k) * (self.input_dim * 2) ** -0.5

        att = F.softmax(qk, dim=-1)
        att = self.dropout(att)

        att = torch.einsum("bsS,bSe->bse", att, v)

        out = self.fc2(att)

        return out


class SelfAttentionShard(nn.Module):
    def __init__(self, device, shard_dim, data_dim, dropout):
        super(SelfAttentionShard, self).__init__()
        self.device = device
        self.shard_dim = shard_dim  # Sharded head dimension
        self.data_dim = data_dim  # Project full-sized data to qkv

        self.net = SelfAttention(self.data_dim, self.shard_dim, dropout).to(self.device)

    def parameter_rrefs(self):
        return [dist.rpc.RRef(p) for p in self.parameters()]

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        out = self.net(x)
        return out.cpu()


class DistAttention(nn.Module):
    def __init__(
        self,
        main_num_heads,
        main_input_data_dim,
        main_hidden_dim,
        main_workers,
        main_dropout,
        worker_shard_dim,
        worker_data_dim,
        worker_dropout,
    ):
        super(DistAttention, self).__init__()

        self.main_num_heads = main_num_heads
        self.main_input_data_dim = main_input_data_dim  # Total data embedding dim
        self.main_hidden_dim = (
            main_hidden_dim  # Effective hidden dimension, summed across shards
        )
        self.main_workers = main_workers

        # Make MSA shards
        self.msa_shards = []
        for i in range(self.main_num_heads):
            head_rref = rpc.remote(
                self.main_workers[i],
                SelfAttentionShard,
                args=(
                    f"cuda:{i}",
                    worker_shard_dim,
                    worker_data_dim,
                    worker_dropout,
                ),
            )
            self.msa_shards.append(head_rref)

        # Make MLP shards
        self.mlp_shards = []
        for i in range(self.main_num_heads):  # Shard the same as MSA
            mlp_rref = rpc.remote(
                self.main_workers[i],
                MLPShard,
                args=(
                    f"cuda:{i}",
                    main_hidden_dim,
                    main_hidden_dim,  # all shards have same output hidden dim since row split
                    worker_dropout,
                ),
            )
            self.mlp_shards.append(mlp_rref)

        # Initial layer norm and final fc layer for testing
        self.dropout = nn.Dropout(p=main_dropout)
        self.final_fc = nn.Linear(self.main_hidden_dim, 1)
        self.norm1 = nn.LayerNorm(self.main_input_data_dim)
        self.norm2 = nn.LayerNorm(self.main_hidden_dim)

    def forward(self, x):
        x = f.apply(x)  # .backward should all_reduce on master model

        x = self.norm1(x)  # TODO: Why does placing this before f error?

        # Sharded MSA
        out_futures = []
        for head_shard in self.msa_shards:
            x_rref = rpc.RRef(x)
            y_rref = head_shard.rpc_async().forward(x_rref)
            out_futures.append(y_rref)

        # All-gather
        attention_out = torch.cat(torch.futures.wait_all(out_futures), dim=-1)

        attention_out = self.dropout(attention_out)

        attention_out = self.norm2(attention_out)

        # Sharded MLP
        out_futures = []
        for mlp_shard in self.mlp_shards:
            x_rref = rpc.RRef(attention_out)
            y_rref = mlp_shard.rpc_async().forward(x_rref)
            out_futures.append(y_rref)

        # All-reduce
        mlp_out = torch.sum(torch.stack(torch.futures.wait_all(out_futures)), dim=0)

        out = self.final_fc(mlp_out)
        out = self.dropout(out)

        return out

    def parameter_rrefs(self):
        remote_params = []
        for head_shard in self.msa_shards:
            remote_params.extend(head_shard.remote().parameter_rrefs().to_here())
        for mlp_shard in self.mlp_shards:
            remote_params.extend(mlp_shard.remote().parameter_rrefs().to_here())
        remote_params.extend([dist.rpc.RRef(p) for p in self.parameters()])
        return remote_params


def run_master(num_heads, num_batches, batch_size, hidden_dim, sequence_length):
    input_data_dim = hidden_dim

    assert hidden_dim % num_heads == 0
    shard_dim = hidden_dim // num_heads

    model = DistAttention(
        main_num_heads=num_heads,
        main_input_data_dim=input_data_dim,
        main_hidden_dim=hidden_dim,
        main_workers=[f"worker{i}" for i in range(1, num_heads + 1)],
        main_dropout=0.1,
        worker_shard_dim=shard_dim,
        worker_data_dim=input_data_dim,
        worker_dropout=0.1,
    )
    loss_fn = nn.MSELoss()
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=1e-3,
    )

    one_hot_indices = (
        torch.LongTensor(batch_size).random_(0, sequence_length).view(batch_size, 1)
    )

    from tqdm import tqdm

    for i in tqdm(range(num_batches)):
        inputs = torch.randn(batch_size, sequence_length, hidden_dim - 1)
        labels = torch.zeros(batch_size, sequence_length).scatter_(
            1, one_hot_indices, 1
        )
        inputs = torch.cat((inputs, labels.unsqueeze(-1)), dim=-1)

        with dist_autograd.context() as context_id:
            outputs = model(inputs)
            dist_autograd.backward(context_id, [loss_fn(outputs.squeeze(), labels)])
            opt.step(context_id)


def run_worker(rank, world_size, num_heads):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)

    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options,
        )
        run_master(num_heads, 100, 64, 384, 1024)
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options,
        )

    rpc.shutdown()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    num_heads = world_size - 2
    mp.spawn(run_worker, args=(world_size, num_heads), nprocs=world_size, join=True)
