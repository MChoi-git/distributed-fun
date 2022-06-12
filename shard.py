import threading
import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.autograd as dist_autograd


logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)


class ModuleShard(nn.Module):
    def __init__(self, device, module, *args, **kwargs):
        """Abstracted module shard for use in megatron-style tensor parallelism
        and pipeline parallelism
        Args:
            device (str): Device to place module on
            module (nn.Module): Arbtrary pytorch module to wrap
        """
        # TODO: Distributed network will contain rrefs to each ModuleShard, one
        #       per GPU/worker (initialized via remote()). Each ModuleShard
        #       can contain arbitrary forward pass code (ie. nn.Modules).
        #       Example:
        #           - 2 attention layers, each layer is further split into two
        #             tensor streams
        #           - Ie.
        #                   L1                      L2
        #                |->attn(GPU1) --|         |->attn(GPU3)
        #   microbatch1->|               |->master-|
        #                |->attn(GPU2) --|         |->attn(GPU4)
        #                                             L1                      L2
        #                                          |->attn(GPU1) --|         |->attn(GPU3)
        #                             microbatch2->|               |->master-|
        #                                          |->attn(GPU2) --|         |->attn(GPU4)
        #           - Only uses 4 GPUs, 1 GPU per layer per tensor head
        #       Overall design of (DistributedNet, ModuleShard) should be able
        #       to facilitate any tensor+pipeline parallel use cases.
        #       ModuleShard should live on worker GPU, with DistributedNet
        #       holding the rref. ModuleShard will contain an arbitrary
        #       nn.Module for the forward pass. ModuleShard should not handle
        #       any torch operations on all-reduced tensors. In this case,
        #       (ex. dropout, linear projections, etc.), another object
        #       between DistributedNet and ModuleShard can handle these
        #       collective operations.
        super(ModuleShard, self).__init__()
        self.device = device
        # TODO: This lock may not be needed, since there should NOT be multiple
        #       workers interacting with the same ModuleShard object.
        self._lock = threading.Lock()

        self.net = module(*args, **kwargs).to(self.device)
        logging.info(f"Module shard on device: {self.device} initialized")
        #print(f"Module shard on device: {self.device} initialized")

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.net(x)
        return out

    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.net.parameters()]


class MLP(nn.Module):
    """Non-distributed generalized matrix multiply"""

    def __init__(self, hidden_dim):
        super(MLP, self).__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim * 4)
        # Output dimension is the same as input dimension since all GEMM are
        # all-reduced, not gathered
        self.fc2 = nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        self.nonlin = F.gelu

    def forward(self, x):
        xa = self.fc1(x)
        y = self.nonlin(xa)
        z = self.fc2(y)
        return z


class SelfAttention(nn.Module):
    """Non-distributed self attention module"""

    def __init__(self, input_data_dim, hidden_dim, msa_heads, qkv_dropout, msa_dropout):
        super(SelfAttention, self).__init__()
        self.input_data_dim = input_data_dim
        self.hidden_dim = hidden_dim

        self.msa_heads = msa_heads

        # hidden dim for this layer is 2 * model dim
        self.ln = nn.LayerNorm(input_data_dim)
        self.fc1 = nn.Linear(input_data_dim, hidden_dim * 3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.qkv_dropout = nn.Dropout(p=qkv_dropout)
        self.msa_dropout = nn.Dropout(p=msa_dropout)

    def forward(self, x):
        B, S, E = x.shape
        x = self.ln(x)
        qkv = self.fc1(x)
        qkv_heads = qkv.reshape(self.msa_heads, B, S, -1)
        q, k, v = torch.split(qkv_heads, qkv_heads.shape[-1] // 3, dim=-1)
        qk_heads = torch.einsum("hbse,hbSe->hbsS", q, k) * (self.input_dim * 2) ** -0.5
        att_heads = F.softmax(qk_heads, dim=-1)
        att_heads = self.dropout(att_heads)
        full_att = torch.einsum("hbsS,hbSe->hbse", att_heads, v)
        full_att = full_att.reshape(B, S, -1)
        out = self.fc2(full_att)
        out = self.msa_dropout(out)
        return out


class SimpleTransformer(nn.Module):
    """1 or 2 layer transformer to test model/data parallel usage of ModuleShard"""

    def __init__(
        self,
        world_size,
        num_heads,
        num_layers,
        num_splits,
        input_data_dim,
        model_hidden_dim,
        msa_heads,
        qkv_dropout,
        msa_dropout,
        mlp_dropout,
    ):
        self.world_size = world_size
        self.num_heads = num_heads
        self.num_layers = num_layers  # each layer is sharded for pipeline
        self.num_splits = num_splits

        assert self.world_size >= self.num_heads * self.num_layers

        self.input_data_dim = input_data_dim
        self.model_hidden_dim = model_hidden_dim
        self.msa_heads = msa_heads
        self.qkv_dropout = qkv_dropout
        self.msa_dropout = msa_dropout
        self.mlp_dropout = mlp_dropout

        # Each worker gets one device in the mesh
        self.workers_mesh = [
            [
                f"worker{j + 1}"
                for j in range(self.num_heads * i, self.num_heads * (i + 1))
            ]
            for i in range(self.num_layers)
        ]
        self.devices_mesh = [
            [f"cuda:{j + 1}" for j in range(self.num_heads * i, self.num_heads * (i + 1))]
            for i in range(self.num_layers)
        ]

        logging.info(f"Workers mesh: {self.workers_mesh}")
        logging.info(f"Device mesh: {self.devices_mesh}")
        #print(f"Workers mesh: {self.workers_mesh}")
        #print(f"Device mesh: {self.devices_mesh}")

        self.msa_layer_rrefs, self.mlp_layer_rrefs = self.make_mesh_by_layer()

    def parameter_rrefs(self):
        remote_params = []
        for msa_layer, mlp_layer in zip(self.msa_layer_rrefs, self.mlp_layer_rrefs):
            for msa_rref in msa_layer:
                remote_params.extend(msa_rref.remote().parameter_rrefs().to_here())
            for mlp_rref in mlp_layer:
                remote_params.extend(mlp_rref.remote().parameter_rrefs().to_here())
        return remote_params

    def make_mesh_by_layer(self):
        """Create a mesh using the given hparams."""
        msa_layer_rrefs = {}
        mlp_layer_rrefs = {}
        for layer_idx, (devices, workers) in enumerate(zip(self.devices_mesh, self.workers_mesh)):
            msa_shards, mlp_shards = self._make_shards(workers, devices, layer_idx)

            msa_layer_rrefs[f"layer{layer_idx}"] = msa_shards
            mlp_layer_rrefs[f"layer{layer_idx}"] = mlp_shards

        return msa_layer_rrefs, mlp_layer_rrefs

    def _make_shards(self, workers, devices, layer_idx):
        """Create ModuleShards on self.workers for both tensor and pipeline
        parallel, and return the shard rrefs.
        """
        msa_shards = {}
        mlp_shards = {}
        for worker, device in zip(workers, devices):
            msa_worker_rref = rpc.remote(
                worker,
                ModuleShard,
                args=(device, SelfAttention)
                + (
                    self.input_data_dim if layer_idx == 0 else self.model_hidden_dim,
                    self.model_hidden_dim,
                    self.msa_heads,
                    self.qkv_dropout,
                    self.msa_dropout,
                ),
            )
            msa_shards[f"{worker}-{device}"] = msa_worker_rref

            mlp_worker_rref = rpc.remote(
                worker,
                ModuleShard,
                args=(device, MLP) + (self.model_hidden_dim,),
            )
            mlp_shards[f"{worker}-{device}"] = mlp_worker_rref

        return msa_shards, mlp_shards

    def forward(self, xs):
        """Define forward mesh propagation"""
        pipeline_futures = []
        for x in iter(xs.split(self.num_splits, dim=0)):
            x_rref = rpc.RRef(x)

            for layer_idx in range(self.num_layers):
                msa_futures = []
                for head_shard in self.msa_layer_rrefs[f"layer{layer_idx}"]:
                    x_rref = rpc.RRef(x)
                    y_rref = head_shard.rpc_async().forward(x_rref)
                    msa_futures.append(y_rref)
                msa_out = torch.cat(torch.futures.wait_all(msa_futures), dim=-1)

                mlp_futures = []
                for mlp_shard in self.mlp_layer_rrefs[f"layer{layer_idx}"]:
                    x_rref = rpc.RRef(msa_out)
                    y_rref = head_shard.rpc_async().forward(x_rref)
                    mlp_futures.append(y_rref)
                mlp_out = torch.sum(torch.stack(torch.futures.wait_all(mlp_futures)), dim=0)

                x = rpc.RRef(mlp_out)

                pipeline_futures.append(mlp_out)
        output = torch.cat(torch.futures.wait_all(pipeline_futures), dim=0)

        return output


def run_master(transformer_args):
    model = SimpleTransformer(**transformer_args)

    logging.info(f"Distributed model on device: cuda:{torch.cuda.current_device()} initialized")
    logging.warning("End call to run_master")

    loss_fn = nn.MSELoss()
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=1e-3,
    )

    # temporary hparams
    num_batches = 100
    batch_size = 64
    sequence_length = 1024
    hidden_dim = 256

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


def run_worker(rank, world_size, transformer_args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "42069"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "1"
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=64)

    if rank == 0:
        dist.rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options,
        )
        logging.info(f"Master rank {rank} on device {torch.cuda.current_device()} initialized")
        run_master(transformer_args)
    else:
        dist.rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options,
        )
        logging.info(f"Worker rank {rank} initialized")

    rpc.shutdown()


def main():
    world_size = 8

    # hparams
    transformer_args = {
        "world_size": world_size,
        "num_heads": 2,
        "num_layers": 2,
        "num_splits": 2,
        "input_data_dim": 256,
        "model_hidden_dim": 256,
        "msa_heads": 4,
        "qkv_dropout": 0.1,
        "msa_dropout": 0.1,
        "mlp_dropout": 0.1,
    }

    num_workers = transformer_args["num_heads"] * transformer_args["num_layers"]
    assert num_workers <= world_size

    logging.info(f"World size: {world_size}")
    logging.info(f"Spawning {num_workers + 1} workers")

    mp.spawn(
        run_worker,
        args=(world_size, transformer_args),
        nprocs=world_size,  # TODO: Why does this need to always be world_size?
        join=True,
    )


if __name__ == "__main__":
    main()
