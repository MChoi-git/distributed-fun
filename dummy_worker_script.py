import os
import argparse
import logging
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
import torch.autograd.profiler as profiler
import torch.multiprocessing as mp
from torch.distributed.optim import DistributedOptimizer


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)


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


class Dense(nn.Module):
    def __init__(self, rank, local_world_size, in_features, out_features):
        super(Dense, self).__init__()
        self.rank = rank
        self.local_rank = self.rank % local_world_size
        self.in_features = in_features
        self.out_features = out_features
        self.lock = threading.Lock()

        self.device = f"cuda:{self.local_rank}"

        self.fc = nn.Linear(self.in_features, self.out_features).to(self.device)

        logging.info(f"Initialized worker{self.rank} on device {self.device}")

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self.lock:
            out = self.fc(x)
        return out.cpu()

    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.fc.parameters()]


class ModuleShard(nn.Module):
    def __init__(self, rank, local_world_size, module, *args, **kwargs):
        super(ModuleShard, self).__init__()
        self.rank = rank
        self.local_rank = self.rank % local_world_size
        self.lock = threading.Lock()

        self.device = f"cuda:{self.local_rank}"

        self.net = module(*args, **kwargs).to(self.device)

        logging.info(f"Initialized worker{self.rank} on device {self.device}")

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self.lock:
            out = self.net(x)
        return out.cpu()

    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.net.parameters()]


class AsyncLayer(nn.Module):
    def __init__(self, worker_rrefs, layer_idx):
        super(AsyncLayer, self).__init__()
        self.worker_rrefs = worker_rrefs
        self.layer_idx = layer_idx
        self.lock = threading.Lock()

    @rpc.functions.async_execution
    def forward(self, x_rref):

        x_input = x_rref.to_here()
        with self.lock:
            fut = torch.futures.Future()
            x_chunks = torch.chunk(x_input, len(self.worker_rrefs), dim=-1)

            out_futures = []
            for i, worker in enumerate(self.worker_rrefs):
                x_chunk_rref = rpc.RRef(x_chunks[i])
                y = worker.rpc_async().forward(x_chunk_rref)
                out_futures.append(y)

            out = torch.cat(torch.futures.wait_all(out_futures), dim=-1)

            fut.set_result(out)
        return fut

    def parameter_rrefs(self):
        remote_params = []
        for worker in self.worker_rrefs:
            remote_params.extend(worker.remote().parameter_rrefs().to_here())
        return remote_params


class DistributedModel(nn.Module):
    def __init__(self, async_layer_rrefs):
        super(DistributedModel, self).__init__()
        self.async_layer_rrefs = async_layer_rrefs

        # TODO: Remove later
        assert len(self.async_layer_rrefs) == 2

        self.device = torch.cuda.current_device()

        logging.info(f"Initialized DistributedModel on device {self.device}")

    def forward(self, x):
        logging.info("Starting dist forward")
        out_futures = []
        # Pipeline parallel microbatch split
        for x in iter(x.chunk(len(self.async_layer_rrefs), dim=0)):
            x_rref = rpc.RRef(x)

            y_rref = self.async_layer_rrefs[0].remote().forward(x_rref)
            y2 = self.async_layer_rrefs[1].rpc_async().forward(y_rref)

            out_futures.append(y2)

        out = torch.cat(torch.futures.wait_all(out_futures), dim=0)
        logging.info("Finished dist forward")

        #logging.info(f"Profile: {prof.key_averages().table()}")

        return out

    def parameter_rrefs(self):
        remote_params = []
        for layer_rref in self.async_layer_rrefs:
            remote_params.extend(layer_rref.remote().parameter_rrefs().to_here())
        return remote_params


def run_master(local_world_size):
    logging.info("Master runnning~")

    # Place models on workers
    worker1_model_rref = rpc.remote(
        "worker1", Dense, args=(1, local_world_size, 64, 64)
    )
    worker2_model_rref = rpc.remote(
        "worker2", Dense, args=(2, local_world_size, 64, 64)
    )
    worker3_model_rref = rpc.remote(
        "worker3", Dense, args=(3, local_world_size, 128, 1)
    )

    async_layer_rrefs = [worker1_model_rref, worker2_model_rref]
    layer1_shard_rref = rpc.remote("worker1", AsyncLayer, args=(async_layer_rrefs, 0))

    model = DistributedModel(
        async_layer_rrefs=(layer1_shard_rref, worker3_model_rref)
    )

    loss_fn = nn.MSELoss()

    opt = DistributedOptimizer(
        torch.optim.SGD,
        model.parameter_rrefs(),
        lr=3e-3,
    )

    data = torch.randn(100, 64, 128)    # (num_baches, batch_size, features)
    labels = torch.tile(torch.arange(64), (100, 1)).float()

    for i, (batch, y) in enumerate(zip(data, labels)):
        logging.info(f"Processing batch {i}")

        with dist_autograd.context() as context_id:
            outputs = model(batch)

            loss = loss_fn(outputs.squeeze(), y)
            logging.info(f"Batch {i} loss: {loss}")

            dist_autograd.backward(context_id, [loss])

            opt.step(context_id)


def run_worker(rank, node_rank, local_world_size, world_size, master_addr, master_port):
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=128,
        init_method="env://",
        _transports=["uv"],
    )

    global_rank = node_rank * local_world_size + rank

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    logging.info(
        f"Running global rank: {global_rank} | Master addr: {master_addr} | Master port: {master_port}"
    )

    if global_rank == 0:
        rpc.init_rpc(
            "master",
            rank=global_rank,
            world_size=world_size,
            rpc_backend_options=options,
        )
        logging.info(f"Master on rank {global_rank} initialized")
        run_master(local_world_size)

    else:
        rpc.init_rpc(
            f"worker{global_rank}",
            rank=global_rank,
            world_size=world_size,
            rpc_backend_options=options,
        )
        logging.info(f"Worker on rank {global_rank} initialized")

    rpc.shutdown()


def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument("--nnodes", type=int)  # Number of nodes
    parser.add_argument("--node_rank", type=int)  # Current node rank
    parser.add_argument("--nproc_per_node", type=int)  # Number of workers/GPUs per node
    parser.add_argument("--master_addr", type=str)
    parser.add_argument("--master_port", type=str)

    args = parser.parse_args()
    return args


def main(args):
    local_world_size = torch.cuda.device_count()
    assert local_world_size == args.nproc_per_node

    world_size = args.nnodes * args.nproc_per_node

    logging.info(
        f"World size: {world_size} | Number of nodes: {args.nnodes} | Node rank: {args.node_rank}| GPUs: {args.nproc_per_node}"
    )

    mp.spawn(
        run_worker,
        args=(
            args.node_rank,
            local_world_size,
            world_size,
            args.master_addr,
            args.master_port,
        ),
        nprocs=(args.nproc_per_node),
        join=True,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
