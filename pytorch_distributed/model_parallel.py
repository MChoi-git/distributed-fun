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

    def parameter_count(self):
        return self.net.parameter_count()

    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.net.parameters()]


class AsyncTransformerLayer(nn.Module):
    def __init__(self, msa_worker_rrefs, mlp_worker_rrefs, layer_idx):
        super(AsyncTransformerLayer, self).__init__()
        self.msa_worker_rrefs = msa_worker_rrefs
        self.mlp_worker_rrefs = mlp_worker_rrefs
        self.layer_idx = layer_idx
        self.lock = threading.Lock()

    @rpc.functions.async_execution
    def forward(self, x_rref):
        # TODO: Add residual connection (test non-residual first)
        fut = torch.futures.Future()
        with self.lock:
            msa_futures = []
            for i, msa_worker in enumerate(self.msa_worker_rrefs):
                y_msa = msa_worker.rpc_async().forward(x_rref)
                msa_futures.append(y_msa)
            msa_out = torch.cat(torch.futures.wait_all(msa_futures), dim=-1)

            mlp_futures = []
            for j, mlp_worker in enumerate(self.mlp_worker_rrefs):
                msa_out_rref = rpc.RRef(msa_out)
                y_mlp = mlp_worker.rpc_async().forward(msa_out_rref)
                mlp_futures.append(y_mlp)
            mlp_out = torch.sum(torch.stack(torch.futures.wait_all(mlp_futures)), dim=0)

            fut.set_result(mlp_out)
        return fut

    def parameter_count(self):
        params_count = 0
        for msa_worker in self.msa_worker_rrefs:
            params_count += msa_worker.remote().parameter_count().to_here()
        for mlp_worker in self.mlp_worker_rrefs:
            params_count += mlp_worker.remote().parameter_count().to_here()
        return params_count

    def parameter_rrefs(self):
        remote_params = []
        for msa_worker in self.msa_worker_rrefs:
            remote_params.extend(msa_worker.remote().parameter_rrefs().to_here())
        for mlp_worker in self.mlp_worker_rrefs:
            remote_params.extend(mlp_worker.remote().parameter_rrefs().to_here())
        return remote_params


class AsyncLayer(nn.Module):
    def __init__(self, worker_rrefs, layer_idx):
        super(AsyncLayer, self).__init__()
        self.worker_rrefs = worker_rrefs
        self.layer_idx = layer_idx
        self.lock = threading.Lock()

    @rpc.functions.async_execution
    def forward(self, x_rref):

        x_input = x_rref.to_here()
        fut = torch.futures.Future()
        with self.lock:
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

        return out

    def parameter_count(self):
        params_count = 0
        for layer_rref in self.async_layer_rrefs:
            params_count += layer_rref.remote().parameter_count().to_here()
        return params_count

    def parameter_rrefs(self):
        remote_params = []
        for layer_rref in self.async_layer_rrefs:
            remote_params.extend(layer_rref.remote().parameter_rrefs().to_here())
        return remote_params
