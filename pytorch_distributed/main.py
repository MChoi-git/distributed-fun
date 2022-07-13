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


def construct_modules():
    raise NotImplementedError


def run_master(args):
    logging.info(
        f"Running master on node: {args.node_rank} - ",
        f"with args: {args}",
    )


def run_worker(rank, args):
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=128,
        init_method="env://",
        _transports=["uv"],
    )

    global_rank = args.node_rank * args.local_world_size + args.rank

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    logging.info(
        f"Running global rank: {global_rank} - "
        f"Master addr: {args.master_addr} - "
        f"Master port: {args.master_port}"
    )

    if global_rank == 0:
        rpc.init_rpc(
            "master",
            rank=global_rank,
            world_size=args.nnodes * args.nproc_per_node,
            rpc_backend_options=options,
        )
        logging.info(f"Master on rank {global_rank} initialized")
        run_master(args)

    else:
        rpc.init_rpc(
            f"worker{global_rank}",
            rank=global_rank,
            world_size=args.nnodes * args.nproc_per_node,
            rpc_backend_options=options,
        )
        logging.info(f"Worker on rank {global_rank} initialized")

    rpc.shutdown()


def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser")

    # Distributed args
    parser.add_argument("--nnodes", type=int)  # Number of nodes
    parser.add_argument("--node_rank", type=int)  # Current node rank
    parser.add_argument("--nproc_per_node", type=int)  # Number of workers/GPUs per node
    parser.add_argument("--master_addr", type=str)
    parser.add_argument("--master_port", type=str)
    parser.add_argument("--slurm_job_id", type=str)

    # Master args (train/model)
    parser.add_argument("--tensor_shard_heads", type=int, default=1)
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--msa_heads", type=int, default=4)
    parser.add_argument("--qkv_dropout", type=float, default=0.1)
    parser.add_argument("--msa_dropout", type=float, default=0.1)

    args = parser.parse_args()
    return args


def main(args):
    logging.basicConfig(
        filename=f"distributed_transformer_{args.slurm_job_id}.log",
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    logger = getLogger(__name__)

    assert args.nproc_per_node == torch.cuda.device_count()

    logger.info(
        f"Total world size: {args.nnodes * args.nproc_per_node} - "
        f"Number of nodes: {args.nnodes} - "
        f"Node rank: {args.node_rank} - "
        f"GPUs: {args.nproc_per_node}"
    )

    # Every worker including master gets the cmdline args
    worker_args = vars(args)
    mp.spawn(run_worker, args=worker_args, nprocs=args.nproc_per_node, join=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
