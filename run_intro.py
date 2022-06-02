import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run_point_2_point(rank, size):
    """ Distributed function to be implemented later. """
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        req = dist.isend(tensor=tensor, dst=1)
    else:
        req = dist.irecv(tensor=tensor, src=0)
    req.wait()
    print('Rank', rank, ' has data ', tensor[0])


def run_all_reduce(rank, size):
    group = dist.new_group([0, 1])
    tensor = torch.ones(2)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank', rank, ' has data ', tensor)


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run_all_reduce))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
