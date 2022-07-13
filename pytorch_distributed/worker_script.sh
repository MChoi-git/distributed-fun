#!/bin/bash

echo Hostname: `hostname -s`
echo Node Rank ${SLURM_PROCID}
# prepare environment
source /ssd003/projects/aieng/envs/distributed_env/bin/activate
echo Using Python from: `which python`


if [[ ${SLURM_PROCID} != '0' ]]
then
    echo waiting for 5 seconds for main worker to start first
    sleep 5
fi

env

NUM_GPUs=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`

cmd="torchrun \
	--nnodes ${SLURM_NNODES} \
	--node_rank ${SLURM_NODEID} \
	--nproc_per_node ${NUM_GPUs} \
	--master_addr ${MASTER_ADDR} \
	--master_port ${MASTER_PORT} \
	dummy_worker_script.py
	$* \
	"

#echo $cmd
#eval $cmd
python dummy_worker_script.py --nnodes ${SLURM_NNODES} \
	--node_rank ${SLURM_NODEID} \
	--nproc_per_node ${NUM_GPUs} \
	--master_addr ${MASTER_ADDR} \
	--master_port ${MASTER_PORT} \
	--slurm_job_id ${SLURM_JOB_ID}
