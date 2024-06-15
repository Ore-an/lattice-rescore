#!/usr/bin/env bash

# This script functions as a wrapper of a bash command that uses GPUs.
#
# It sets the CUDA_VISIBLE_DEVICES variable so that it limits the number of GPUs
# used for programs. It is neccesary for running a job on the grid if the job
# would automatically grabs all resources available on the system, e.g. a
# TensorFlow program.

num_gpus=1 # this variable indicates how many GPUs we will allow the command
           # passed to this script will run on. We achieve this by setting the
           # CUDA_VISIBLE_DEVICES variable
set -e

vis_dev=()

while [ $1 != "--" ]; do
    vis_dev+=(${1})
    shift
done
shift
job=$1
shift

if [ $# -eq 0 ]; then
  echo "Usage:  $0 <visible_devices> -- <command> [<arg1>...]"
  echo "Runs <command> with args after setting CUDA_VISIBLE_DEVICES to "
  echo "make sure the jobs are distributed evenly to all idle gpus."
  exit 1
fi

time=$(( 10*${job}))
sleep ${time}
min_n_proc=100
min_gpu=${vis_dev[0]}


for gpu in ${vis_dev[@]}; do
    n_proc=$(nvidia-smi -i ${gpu} --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [ $n_proc -lt $min_n_proc ]; then
	min_n_proc=$n_proc
	min_gpu=${gpu}
    fi
done


export CUDA_VISIBLE_DEVICES=${min_gpu}
echo "$0: Running the job on GPU $CUDA_VISIBLE_DEVICES"


"$@"
