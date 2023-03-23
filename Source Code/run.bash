#!/bin/bash



SCRIPT=test.py
#FLAGS=--no-cuda
FLAGS=
DATASET=beans
XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_PATH python3 $SCRIPT $FLAGS --dataset $DATASET


