#!/usr/bin/env bash
# torchrun --no-python entry used when SCORECARD_PIN_GPU=1 (run_scorecard.py).
# Pins each rank to its own GPU before CUDA starts, so every library, worker
# thread and default "cuda" device of the rank lands on that GPU; without it,
# cudaSetDevice is per host thread and stray allocations pile up on GPU 0,
# which runs local rank 0 out of memory in large jobs.  PhysicsNeMo derives
# cuda:{LOCAL_RANK}, so with one visible device the local rank is 0.
export CUDA_VISIBLE_DEVICES="${LOCAL_RANK:-0}"
export LOCAL_RANK=0
exec python "$@"
