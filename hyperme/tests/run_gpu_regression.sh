#!/bin/bash
# S2 TC-ME retrain regression (two seeds) on a GPU that no other user is using.
# Waits until such a GPU exists, then runs only on it.  Usage, from the repository root:
#   PY=~/spectralmarl_venv/bin/python bash hyperme/tests/run_gpu_regression.sh
set -e
PY=${PY:-python}
cd "$(dirname "$0")/../.."
GPU=$($PY -m hyperme.gpu_guard --wait | awk '{print $1}')
echo "$(date) using GPU $GPU"
export FH_ROOT=$(pwd) CUDA_VISIBLE_DEVICES=$GPU HYPERME_DEVICE=cuda OMP_NUM_THREADS=8
$PY -m pytest hyperme/tests/test_regression.py -m slow -k "s2 or tcme" -q -rA
