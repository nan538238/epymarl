#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy
export LD_LIBRARY_PATH="$HOME/.local/lib/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PYTHON="${EPYMARL_PYTHON:-$HOME/.venvs/epymarl/bin/python}"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
exec "$PYTHON" src/main.py --config=ippo --env-config=lbf_feedback with batch_size_run=4 batch_size=4 buffer_size=4 use_cuda=True use_wandb=False use_tensorboard=False save_model=True save_model_interval=20000 "$@"
