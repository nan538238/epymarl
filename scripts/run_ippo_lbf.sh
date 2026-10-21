#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${EPYMARL_PYTHON:-$HOME/.venvs/epymarl/bin/python}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy
if [[ -d "$HOME/.local/lib/nvidia" ]]; then
  export LD_LIBRARY_PATH="$HOME/.local/lib/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable; check driver libraries"; print("GPU:", torch.cuda.get_device_name(0))'
exec "$PYTHON" src/main.py --config=ippo --env-config=gymma with   env_args.key=lbforaging:Foraging-8x8-2p-3f-v3 env_args.time_limit=50   batch_size_run=2 batch_size=2 buffer_size=2 test_nepisode=4   t_max=2000 test_interval=1000 log_interval=1000   use_cuda=True use_wandb=False use_tensorboard=False save_model=True seed=1 "$@"
