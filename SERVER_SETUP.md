# Server environment

The development branch currently runs an IPPO / LBF GPU smoke test. This validates the training stack, not the proposed feedback method or UAV task.

## Installation

Reuse the existing Python 3.11 and PyTorch 2.1.1 CUDA 11.8 environment:

```bash
/opt/conda/envs/pytorch-2.1.1/bin/python -m venv --system-site-packages ~/.venvs/epymarl
~/.venvs/epymarl/bin/python -m pip install -r requirements-lbf.txt
bash scripts/run_ippo_lbf.sh
```

The virtual environment inherits the base PyTorch installation; keep that base environment available. Optional SMAClite dependencies are loaded only when that environment is selected.

## GPU driver workaround

The server image had empty libcuda.so.1 and libnvidia-ml.so.1 files. User-local links in ~/.local/lib/nvidia point to the existing versioned 595.84 libraries. The launcher adds that directory to LD_LIBRARY_PATH. System driver files are unchanged. Recheck these links after an image or driver update.

## Validation

The GPU smoke run completed 2100 environment steps (2000 requested; episode batching overshoots). Training updates, evaluation and checkpoint writing completed. This is not convergence evidence.

```bash
bash scripts/run_ippo_lbf.sh t_max=200000 seed=2
```

The launcher uses two workers, seed 1 by default, disables online logging, and saves checkpoints under results/. An early checkpoint is saved during a short run; it should not be mistaken for the final policy. EPYMARL_PYTHON overrides the interpreter path.
