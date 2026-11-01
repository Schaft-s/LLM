#!/usr/bin/env bash
# Safe Colab install: upgrade only our pins, do not force-reinstall the world.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pip install -q --upgrade pip
python -m pip uninstall -y transformers trl peft accelerate bitsandbytes 2>/dev/null || true
python -m pip install -q -r requirements-colab.txt

python - <<'PY'
import glob
from pathlib import Path

import numpy
import peft
import torch
import transformers
import trl

print("transformers", transformers.__version__)
print("trl", trl.__version__)
print("peft", peft.__version__)
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
assert transformers.__version__.startswith("4.57"), transformers.__version__
major_np = tuple(int(x) for x in numpy.__version__.split(".")[:2])
assert major_np < (2, 3), f"numpy too new for Colab scipy/sklearn: {numpy.__version__}"

import bitsandbytes as bnb

bnb_dir = Path(bnb.__file__).resolve().parent
cuda_libs = sorted(bnb_dir.glob("libbitsandbytes_cuda*.so"))
print("bitsandbytes", bnb.__version__)
print("bnb cuda libs:", [p.name for p in cuda_libs] or "(none)")
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available to torch — pick a GPU runtime in Colab")
if not cuda_libs:
    raise SystemExit(
        "bitsandbytes has no CUDA .so (CPU-only wheel). "
        "Install bitsandbytes>=0.49.2 for Colab CUDA 13."
    )
# Prefer an exact/closest match for torch CUDA
torch_cuda = (torch.version.cuda or "").replace(".", "")
wanted = f"libbitsandbytes_cuda{torch_cuda}.so"
names = {p.name for p in cuda_libs}
if wanted not in names and not any(n.startswith("libbitsandbytes_cuda13") for n in names):
    print("WARNING: no exact", wanted, "- will rely on bitsandbytes fallback")
print("INSTALL_OK")
PY
