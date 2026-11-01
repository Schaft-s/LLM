#!/usr/bin/env bash
# Safe Colab install: upgrade only our pins, do not force-reinstall the world.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pip install -q --upgrade pip
# Uninstall only packages we pin, so a leftover transformers 5 / peft mismatch is cleared
python -m pip uninstall -y transformers trl peft accelerate bitsandbytes 2>/dev/null || true
python -m pip install -q -r requirements-colab.txt

python - <<'PY'
import transformers, trl, peft, numpy
print("transformers", transformers.__version__)
print("trl", trl.__version__)
print("peft", peft.__version__)
print("numpy", numpy.__version__)
assert transformers.__version__.startswith("4.57"), transformers.__version__
major_np = tuple(int(x) for x in numpy.__version__.split(".")[:2])
assert major_np < (2, 3), f"numpy too new for Colab scipy/sklearn: {numpy.__version__}"
print("INSTALL_OK")
PY
