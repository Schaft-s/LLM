#!/usr/bin/env python3
"""Local smoke: imports + SFTConfig/SFTTrainer kwargs for the Colab pin set."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    import peft
    import transformers
    import trl
    from trl import SFTConfig, SFTTrainer

    print(f"transformers={transformers.__version__}")
    print(f"trl={trl.__version__}")
    print(f"peft={peft.__version__}")

    major = int(transformers.__version__.split(".", 1)[0])
    assert major == 4, f"expected transformers 4.x pin, got {transformers.__version__}"

    from src.train.sft_qlora import _normalize_warmup_args, load_yaml

    cfg = load_yaml(ROOT / "configs/train.yaml")
    t = _normalize_warmup_args(dict(cfg["trainer"]))
    assert "warmup_ratio" in t, t
    assert "warmup_steps" not in t or isinstance(t.get("warmup_steps"), int), t

    sft = cfg["sft"]
    max_length = sft.get("max_length", sft.get("max_seq_length", 1024))
    args = SFTConfig(
        output_dir="/tmp/rk1_smoke_sft",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        max_steps=1,
        logging_steps=1,
        report_to="none",
        fp16=False,
        bf16=False,
        max_length=max_length,
        packing=False,
        **{k: v for k, v in t.items() if k in {
            "warmup_ratio", "warmup_steps", "lr_scheduler_type", "optim",
            "eval_strategy", "save_strategy", "seed",
        } and k not in {"optim"}},  # optim may need bnb; skip in smoke
        seed=cfg.get("seed", 42),
    )
    print("SFTConfig OK", type(args).__name__, "max_length=", getattr(args, "max_length", None))

    sig = inspect.signature(SFTTrainer.__init__)
    params = set(sig.parameters)
    assert "processing_class" in params, params
    assert "tokenizer" not in params or "processing_class" in params
    assert "peft_config" in params
    print("SFTTrainer params OK: processing_class, peft_config present")

    # Import project training entry without launching train
    from src.train import sft_qlora  # noqa: F401
    from src.train.generate import generate_chat, load_base_or_adapter  # noqa: F401

    print("SMOKE_OK")


if __name__ == "__main__":
    main()
