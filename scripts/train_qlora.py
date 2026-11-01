#!/usr/bin/env python3
"""Run QLoRA SFT (intended for Colab T4)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train.sft_qlora import load_yaml, run_sft


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default=str(ROOT / "configs/model.yaml"))
    parser.add_argument("--train-config", default=str(ROOT / "configs/train.yaml"))
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)
    # Resolve data paths relative to repo root when relative
    for key in ("train_file", "eval_file"):
        p = train_cfg["data"].get(key)
        if p and not Path(p).is_absolute():
            train_cfg["data"][key] = str(ROOT / p)
    for key in ("output_dir",):
        if not Path(train_cfg[key]).is_absolute():
            train_cfg[key] = str(ROOT / train_cfg[key])
    if not Path(train_cfg["pilot"]["output_dir"]).is_absolute():
        train_cfg["pilot"]["output_dir"] = str(ROOT / train_cfg["pilot"]["output_dir"])
    out = run_sft(model_cfg=model_cfg, train_cfg=train_cfg, pilot=args.pilot)
    print(f"Saved adapter to {out}")


if __name__ == "__main__":
    main()
