#!/usr/bin/env python3
"""Generate 1–2 thematic baseline answers from the base model (or dry-run prompts)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Save prompts only")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    model_cfg = yaml.safe_load((ROOT / "configs/model.yaml").read_text(encoding="utf-8"))
    eval_cfg = yaml.safe_load((ROOT / "configs/eval.yaml").read_text(encoding="utf-8"))
    prompts = eval_cfg["baseline_prompts"]
    out_dir = ROOT / "reports/metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_plant_samples.jsonl"

    rows = []
    if args.dry_run:
        for p in prompts:
            rows.append(
                {
                    "id": p["id"],
                    "model": model_cfg["model_id"],
                    "status": "dry_run",
                    "prompt": p["content"].strip(),
                    "response": None,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
            )
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = model_cfg["model_id"]
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        dtype = torch.float16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        # Full precision on MPS/CPU for baseline smoke; Colab uses 4-bit separately.
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device if device != "cpu" else None)
        if device == "cpu":
            model = model.to(device)
        gen = dict(model_cfg["generation"])
        if args.max_new_tokens:
            gen["max_new_tokens"] = args.max_new_tokens
        system = model_cfg.get("system_prompt", "").strip()
        for p in prompts:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": p["content"].strip()})
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=gen.get("max_new_tokens", 512),
                    do_sample=gen.get("do_sample", True),
                    temperature=gen.get("temperature", 0.2),
                    top_p=gen.get("top_p", 0.9),
                    pad_token_id=tok.eos_token_id,
                )
            completion = tok.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
            rows.append(
                {
                    "id": p["id"],
                    "model": model_id,
                    "device": device,
                    "status": "ok",
                    "prompt": p["content"].strip(),
                    "response": completion.strip(),
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
            )

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
