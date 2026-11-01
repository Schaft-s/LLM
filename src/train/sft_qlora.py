"""QLoRA SFT helpers for Colab T4 (transformers v5 + modern TRL + PEFT + bitsandbytes)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_bnb_config(model_cfg: dict[str, Any]):
    from transformers import BitsAndBytesConfig
    import torch

    dtype = getattr(torch, model_cfg.get("bnb_4bit_compute_dtype", "float16"))
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
    )


def load_model_and_tokenizer(model_cfg: dict[str, Any]):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = model_cfg["model_id"]
    revision = model_cfg.get("revision") or None
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = build_bnb_config(model_cfg) if model_cfg.get("load_in_4bit", True) else None
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    model.config.use_cache = False
    return model, tok


def build_lora_config(train_cfg: dict[str, Any]):
    from peft import LoraConfig

    l = train_cfg["lora"]
    return LoraConfig(
        r=l["r"],
        lora_alpha=l["lora_alpha"],
        lora_dropout=l["lora_dropout"],
        bias=l["bias"],
        task_type=l["task_type"],
        target_modules=l["target_modules"],
    )


def _transformers_major() -> int:
    import transformers

    return int(transformers.__version__.split(".", 1)[0])


def _normalize_warmup_args(trainer_cfg: dict[str, Any]) -> dict[str, Any]:
    """transformers v4 wants warmup_ratio; v5 removed it in favor of float warmup_steps."""
    t = dict(trainer_cfg)
    ratio = t.pop("warmup_ratio", None)
    steps = t.get("warmup_steps", None)
    major = _transformers_major()
    if major >= 5:
        if ratio is not None and steps is None:
            t["warmup_steps"] = ratio
        # float warmup_steps is valid on v5; drop legacy key already popped
    else:
        # v4: warmup_steps must be int; restore ratio if we only have a float fraction
        if isinstance(steps, float) and steps < 1:
            t.pop("warmup_steps", None)
            t["warmup_ratio"] = steps
        elif ratio is not None:
            t["warmup_ratio"] = ratio
            if isinstance(steps, float) and steps < 1:
                t.pop("warmup_steps", None)
    return t


def _with_system_prompt(example: dict[str, Any], system_prompt: str) -> dict[str, Any]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(example["messages"])
    return {"messages": messages}


def _prepare_chat_dataset(ds, system_prompt: str):
    """Keep only conversational `messages` (native TRL format)."""
    ds = ds.map(lambda ex: _with_system_prompt(ex, system_prompt))
    drop = [c for c in ds.column_names if c != "messages"]
    if drop:
        ds = ds.remove_columns(drop)
    return ds


def run_sft(
    *,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    pilot: bool = False,
) -> str:
    """Train and return adapter output directory."""
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tok = load_model_and_tokenizer(model_cfg)
    peft_config = build_lora_config(train_cfg)

    train_ds = load_dataset("json", data_files=train_cfg["data"]["train_file"], split="train")
    eval_file = train_cfg["data"].get("eval_file")
    eval_ds = (
        load_dataset("json", data_files=eval_file, split="train") if eval_file else None
    )

    system_prompt = model_cfg.get("system_prompt", "").strip()
    train_ds = _prepare_chat_dataset(train_ds, system_prompt)
    if eval_ds is not None:
        eval_ds = _prepare_chat_dataset(eval_ds, system_prompt)

    out_dir = train_cfg["pilot"]["output_dir"] if pilot else train_cfg["output_dir"]
    t = dict(train_cfg["trainer"])
    t = _normalize_warmup_args(t)

    if pilot:
        t["max_steps"] = train_cfg["pilot"]["max_steps"]
        t["num_train_epochs"] = 1
        t["eval_strategy"] = "no"
        t["save_strategy"] = "steps"
        t["save_steps"] = max(1, train_cfg["pilot"]["max_steps"])
        eval_ds = None

    sft = train_cfg.get("sft", {})
    max_length = sft.get("max_length", sft.get("max_seq_length", 1024))
    packing = bool(sft.get("packing", False))

    sft_kwargs: dict[str, Any] = {
        "output_dir": out_dir,
        **{k: v for k, v in t.items() if k != "output_dir"},
        "seed": train_cfg.get("seed", 42),
        "packing": packing,
    }
    # Prefer modern max_length; fall back if an older TRL still wants max_seq_length only.
    try:
        args = SFTConfig(**sft_kwargs, max_length=max_length, assistant_only_loss=True)
    except TypeError:
        try:
            args = SFTConfig(**sft_kwargs, max_length=max_length)
        except TypeError:
            args = SFTConfig(**sft_kwargs, max_seq_length=max_length)

    # Pass base quantized model + peft_config (do not wrap with get_peft_model first).
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir
