"""QLoRA SFT helpers for Colab T4 (transformers + TRL + PEFT + bitsandbytes)."""

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


def formatting_prompts_func(example: dict[str, Any], tokenizer, system_prompt: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(example["messages"])
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def run_sft(
    *,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    pilot: bool = False,
) -> str:
    """Train and return adapter output directory."""
    from datasets import load_dataset
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import TrainingArguments
    from trl import SFTTrainer

    model, tok = load_model_and_tokenizer(model_cfg)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, build_lora_config(train_cfg))

    train_ds = load_dataset("json", data_files=train_cfg["data"]["train_file"], split="train")
    eval_file = train_cfg["data"].get("eval_file")
    eval_ds = (
        load_dataset("json", data_files=eval_file, split="train") if eval_file else None
    )

    system_prompt = model_cfg.get("system_prompt", "").strip()

    def _format(ex):
        return formatting_prompts_func(ex, tok, system_prompt)

    out_dir = train_cfg["pilot"]["output_dir"] if pilot else train_cfg["output_dir"]
    t = dict(train_cfg["trainer"])
    if pilot:
        t["max_steps"] = train_cfg["pilot"]["max_steps"]
        t["num_train_epochs"] = 1
        t["eval_strategy"] = "no"
        t["save_strategy"] = "steps"
        t["save_steps"] = max(1, train_cfg["pilot"]["max_steps"])

    args = TrainingArguments(
        output_dir=out_dir,
        **{k: v for k, v in t.items() if k != "output_dir"},
        seed=train_cfg.get("seed", 42),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        formatting_func=_format,
        max_seq_length=train_cfg["sft"]["max_seq_length"],
        packing=train_cfg["sft"].get("packing", False),
        args=args,
    )
    trainer.train()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir
