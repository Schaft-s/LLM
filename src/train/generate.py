"""Shared generation for base vs adapter comparison."""

from __future__ import annotations

from typing import Any


def generate_chat(
    model,
    tokenizer,
    user_content: str,
    *,
    system_prompt: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    import torch

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()


def load_base_or_adapter(model_cfg: dict[str, Any], adapter_dir: str | None = None):
    from peft import PeftModel
    from src.train.sft_qlora import load_model_and_tokenizer

    model, tok = load_model_and_tokenizer(model_cfg)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tok
