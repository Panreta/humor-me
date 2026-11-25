"""Shared pieces of the three training scripts (train_sft, train_dpo, train_ogrpo).

- load_policy: the base model, with an earlier stage's LoRA adapter merged in when given. DPO and O-GRPO start
  from the SFT model this way (HumorGen 4.2-4.3), and train a fresh LoRA on top of it.
- lora: one LoRA shape for every stage, r = 16 on all linear layers (HumorGen Table 19).
- precision: bf16 on GPUs that have it (H200), fp16 on a V100 (no bfloat16), fp32 on CPU (tests).
Heavy imports happen inside the functions, so importing this module needs no torch.
"""
from __future__ import annotations  # cluster venv was Python 3.9: keeps `X | None` hints unevaluated

import os


def precision(name: str = "auto") -> str:
    """'bf16', 'fp16' or 'fp32' for this machine; `name` other than auto forces one."""
    if name != "auto":
        return name
    import torch
    if not torch.cuda.is_available():
        return "fp32"
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


def load_policy(base: str, init_adapter: str | None = None, prec: str = "fp32"):
    """(tokenizer, model). With `init_adapter`, that adapter is merged into the weights, so the result is the
    earlier stage's model as a plain model: a new LoRA trains on top, and with ref_model=None TRL's reference
    (adapter disabled) is that earlier model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[prec]
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
    if init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, init_adapter).merge_and_unload()
    return tok, model


def upcast_trainable(model) -> None:
    """fp16 runs: keep the frozen 7B in fp16 (fp32 would not fit a V100) but hold the LoRA weights in fp32,
    which the fp16 grad scaler needs ("Attempting to unscale FP16 gradients" otherwise)."""
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()


def lora(r: int = 16, alpha: int = 16, dropout: float = 0.05):
    from peft import LoraConfig
    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules="all-linear",
                      task_type="CAUSAL_LM")


def last_checkpoint(out_dir: str) -> str | None:
    """The newest checkpoint in out_dir, so a resubmitted job resumes instead of starting over."""
    from transformers.trainer_utils import get_last_checkpoint
    return get_last_checkpoint(out_dir) if os.path.isdir(out_dir) else None


def default_out(name: str) -> str:
    return os.path.join(os.environ.get("COURSE_DIR", "."), "runs", name)
