"""Load an A/B judge (a chat model, optionally with a LoRA adapter) and score prompts in batches.

The judge answers "A" or "B". Its score for a prompt is P(A) = sigmoid(logit_A - logit_B) at the first answer
token, i.e. the probability of A when only A or B is allowed. One forward pass per prompt, no generation.
Used by eval_judge.py (step 0 gate) and rank_pool.py (step 2), so both score the same way.
Heavy imports happen inside the functions, so importing this module needs no torch.
"""
from __future__ import annotations  # cluster venv was Python 3.9: keeps `X | None` hints unevaluated

import math


def pick_dtype(name: str):
    """auto: bfloat16 where the GPU supports it (H200, not V100), float16 otherwise, float32 on CPU."""
    import torch
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load(base: str, adapter: str | None = None, dtype: str = "auto"):
    """(tokenizer, model) in eval mode, on the GPU when there is one. Left padding: in a batch, every row's
    last position is then its real last token, which is where the answer's logits are read."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(base, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # straight to the GPU: a 32B teacher is ~65 GB, more than a job's RAM if it is loaded on the CPU first
    model = AutoModelForCausalLM.from_pretrained(base, dtype=pick_dtype(dtype), device_map=device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return tok, model.eval()


def answer_ids(tok) -> tuple[int, int]:
    """Token ids of "A" and "B" as the first answer token. They must be single tokens, or the score is wrong."""
    ids = [tok.encode(x, add_special_tokens=False) for x in ("A", "B")]
    if any(len(i) != 1 for i in ids):
        raise ValueError(f"'A'/'B' are not single tokens for this tokenizer: {ids}")
    return ids[0][0], ids[1][0]


def make_scorer(tok, model, batch_size: int = 16):
    """score(prompts) -> list of P(A), in input order. `prompts` are user-message strings. Batches are
    length-sorted; on out-of-memory a batch is split in half, and a single prompt that still doesn't fit
    scores NaN instead of killing the run."""
    import torch

    id_a, id_b = answer_ids(tok)
    device = next(model.parameters()).device

    def run(texts):
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            z = model(**enc).logits[:, -1].float()
        return torch.sigmoid(z[:, id_a] - z[:, id_b]).tolist()

    def safe(texts):
        try:
            return run(texts)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(texts) == 1:
                return [math.nan]
            mid = len(texts) // 2
            return safe(texts[:mid]) + safe(texts[mid:])

    def score(prompts: list[str]) -> list[float]:
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True) for p in prompts]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [math.nan] * len(texts)
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            for i, p in zip(idx, safe([texts[i] for i in idx])):
                out[i] = p
        return out

    return score
