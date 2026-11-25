"""Batched chat generation with plain transformers (no vLLM on Explorer), for steps 1 and 5.

make_generator(...) -> generate(conversations) -> reply texts, in input order. Batches are length-sorted and
left-padded; on out-of-memory a batch is split in half, and a single conversation that still doesn't fit gets ""
(the caller counts it as a failure) instead of killing the run. Same pattern as label_modes.make_generator, with
sampling and an optional LoRA adapter.
"""
from __future__ import annotations  # cluster venv was Python 3.9: keeps `X | None` hints unevaluated


def make_generator(model: str, adapter: str | None = None, dtype: str = "auto", batch_size: int = 16,
                   temperature: float = 0.9, top_p: float = 0.95, max_new_tokens: int = 256, seed: int = 0):
    import torch

    from explorer.judge_model import load

    tok, lm = load(model, adapter, dtype)
    device = next(lm.parameters()).device
    torch.manual_seed(seed)
    sample = temperature > 0
    kw = dict(do_sample=True, temperature=temperature, top_p=top_p) if sample else dict(do_sample=False)

    def run(texts):
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = lm.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **kw)
        return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def safe(texts):
        try:
            return run(texts)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(texts) == 1:
                return [""]
            mid = len(texts) // 2
            return safe(texts[:mid]) + safe(texts[mid:])

    def generate(convs: list[list[dict]]) -> list[str]:
        texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [""] * len(texts)
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            for i, reply in zip(idx, safe([texts[i] for i in idx])):
                out[i] = reply
        return out

    return generate
