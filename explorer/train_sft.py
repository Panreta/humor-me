"""SFT and O-GRPO for the joke model, with a LoRA adapter (docs/plans/sft-dpo-ogrpo.md, step 4).

    --stage sft    Fine-tune on the judge's top lines (build_sets.py sft_*.jsonl). HumorGen Table 19: lr 2e-4
                   linear, warmup 0.03, weight decay 0.01, 3 epochs, global batch 16, LoRA r 16 / alpha 16.
    --stage ogrpo  Offline GRPO (HumorGen 4.3, Appendix H): the same loss with each line weighted by its group
                   softmax weight (ogrpo_*.jsonl "weight" = w * G), starting from the SFT adapter.
                   lr 1e-6 constant, 10 warmup steps, no weight decay, 5 epochs.
Both stop early on dev loss (patience 2) and keep the best checkpoint.

Usage (cluster, repo root):
    python -m explorer.train_sft --stage sft --data_dir runs/sets
    python -m explorer.train_sft --stage ogrpo --data_dir runs/sets --init_adapter $COURSE_DIR/runs/sft_v1/final

Loss: cross-entropy on the answer tokens only, averaged over each line's tokens, times the line's weight
(1 for SFT), averaged over the batch. Per-line averaging keeps long lines from outweighing short ones; HumorGen
does not say which it used. Resumes from the newest checkpoint in --out_dir.
"""
import argparse
import json
import os

STAGES = {
    "sft": dict(train_file="sft_train.jsonl", dev_file="sft_dev.jsonl", out_name="sft_v1", lr=2e-4,
                scheduler="linear", warmup_ratio=0.03, warmup_steps=0, weight_decay=0.01, epochs=3,
                weighted=False),
    "ogrpo": dict(train_file="ogrpo_train.jsonl", dev_file="ogrpo_dev.jsonl", out_name="ogrpo_v1", lr=1e-6,
                  scheduler="constant_with_warmup", warmup_ratio=0.0, warmup_steps=10, weight_decay=0.0,
                  epochs=5, weighted=True),
}


def tokenize(tok, prompt: list[dict], completion: list[dict], max_len: int) -> dict:
    """input_ids for prompt + answer, with labels -100 on the prompt (only the answer is learned)."""
    p = tok.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    full = tok.apply_chat_template(prompt + completion, tokenize=True)
    if full[:len(p)] != p:
        raise ValueError("chat template: prompt tokens are not a prefix of prompt + answer")
    full = full[:max_len]
    return {"input_ids": full, "labels": [-100] * len(p) + full[len(p):]}


def collate(pad_id: int):
    import torch

    def fn(batch: list[dict]) -> dict:
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id)
        labels = torch.full((len(batch), n), -100)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for r, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[r, :k] = torch.tensor(b["input_ids"])
            labels[r, :k] = torch.tensor(b["labels"])
            mask[r, :k] = 1
        return {"input_ids": ids, "attention_mask": mask, "labels": labels,
                "weight": torch.tensor([float(b["weight"]) for b in batch])}

    return fn


def weighted_loss(logits, labels, weight):
    """mean over the batch of weight * (mean answer-token cross-entropy of the line)."""
    import torch.nn.functional as F

    logits, labels = logits[:, :-1].float(), labels[:, 1:]
    tok_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100,
                               reduction="none").view(labels.shape)
    mask = (labels != -100).float()
    per_line = (tok_loss * mask).sum(1) / mask.sum(1).clamp(min=1)
    return (weight.to(per_line.dtype) * per_line).mean()


def make_trainer_class():
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            # compute_loss below ignores num_items_in_batch, so the Trainer must divide by the accumulation steps
            # itself; left True (Qwen's forward takes **kwargs), the effective learning rate is accum x too high.
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            weight, labels = inputs.pop("weight"), inputs.pop("labels")
            out = model(**inputs)
            loss = weighted_loss(out.logits, labels, weight)
            return (loss, out) if return_outputs else loss

    return WeightedTrainer


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=STAGES, default="sft")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--init_adapter", default=None, help="ogrpo: the SFT adapter to start from")
    ap.add_argument("--data_dir", default="runs/sets")
    ap.add_argument("--train_file")
    ap.add_argument("--dev_file")
    ap.add_argument("--out_dir")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--epochs", type=float)
    ap.add_argument("--batch", type=int, default=4, help="per-device batch size")
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation (4 x 4 = HumorGen's 16)")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--precision", default="auto", help="auto | bf16 | fp16 | fp32")
    args = ap.parse_args(argv)
    for k, v in STAGES[args.stage].items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    if args.out_dir is None:
        from explorer.train_common import default_out
        args.out_dir = default_out(args.out_name)
    if args.stage == "ogrpo" and not args.init_adapter:
        ap.error("--stage ogrpo needs --init_adapter (the SFT adapter): O-GRPO starts from the SFT model")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    from datasets import load_dataset
    from peft import get_peft_model
    from transformers import EarlyStoppingCallback, TrainingArguments

    from explorer.train_common import last_checkpoint, load_policy, lora, precision, upcast_trainable

    prec = precision(args.precision)
    tok, model = load_policy(args.model, args.init_adapter, prec)
    if prec != "fp32":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, lora())
    if prec == "fp16":
        upcast_trainable(model)
    model.print_trainable_parameters()

    def load(name):
        ds = load_dataset("json", data_files=os.path.join(args.data_dir, name), split="train")
        return ds.map(lambda r: {**tokenize(tok, r["prompt"], r["completion"], args.max_len),
                                 "weight": r["weight"] if args.weighted else 1.0},
                      remove_columns=ds.column_names)

    train, dev = load(args.train_file), load(args.dev_file)
    print(f"stage {args.stage}: train {len(train)} lines, dev {len(dev)} lines, precision {prec}", flush=True)

    cfg = TrainingArguments(
        output_dir=args.out_dir, learning_rate=args.lr, lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio, warmup_steps=args.warmup_steps, weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum, num_train_epochs=args.epochs, max_steps=args.max_steps,
        bf16=prec == "bf16", fp16=prec == "fp16", logging_steps=10, eval_strategy="steps",
        eval_steps=args.eval_steps, save_strategy="steps", save_steps=args.eval_steps, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        remove_unused_columns=False, prediction_loss_only=True, report_to="none",
    )
    trainer = make_trainer_class()(model=model, args=cfg, train_dataset=train, eval_dataset=dev,
                                   data_collator=collate(tok.pad_token_id),
                                   callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)])
    trainer.train(resume_from_checkpoint=last_checkpoint(args.out_dir))
    final = os.path.join(args.out_dir, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    json.dump(vars(args), open(os.path.join(final, "run_args.json"), "w"), indent=2)
    print(f"saved {final}", flush=True)


if __name__ == "__main__":
    main()
