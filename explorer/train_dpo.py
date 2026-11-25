"""DPO with a LoRA adapter, for either of two stages (docs/plans/sft-dpo-ogrpo.md).

    --stage judge      Step 0: teach Qwen-7B to answer A/B on rJokes upvote pairs (build_judge_pairs.py output).
                       The settings of the first run (2026-11-14) are the defaults.
    --stage generator  Step 4: align the SFT joke model on judge-ranked (chosen, rejected) lines (build_sets.py
                       output), starting from the SFT adapter. Settings from HumorGen Table 19.

Usage (cluster, repo root; the .sbatch files wrap these):
    python -m explorer.train_dpo --stage judge --train_records 20000
    python -m explorer.train_dpo --stage generator --data_dir runs/sets --init_adapter runs/sft/final

Resumes from the newest checkpoint in --out_dir when one exists, so a job cut by the time limit can be
resubmitted unchanged. Every flag left unset takes the stage's default.
"""
import argparse
import json
import os

STAGES = {
    "judge": dict(data_dir=f"/scratch/{os.environ.get('USER', 'me')}/dpo", train_file="dpo_train.jsonl",
                  dev_file="dpo_dev.jsonl", out_name="dpo_judge_v1", train_records=20000, dev_records=2000,
                  lr=2e-5, scheduler="cosine", warmup=0.05, epochs=1, batch=8, accum=4, lora_alpha=32,
                  max_prompt_tokens=448, patience=0, eval_steps=100),
    "generator": dict(data_dir="runs/sets", train_file="dpo_train.jsonl", dev_file="dpo_dev.jsonl",
                      out_name="dpo_generator_v1", train_records=-1, dev_records=-1,
                      lr=5e-7, scheduler="constant_with_warmup", warmup=0.1, epochs=5, batch=4, accum=4,
                      lora_alpha=16, max_prompt_tokens=0, patience=2, eval_steps=50),
}


def to_chat(rec: dict) -> dict:
    """Judge records hold plain strings; TRL applies the chat template when fields are message lists.
    Generator records are message lists already and pass through."""
    if isinstance(rec["prompt"], list):
        return {k: rec[k] for k in ("prompt", "chosen", "rejected")}
    return {"prompt": [{"role": "user", "content": rec["prompt"]}],
            "chosen": [{"role": "assistant", "content": rec["chosen"]}],
            "rejected": [{"role": "assistant", "content": rec["rejected"]}]}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=STAGES, default="judge")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--init_adapter", default=None, help="generator: the SFT adapter to start from")
    ap.add_argument("--data_dir")
    ap.add_argument("--train_file")
    ap.add_argument("--dev_file")
    ap.add_argument("--out_dir")
    ap.add_argument("--train_records", type=int, help="first n training records (-1 = all)")
    ap.add_argument("--dev_records", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--epochs", type=float)
    ap.add_argument("--batch", type=int, help="per-device batch size")
    ap.add_argument("--accum", type=int, help="gradient accumulation steps")
    ap.add_argument("--eval_steps", type=int)
    ap.add_argument("--max_steps", type=int, default=-1, help="-1 = run the full schedule")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--precision", default="auto", help="auto | bf16 | fp16 | fp32")
    ap.add_argument("--fp16", action="store_true", help="old spelling of --precision fp16")
    args = ap.parse_args(argv)
    for k, v in STAGES[args.stage].items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    if args.out_dir is None:
        from explorer.train_common import default_out
        args.out_dir = default_out(args.out_name)
    if args.fp16:
        args.precision = "fp16"
    if args.stage == "generator" and not args.init_adapter:
        ap.error("--stage generator needs --init_adapter (the SFT adapter): DPO starts from the SFT model")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    from datasets import load_dataset
    from transformers import EarlyStoppingCallback
    from trl import DPOConfig, DPOTrainer

    from explorer.train_common import last_checkpoint, load_policy, lora, precision, upcast_trainable

    prec = precision(args.precision)
    tok, model = load_policy(args.model, args.init_adapter, prec)

    def load(name, n):
        ds = load_dataset("json", data_files=os.path.join(args.data_dir, name), split="train")
        if n > 0:
            ds = ds.select(range(min(n, len(ds))))
        if args.max_prompt_tokens:  # judge: drop very long joke pairs rather than truncate a joke away
            ds = ds.filter(lambda r: len(tok(r["prompt"]).input_ids) <= args.max_prompt_tokens)
        return ds.map(to_chat, remove_columns=ds.column_names)

    train, dev = load(args.train_file, args.train_records), load(args.dev_file, args.dev_records)
    print(f"stage {args.stage}: train {len(train)} records, dev {len(dev)} records, precision {prec}", flush=True)

    cfg = DPOConfig(
        output_dir=args.out_dir, beta=args.beta, learning_rate=args.lr, lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup, per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum, num_train_epochs=args.epochs, max_steps=args.max_steps,
        max_length=512, bf16=prec == "bf16", fp16=prec == "fp16",
        gradient_checkpointing=prec != "fp32", gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10, eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps",
        save_steps=args.eval_steps, save_total_limit=2, report_to="none",
        load_best_model_at_end=args.patience > 0, metric_for_best_model="eval_loss", greater_is_better=False,
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)] if args.patience else []
    trainer = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=train, eval_dataset=dev,
                         processing_class=tok, peft_config=lora(alpha=args.lora_alpha), callbacks=callbacks)
    if prec == "fp16":
        upcast_trainable(trainer.model)

    trainer.train(resume_from_checkpoint=last_checkpoint(args.out_dir))
    final = os.path.join(args.out_dir, "final")
    trainer.save_model(final)
    json.dump({k: v for k, v in vars(args).items()}, open(os.path.join(final, "run_args.json"), "w"), indent=2)
    print(f"saved {final}", flush=True)


if __name__ == "__main__":
    main()
