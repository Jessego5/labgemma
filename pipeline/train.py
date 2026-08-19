"""
Days 4-5: LoRA fine-tuning for figure-caption matching, and scoring.

A self-contained transformers + peft path, so the project does not depend on
an external training harness working. Two subcommands:

    python -m pipeline.train fit  --arm a1
    python -m pipeline.train fit  --arm a2
    python -m pipeline.train score --adapter <dir> --out preds_a2.json
    python -m pipeline.train score --out preds_zeroshot.json          # base model
    python -m pipeline.train score --adapter <dir> --masked ...       # ablation

SCORING IS A LOGIT RATIO, NOT GENERATION. For each example we read the logits
at the first answer position and take

    P(yes) / (P(yes) + P(no))

That yields a CONTINUOUS score in [0,1], which is what AUROC needs. Generating
a discrete "yes"/"no" string would throw away the confidence and make AUROC
degenerate to accuracy -- and AUROC is the primary metric precisely because
the operating point is unstable on this data (see pipeline/baselines.py).

Only the answer tokens contribute to the loss; the prompt and image tokens are
masked to -100. Training on the prompt would spend capacity modelling caption
text rather than the yes/no decision.

The vision tower is frozen and LoRA is applied to the language model's
attention projections, per the project plan.
"""

import argparse
import csv
import json
import os
import sys

from . import config

DEFAULT_MODEL = "google/gemma-3-4b-it"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def load_rows(path, limit=None):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def remap_masked(rows):
    """Point image_path at the masked variants, for the Day 6 ablation."""
    mdir = config.DATA_ROOT / "images_masked"
    out = []
    for r in rows:
        p = mdir / os.path.basename(r["image_path"])
        if p.exists():
            r = dict(r, image_path=str(p))
        out.append(r)
    return out


class MatchingDataset:
    """(image, question, yes/no) -> input_ids with loss only on the answer."""

    def __init__(self, rows, processor, max_len=768):
        self.rows = rows
        self.proc = processor
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import torch
        from PIL import Image

        r = self.rows[i]
        img = Image.open(r["image_path"]).convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": r["question"]}]}]

        prompt = self.proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False)
        full = prompt + r["answer"]

        enc = self.proc(text=full, images=img, return_tensors="pt",
                        truncation=True, max_length=self.max_len)
        enc = {k: v[0] for k, v in enc.items()}

        # Mask everything before the answer.
        p_len = len(self.proc(text=prompt, images=img,
                              return_tensors="pt",
                              truncation=True,
                              max_length=self.max_len)["input_ids"][0])
        labels = enc["input_ids"].clone()
        labels[:p_len] = -100
        pad_id = self.proc.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        enc["labels"] = labels
        return enc


def collate(batch, pad_id):
    """Right-pad variable-length sequences; stack fixed-shape image tensors."""
    import torch

    maxlen = max(len(b["input_ids"]) for b in batch)
    out = {}
    for key in ("input_ids", "attention_mask", "labels", "token_type_ids"):
        if key not in batch[0]:
            continue
        fill = {"input_ids": pad_id, "attention_mask": 0,
                "labels": -100, "token_type_ids": 0}[key]
        out[key] = torch.stack([
            torch.cat([b[key],
                       torch.full((maxlen - len(b[key]),), fill,
                                  dtype=b[key].dtype)])
            for b in batch])
    for key in batch[0]:
        if key not in out:
            out[key] = torch.stack([b[key] for b in batch])
    return out


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


def load_model(model_id, adapter=None, train=False, attn="sdpa"):
    """
    Load Gemma 3 with LoRA on the language model, vision tower frozen.

    attn defaults to "sdpa", NOT "eager". Eager attention makes SigLIP
    materialise the full patch-by-patch attention matrix: at 896x896 the
    vision tower sees 4096 patches, so one head is a 4096x4096 fp32 matrix
    (67 MB), and 16 heads at batch 2 is ~2 GB of transient activation -- which
    OOMs a 24 GB card before the language model even runs. SDPA computes the
    same thing without ever materialising it.
    """
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16,
        device_map="auto", attn_implementation=attn)

    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    elif train:
        from peft import LoraConfig, get_peft_model
        # Freeze the vision tower: the plan trains the language side and the
        # projector, not the image encoder.
        for name, p in model.named_parameters():
            if "vision_tower" in name:
                p.requires_grad = False
        cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
        model.config.use_cache = False   # incompatible with grad checkpointing
    return proc, model


def yes_no_ids(proc):
    """Token ids for the answer tokens, with and without a leading space."""
    tok = proc.tokenizer
    def first(s):
        ids = tok.encode(s, add_special_tokens=False)
        return ids[0] if ids else None
    cands = {"yes": [first("yes"), first(" yes"), first("Yes")],
             "no": [first("no"), first(" no"), first("No")]}
    return ({k: sorted({i for i in v if i is not None})
             for k, v in cands.items()})


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------


def cmd_fit(args) -> int:
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    task = config.DATA_ROOT / "task" / args.arm
    rows = load_rows(task / "train.csv", args.limit)
    print(f"arm {args.arm}: {len(rows):,} train rows")
    print(f"  label balance: "
          f"{sum(1 for r in rows if r['answer']=='yes'):,} yes / "
          f"{sum(1 for r in rows if r['answer']=='no'):,} no")
    print(f"  negatives    : "
          f"{sum(1 for r in rows if r['neg_type']=='hard'):,} hard / "
          f"{sum(1 for r in rows if r['neg_type']=='easy'):,} easy")

    proc, model = load_model(args.model, train=True, attn=args.attn)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    ds = MatchingDataset(rows, proc, args.max_len)
    pad_id = proc.tokenizer.pad_token_id or 0
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    collate_fn=lambda b: collate(b, pad_id), num_workers=2)

    steps = (len(dl) // args.accum) * args.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, int(0.03 * steps), steps)

    model.train()
    step = 0
    for ep in range(args.epochs):
        run = 0.0
        for i, batch in enumerate(dl, 1):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(**batch).loss / args.accum
            loss.backward()
            run += loss.item() * args.accum
            if i % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % 20 == 0:
                    print(f"  ep{ep+1} step {step}/{steps} "
                          f"loss {run/i:.4f}", flush=True)
        print(f"epoch {ep+1} mean loss {run/max(len(dl),1):.4f}")

    out = args.out or str(config.DATA_ROOT / "runs" / args.arm)
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    proc.save_pretrained(out)
    print(f"\nsaved adapter -> {out}")
    return 0


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------


def cmd_score(args) -> int:
    import torch

    rows = load_rows(args.eval or str(config.DATA_ROOT / "task" / "test.csv"),
                     args.limit)
    if args.masked:
        rows = remap_masked(rows)
        print("(scoring against text-MASKED images)")
    print(f"{len(rows):,} eval rows")

    proc, model = load_model(args.model, adapter=args.adapter, attn=args.attn)
    model.eval()
    ids = yes_no_ids(proc)
    print(f"yes ids {ids['yes']}  no ids {ids['no']}")

    from PIL import Image
    preds = []
    with torch.no_grad():
        for i, r in enumerate(rows, 1):
            img = Image.open(r["image_path"]).convert("RGB")
            msgs = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": r["question"]}]}]
            prompt = proc.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False)
            enc = proc(text=prompt, images=img, return_tensors="pt",
                       truncation=True, max_length=args.max_len)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            logits = model(**enc).logits[0, -1]          # next-token logits
            probs = torch.softmax(logits.float(), dim=-1)
            py = float(probs[ids["yes"]].sum())
            pn = float(probs[ids["no"]].sum())
            preds.append({
                "score": py / max(py + pn, 1e-9),
                "label": int(r["label"]), "neg_type": r["neg_type"],
                "pmcid": r["pmcid"], "figure_id": r["figure_id"],
                "compound_figure": r.get("compound_figure", ""),
            })
            if i % 200 == 0:
                print(f"  [{i:,}/{len(rows):,}]", flush=True)

    out = args.out or str(config.DATA_ROOT / "preds.json")
    with open(out, "w") as f:
        json.dump(preds, f)
    print(f"\nwrote {out}")
    print(f"score it with: python -m pipeline.report {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--attn", default="sdpa",
                    choices=["sdpa", "eager", "flash_attention_2"])

    # The same options are accepted AFTER the subcommand too, which is where
    # anyone would naturally type them. argparse.SUPPRESS is what makes that
    # safe: without it the subparser's default would overwrite a value given
    # before the subcommand, silently ignoring it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=argparse.SUPPRESS)
    common.add_argument("--max-len", dest="max_len", type=int,
                        default=argparse.SUPPRESS)
    common.add_argument("--limit", type=int, default=argparse.SUPPRESS)
    common.add_argument("--attn", default=argparse.SUPPRESS)

    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", parents=[common])
    f.add_argument("--arm", required=True, choices=["a1", "a2"])
    f.add_argument("--epochs", type=int, default=2)
    # batch 1 x accum 16 keeps the effective batch at 16 while holding only
    # one image's vision activations at a time -- the binding constraint on
    # a 24 GB card is the vision tower, not the language model.
    f.add_argument("--batch", type=int, default=1)
    f.add_argument("--accum", type=int, default=16)
    f.add_argument("--grad-ckpt", action="store_true",
                   help="trade ~30%% speed for a large activation-memory cut")
    f.add_argument("--lr", type=float, default=1e-4)
    f.add_argument("--out", default=None)
    f.set_defaults(func=cmd_fit)

    s = sub.add_parser("score", parents=[common])
    s.add_argument("--adapter", default=None, help="omit for zero-shot base")
    s.add_argument("--eval", default=None)
    s.add_argument("--masked", action="store_true")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_score)

    args = ap.parse_args(argv)
    config.ensure_dirs()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
