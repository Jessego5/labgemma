"""
Day 6: score prediction files and compare runs side by side.

    python -m pipeline.report preds_zeroshot.json preds_a1.json preds_a2.json
    python -m pipeline.report preds_a2.json preds_a2_masked.json

Reads the JSON written by `pipeline.train score` and reports the same metrics
as pipeline/baselines.py, so a fine-tuned model, the zero-shot base model, and
the no-training OCR baseline all land in one comparable table.

Threshold defaults to 0.5, which is the natural operating point for a
P(yes)/(P(yes)+P(no)) ratio -- unlike the symbol-overlap baseline, this score
is already calibrated around a decision boundary. AUROC remains primary.

Breakdowns reported: easy vs hard negatives (shortcut reliance), and compound
vs single-panel figures (the panel-layout confound -- 72% of the corpus is
compound, and a caption enumerating (A)-(D) against a four-panel image is
matchable without reading anything).
"""

import argparse
import json
import sys
from collections import defaultdict

from .baselines import auroc, binary_metrics


def load(path):
    with open(path) as f:
        return json.load(f)


def slice_metrics(preds, thr):
    scored = [(p["score"], p["label"]) for p in preds]
    if not scored:
        return None
    m = binary_metrics(scored, thr)
    return {"auroc": auroc(scored), **m}


def analyse(preds, thr):
    out = {"overall": slice_metrics(preds, thr)}
    pos = [p for p in preds if p["label"] == 1]
    for t in ("easy", "hard"):
        sub = [p for p in preds if p["neg_type"] == t]
        if sub:
            out[t] = slice_metrics(pos + sub, thr)
    for name, want in (("compound", "true"), ("single", "false")):
        sub = [p for p in preds
               if str(p.get("compound_figure", "")).lower() == want]
        if sub:
            out[name] = slice_metrics(sub, thr)
    return out


CELL = 27


def fmt(m):
    """One cell. Fixed width so columns cannot collide when n grows."""
    if not m:
        return "--".ljust(CELL)
    return (f"{m['auroc']:.3f} {m['accuracy']:.3f} {m['f1']:.3f} "
            f"{m['n']:>6,}").ljust(CELL)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    runs = {}
    for p in args.preds:
        name = p.split("/")[-1].replace("preds_", "").replace(".json", "")
        runs[name] = analyse(load(p), args.threshold)

    keys = ["overall", "easy", "hard", "compound", "single"]
    width = max(len(k) for k in runs) + 2

    print(f"{'':<{width}}" + "".join(k.ljust(CELL) for k in keys))
    print(f"{'':<{width}}" + "".join(
        "AUROC   acc    F1      n".ljust(CELL) for _ in keys))
    print("-" * (width + CELL * len(keys)))
    for name, res in runs.items():
        print(f"{name:<{width}}" + "".join(fmt(res.get(k)) for k in keys))

    # The two comparisons the project exists to make.
    print("\nkey deltas:")
    names = list(runs)
    for a, b in zip(names, names[1:]):
        for k in ("overall", "hard"):
            ma, mb = runs[a].get(k), runs[b].get(k)
            if ma and mb:
                d = mb["auroc"] - ma["auroc"]
                print(f"  {k:<8} {a} {ma['auroc']:.3f} -> {b} {mb['auroc']:.3f}"
                      f"   {d:+.3f}")

    for name, res in runs.items():
        e, h = res.get("easy"), res.get("hard")
        if e and h:
            print(f"  {name}: easy-hard gap {e['auroc'] - h['auroc']:+.3f}"
                  f"  (large gap = leaning on shortcuts)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(runs, f, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
