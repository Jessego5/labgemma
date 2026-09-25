"""
Error analysis: where the model fails, and whether the failures have a shape.

    python -m pipeline.errors preds_a2.json --out errors_a2.md
    python -m pipeline.errors preds_a2.json --compare preds_a2_masked.json

An AUROC hides everything about *which* examples are wrong. This groups
failures along the axes the project already has evidence for -- compound vs
single-panel figures, caption length, negative type, and how much visually
checkable content the caption carries -- and dumps the worst cases with their
captions so they can actually be read.

Two failure directions, and they mean different things:

  false positive   the model accepted a caption that belongs to a DIFFERENT
                   figure. On a same-paper negative this is the interesting
                   one: the two figures were too similar to separate.

  false negative   the model rejected a figure's own caption. Often a caption
                   dominated by experimental context with nothing visually
                   checkable in it -- the weak-label problem showing up as an
                   error rather than as a statistic.
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict

from . import config
from .config import HIGH_KEYWORDS

VISUAL_ANCHORS = [
    r"\b\d+\s*kda\b", r"\blane[s]?\b", r"\bladder\b", r"\bmarker\b",
    r"\b(gapdh|β-actin|beta-actin|b-actin|α-tubulin|actin)\b",
] + [re.escape(k) for k in HIGH_KEYWORDS]

NONVISUAL = [
    r"\bp\s*[<>=]", r"\bn\s*=\s*\d", r"±",
    r"\b(higher|lower|increase[ds]?|decrease[ds]?|significant\w*|"
    r"correlat\w+|suggest\w*|indicat\w+)\b",
    r"\b(patient|cohort|mice|mouse|rat|cell line|tissue|biopsy)\b",
]


def caption_of(row):
    q = row.get("question", "")
    return q.split("Caption:", 1)[1].strip() if "Caption:" in q else q


def anchor_count(text):
    t = (text or "").lower()
    return sum(1 for p in VISUAL_ANCHORS if re.search(p, t))


def nonvisual_count(text):
    t = (text or "").lower()
    return sum(1 for p in NONVISUAL if re.search(p, t))


def load_preds(path, test_rows):
    """Join predictions back to their captions via (pmcid, figure_id, label)."""
    with open(path) as f:
        preds = json.load(f)
    # test.csv order matches the prediction order, and both come from the same
    # frozen file, so index alignment is exact.
    if len(preds) != len(test_rows):
        raise SystemExit(
            f"{path}: {len(preds):,} predictions vs {len(test_rows):,} test "
            f"rows -- these were not produced from the same test.csv")
    out = []
    for p, r in zip(preds, test_rows):
        cap = caption_of(r)
        out.append({**p, "caption": cap, "words": len(cap.split()),
                    "anchors": anchor_count(cap),
                    "nonvisual": nonvisual_count(cap)})
    return out


def rate(rows, thr):
    """Error rate for a group."""
    if not rows:
        return None
    bad = sum(1 for r in rows
              if (r["score"] >= thr) != (r["label"] == 1))
    return bad / len(rows), len(rows)


def bucket_table(preds, thr, key, buckets, title, out):
    out.append(f"\n### {title}\n")
    out.append("| group | n | error rate |")
    out.append("|---|---|---|")
    for name, pred in buckets:
        sub = [p for p in preds if pred(p)]
        r = rate(sub, thr)
        if r:
            out.append(f"| {name} | {r[1]:,} | {r[0]:.1%} |")


def worst(preds, thr, want_label, n, neg_type=None):
    """Most confidently wrong examples in one direction."""
    sub = [p for p in preds if p["label"] == want_label]
    if neg_type:
        sub = [p for p in sub if p["neg_type"] == neg_type]
    # wrong and confident: positives scored low, negatives scored high
    sub.sort(key=lambda p: p["score"] if want_label == 1 else -p["score"])
    return [p for p in sub
            if (p["score"] >= thr) != (want_label == 1)][:n]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds")
    ap.add_argument("--test", default=str(config.DATA_ROOT / "task" / "test.csv"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--examples", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    with open(args.test, newline="", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))
    preds = load_preds(args.preds, test_rows)
    thr = args.threshold

    name = args.preds.split("/")[-1].replace("preds_", "").replace(".json", "")
    out = [f"# Error analysis — {name}\n",
           f"{len(preds):,} test rows, threshold {thr}\n"]

    overall = rate(preds, thr)
    out.append(f"Overall error rate: **{overall[0]:.1%}** ({overall[1]:,} rows)\n")

    bucket_table(preds, thr, None, [
        ("positives", lambda p: p["label"] == 1),
        ("easy negatives", lambda p: p["neg_type"] == "easy"),
        ("hard negatives (same paper)", lambda p: p["neg_type"] == "hard"),
    ], "By example type", out)

    bucket_table(preds, thr, None, [
        ("compound (multi-panel)", lambda p: str(p.get("compound_figure","")).lower() == "true"),
        ("single-panel", lambda p: str(p.get("compound_figure","")).lower() == "false"),
    ], "By figure type", out)

    qs = sorted(p["words"] for p in preds)
    q1, q3 = qs[len(qs)//4], qs[3*len(qs)//4]
    bucket_table(preds, thr, None, [
        (f"short (<{q1} words)", lambda p: p["words"] < q1),
        (f"medium ({q1}-{q3})", lambda p: q1 <= p["words"] <= q3),
        (f"long (>{q3} words)", lambda p: p["words"] > q3),
    ], "By caption length", out)

    bucket_table(preds, thr, None, [
        ("no visual anchor", lambda p: p["anchors"] == 0),
        ("1-2 visual anchors", lambda p: 1 <= p["anchors"] <= 2),
        ("3+ visual anchors", lambda p: p["anchors"] >= 3),
    ], "By visually checkable content in the caption", out)

    bucket_table(preds, thr, None, [
        ("context-dominated (nonvisual > anchors)",
         lambda p: p["nonvisual"] > p["anchors"]),
        ("balanced or visual", lambda p: p["nonvisual"] <= p["anchors"]),
    ], "By caption character", out)

    def dump(title, rows, note):
        out.append(f"\n### {title}\n\n_{note}_\n")
        if not rows:
            out.append("\n(none)\n")
            return
        for p in rows:
            out.append(f"\n**PMC{p['pmcid']} {p['figure_id']}** — "
                       f"score {p['score']:.3f}"
                       f"{' · compound' if str(p.get('compound_figure','')).lower()=='true' else ''}"
                       f" · {p['words']}w · {p['anchors']} anchors\n")
            out.append(f"> {p['caption'][:400]}\n")

    dump("Worst false positives — same-paper negatives",
         worst(preds, thr, 0, args.examples, "hard"),
         "The model accepted a caption from a different figure in the SAME "
         "paper. These are the pairs it could not separate.")
    dump("Worst false positives — cross-paper negatives",
         worst(preds, thr, 0, args.examples, "easy"),
         "Accepted a caption from an unrelated paper. These should be easy; "
         "anything here is a genuine confusion.")
    dump("Worst false negatives",
         worst(preds, thr, 1, args.examples),
         "The model rejected a figure's own caption. Look for captions that "
         "describe experimental context with nothing visually checkable.")

    text = "\n".join(out) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
