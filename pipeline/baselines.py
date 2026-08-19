"""
Day 3: baselines that require no training.

Run these BEFORE fine-tuning anything. If symbol overlap alone already scores
well, that reframes how every later number should be read -- and finding that
out after a week of training wastes the week.

  random   labels drawn at chance. The floor.

  ocr      Tesseract the image, pull protein/gene-like symbols out of the
           rendered text, and score each caption by symbol overlap. No model,
           no learning, no biology.

The OCR baseline is the load-bearing one. Measured on real captions: 100%
contain protein/gene symbols, and same-paper caption pairs share a median
Jaccard of only 0.31 -- so symbols discriminate between figures within a
paper, and in a blot they are printed right next to the bands. If a trained
model barely beats this, "figure-caption alignment" is mostly name matching,
and that is the finding.

Threshold is tuned on VALIDATION and applied to test, so the reported accuracy
and F1 are not fitted to the test set. AUROC is threshold-free and is the
primary number.

Everything is reported broken out by negative type. The easy-vs-hard gap is
the shortcut-reliance evidence: easy negatives come from unrelated papers and
can be rejected on vocabulary alone, while same-paper hard negatives cannot.

Run:
    python -m pipeline.baselines
    python -m pipeline.baselines --masked     # score against masked images
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict

from . import config
from .ocr import load_cache, symbols


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def auroc(scored):
    """
    Threshold-free AUROC via the rank-sum identity, with ties averaged.
    `scored` is a list of (score, label).
    """
    pos = [s for s, y in scored if y == 1]
    neg = [s for s, y in scored if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(scored, key=lambda t: t[0])
    ranks, i = {}, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][0] == order[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum = sum(ranks[k] for k, (s, y) in enumerate(order) if y == 1)
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def binary_metrics(scored, thr):
    tp = fp = tn = fn = 0
    for s, y in scored:
        pred = 1 if s >= thr else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 0:
            tn += 1
        else:
            fn += 1
    n = max(tp + fp + tn + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"accuracy": (tp + tn) / n, "precision": prec, "recall": rec,
            "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": n}


def best_threshold(scored):
    """
    Threshold maximising F1 on the given (validation) set.

    Returns (threshold, degenerate). F1 maximisation degenerates when most
    scores tie -- as they do once the text is masked out and nearly every
    overlap is 0.0 -- and picks a threshold that predicts a single class.
    That yields recall 1.0 and accuracy at the base rate, which can sit BELOW
    the random baseline and is not a meaningful operating point. Callers
    should report AUROC and ignore the thresholded metrics when degenerate.
    """
    cands = sorted({s for s, _ in scored})
    if not cands:
        return 0.0, True
    best, best_f1 = cands[0], -1.0
    for t in cands:
        f1 = binary_metrics(scored, t)["f1"]
        if f1 > best_f1:
            best, best_f1 = t, f1
    m = binary_metrics(scored, best)
    degenerate = (m["tp"] + m["fp"] == m["n"]) or (m["tp"] + m["fp"] == 0)
    return best, degenerate


# --------------------------------------------------------------------------
# scorers
# --------------------------------------------------------------------------


def caption_of(row):
    """Recover the caption from the VQA prompt."""
    q = row["question"]
    return q.split("Caption:", 1)[1].strip() if "Caption:" in q else q


def ocr_score(row, cache, img_syms):
    """Jaccard overlap between symbols in the image and in the caption."""
    import os
    name = os.path.basename(row["image_path"])
    a = img_syms.get(name)
    if a is None:
        a = symbols(cache.get(name, {}).get("text", ""))
        img_syms[name] = a
    b = symbols(caption_of(row))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def report(name, scored_by_type, thr, out):
    """Print + collect metrics overall and per negative type."""
    all_scored = scored_by_type["all"]
    m = binary_metrics(all_scored, thr)
    a = auroc(all_scored)
    print(f"\n{name}")
    print(f"  {'overall':<10} AUROC {a:.3f}  acc {m['accuracy']:.3f}  "
          f"P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}"
          f"   (n={m['n']:,})")
    entry = {"overall": {**m, "auroc": a, "threshold": thr}}
    for t in ("easy", "hard"):
        sub = scored_by_type.get(t)
        if not sub:
            continue
        ms = binary_metrics(sub, thr)
        as_ = auroc(sub)
        print(f"  {t:<10} AUROC {as_:.3f}  acc {ms['accuracy']:.3f}  "
            f"P {ms['precision']:.3f}  R {ms['recall']:.3f}  F1 {ms['f1']:.3f}"
              f"   (n={ms['n']:,})")
        entry[t] = {**ms, "auroc": as_}
    out[name] = entry
    return entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default=str(config.DATA_ROOT / "task"))
    ap.add_argument("--masked", action="store_true",
                    help="score against text-masked images (the ablation)")
    ap.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    args = ap.parse_args(argv)

    from pathlib import Path
    task = Path(args.task)
    test = load_rows(task / "test.csv")
    # Validation from a2 -- it carries hard negatives, so the threshold is
    # tuned on the same kind of decision the test set asks for.
    val = load_rows(task / "a2" / "validation.csv")
    print(f"test {len(test):,} rows | val {len(val):,} rows (threshold tuning)")

    results = {}

    # -- random ------------------------------------------------------------
    rng = random.Random(args.seed)
    scored = defaultdict(list)
    for r in test:
        s, y = rng.random(), int(r["label"])
        scored["all"].append((s, y))
        if r["neg_type"] in ("easy", "hard"):
            scored[r["neg_type"]].append((s, y))
        elif y == 1:
            scored["easy"].append((s, y))
            scored["hard"].append((s, y))
    report("random", scored, 0.5, results)

    # -- OCR + symbol overlap ---------------------------------------------
    cache = load_cache(args.masked)
    if not cache:
        which = ("ocr extract --masked" if args.masked else "ocr extract")
        print(f"\n!! No OCR cache. Run: python -m pipeline.{which}")
        return 2
    if args.masked:
        print("(scoring against text-MASKED images -- overlap should collapse "
              "toward 0; that is what validates the masking)")

    img_syms = {}
    vscored = [(ocr_score(r, cache, img_syms), int(r["label"])) for r in val]
    thr, degenerate = best_threshold(vscored)
    print(f"\nOCR threshold tuned on validation: {thr:.4f}")
    if degenerate:
        n_zero = sum(1 for s, _ in vscored if s == 0.0)
        print(f"  !! DEGENERATE: this threshold predicts a single class "
              f"({n_zero:,}/{len(vscored):,} validation scores are exactly 0).")
        print(f"     Read AUROC only -- accuracy/P/R/F1 below are not a "
              f"meaningful operating point.")

    scored = defaultdict(list)
    for r in test:
        s, y = ocr_score(r, cache, img_syms), int(r["label"])
        scored["all"].append((s, y))
        if r["neg_type"] in ("easy", "hard"):
            scored[r["neg_type"]].append((s, y))
        elif y == 1:
            scored["easy"].append((s, y))
            scored["hard"].append((s, y))
    ocr_entry = report("ocr_symbol_overlap", scored, thr, results)

    # -- gallery Recall@K --------------------------------------------------
    gal_path = task / "gallery.csv"
    if gal_path.exists():
        gal = load_rows(gal_path)
        by_q = defaultdict(list)
        for r in gal:
            by_q[r["query_id"]].append(r)
        import os
        hits1 = hits5 = 0
        for q, cands in by_q.items():
            name = os.path.basename(cands[0]["image_path"])
            a = img_syms.get(name) or symbols(cache.get(name, {}).get("text", ""))
            ranked = sorted(
                cands,
                key=lambda c: -(len(a & symbols(c["caption"]))
                                / max(len(a | symbols(c["caption"])), 1) if a else 0))
            correct = [i for i, c in enumerate(ranked) if c["is_correct"] == "1"]
            if correct and correct[0] == 0:
                hits1 += 1
            if correct and correct[0] < 5:
                hits5 += 1
        n = max(len(by_q), 1)
        print(f"\nocr gallery retrieval ({n:,} queries x "
              f"{len(next(iter(by_q.values())))} candidates)")
        print(f"  Recall@1 {hits1/n:.3f}   Recall@5 {hits5/n:.3f}")
        results["ocr_symbol_overlap"]["recall@1"] = hits1 / n
        results["ocr_symbol_overlap"]["recall@5"] = hits5 / n

    out = config.DATA_ROOT / ("baselines_masked.json" if args.masked
                              else "baselines.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")

    # -- read the result ---------------------------------------------------
    hard = ocr_entry.get("hard", {}).get("auroc", float("nan"))
    easy = ocr_entry.get("easy", {}).get("auroc", float("nan"))
    print("\ninterpretation:")
    print(f"  easy-negative AUROC {easy:.3f} vs hard-negative {hard:.3f}")
    if easy - hard > 0.10:
        print("  -> large gap: easy negatives are separable on vocabulary alone.")
        print("     Hard negatives are doing their job; report both separately.")
    if hard > 0.70:
        print("  -> symbol overlap ALONE resolves same-paper negatives well.")
        print("     A fine-tuned model must beat this to have learned anything")
        print("     beyond reading labels off the image.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
