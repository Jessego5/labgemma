"""
Stage 4: turn pairs.csv into the figure-caption MATCHING task.

  pairs.csv  ->  positives + negatives  ->  task/a1/, task/a2/, task/test.csv

pairs.csv is a captioning dataset: every row is a figure and the caption that
belongs to it, so every row is a positive and there is nothing to learn
"match vs no-match" from. This stage manufactures the negatives.

The label is PROVENANCE, not judgement -- nobody looks at the image:

  positive   this caption was published with this figure (same <fig> element)
  negative   we deliberately paired a figure with someone else's caption

Task format is VQA-as-matching, because that is what the training harness
consumes unmodified (image_sub_mode = vqa, columns image_path/question/answer):

  question: "Does the following caption describe this figure? ... Caption: ..."
  answer:   "yes" | "no"

TWO ARMS, and they must stay comparable:

  a1   every positive gets ONE easy negative (caption from a different paper)
  a2   every positive gets ONE hard negative (a different figure from the SAME
       paper); falls back to easy when the article has only one figure

Both arms therefore have identical positives and identical row counts. This
matters more than it looks: if a2 got hard negatives *in addition to* easy
ones it would train on ~50% more rows, and the a1-vs-a2 comparison would be
measuring "more data" rather than "better negatives."

Why hard negatives carry the experiment: an easy negative comes from an
unrelated paper, so it can be rejected on vocabulary alone without looking at
the image. A same-paper negative shares assay, antibodies, cell lines, authors
and figure style -- and because both captions come from one paper, "which
paper is this" gives zero signal, which is what neutralises any memorisation
of the source article from pretraining.

The TEST set is shared by both arms and carries BOTH negative types, tagged in
`neg_type`, so easy-vs-hard can be reported as a breakdown. That gap is the
shortcut-reliance evidence.

Run:
    python -m pipeline.task
    python -m pipeline.task --gallery-size 10
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict

from . import config

PROMPT = ("Does the following caption describe this figure? "
          "Answer yes or no.\n\nCaption: {caption}")

FIELDS = [
    "image_path", "question", "answer",          # consumed by the harness
    "label", "neg_type", "split",                # analysis
    "pmcid", "figure_id",                        # the image's article
    "caption_pmcid", "caption_figure_id",        # where the caption came from
    "tier", "compound_figure",
]


def make_row(fig, caption_row, label, neg_type, split):
    """One task row: `fig`'s image paired with `caption_row`'s caption."""
    return {
        "image_path": str(config.DATA_ROOT / fig["image_path"]),
        "question": PROMPT.format(caption=caption_row["caption"]),
        "answer": "yes" if label else "no",
        "label": int(label),
        "neg_type": neg_type,
        "split": split,
        "pmcid": fig["pmcid"],
        "figure_id": fig["figure_id"],
        "caption_pmcid": caption_row["pmcid"],
        "caption_figure_id": caption_row["figure_id"],
        "tier": fig.get("tier", ""),
        "compound_figure": fig.get("compound_figure", ""),
    }


def pick_easy(fig, pool, rng, tries=20):
    """A caption from a DIFFERENT paper, same split."""
    for _ in range(tries):
        cand = pool[rng.randrange(len(pool))]
        if cand["pmcid"] != fig["pmcid"]:
            return cand
    return None


def pick_hard(fig, same_paper, rng):
    """A different figure from the SAME paper."""
    sibs = [r for r in same_paper if r["figure_id"] != fig["figure_id"]]
    return rng.choice(sibs) if sibs else None


def build_split(rows, split, seed, arm):
    """
    Build one arm's rows for one split: every positive plus exactly one
    negative, so both arms end up the same size.
    """
    rng = random.Random(f"{seed}:{split}:{arm}")
    by_article = defaultdict(list)
    for r in rows:
        by_article[r["pmcid"]].append(r)

    out, counts = [], Counter()
    for fig in rows:
        out.append(make_row(fig, fig, 1, "positive", split))
        neg = None
        if arm == "a2":
            neg = pick_hard(fig, by_article[fig["pmcid"]], rng)
            if neg is not None:
                counts["hard"] += 1
        if neg is None:
            neg = pick_easy(fig, rows, rng)
            if neg is not None:
                counts["easy"] += 1
        if neg is not None:
            out.append(make_row(fig, neg, 0,
                                "hard" if neg["pmcid"] == fig["pmcid"] else "easy",
                                split))
    return out, counts


def build_test(rows, seed):
    """
    Shared eval set: every positive gets BOTH an easy and (where available) a
    hard negative, so easy-vs-hard can be reported as a breakdown.
    """
    rng = random.Random(f"{seed}:test:eval")
    by_article = defaultdict(list)
    for r in rows:
        by_article[r["pmcid"]].append(r)

    out, counts = [], Counter()
    for fig in rows:
        out.append(make_row(fig, fig, 1, "positive", "test"))
        counts["positive"] += 1
        e = pick_easy(fig, rows, rng)
        if e is not None:
            out.append(make_row(fig, e, 0, "easy", "test"))
            counts["easy"] += 1
        h = pick_hard(fig, by_article[fig["pmcid"]], rng)
        if h is not None:
            out.append(make_row(fig, h, 0, "hard", "test"))
            counts["hard"] += 1
    return out, counts


def build_gallery(rows, seed, k):
    """
    Retrieval set for Recall@K: each image against its own caption plus k-1
    distractors from other papers in the same split.
    """
    rng = random.Random(f"{seed}:gallery")
    out = []
    for i, fig in enumerate(rows):
        cands = [fig]
        seen = {fig["pmcid"]}
        for _ in range(200):
            if len(cands) >= k:
                break
            c = rows[rng.randrange(len(rows))]
            if c["pmcid"] not in seen:
                seen.add(c["pmcid"])
                cands.append(c)
        order = list(range(len(cands)))
        rng.shuffle(order)
        for rank, j in enumerate(order):
            c = cands[j]
            out.append({
                "query_id": i,
                "image_path": str(config.DATA_ROOT / fig["image_path"]),
                "candidate_rank": rank,
                "caption": c["caption"],
                "is_correct": int(c is fig),
                "pmcid": fig["pmcid"],
                "figure_id": fig["figure_id"],
            })
    return out


def write(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=str(config.PAIRS))
    ap.add_argument("--out", default=str(config.DATA_ROOT / "task"))
    ap.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    ap.add_argument("--gallery-size", type=int, default=10)
    args = ap.parse_args(argv)

    from pathlib import Path
    out_root = Path(args.out)

    with open(args.pairs, newline="", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    by_split = defaultdict(list)
    for r in pairs:
        by_split[r["split"]].append(r)

    print(f"{len(pairs):,} pairs -> "
          + ", ".join(f"{s} {len(v):,}" for s, v in sorted(by_split.items())))
    print()

    stats = {"pairs": len(pairs), "seed": args.seed, "arms": {}}

    # -- training arms -----------------------------------------------------
    for arm in ("a1", "a2"):
        arm_stats = {}
        for split, fname in (("train", "train.csv"), ("val", "validation.csv")):
            rows, counts = build_split(by_split[split], split, args.seed, arm)
            write(out_root / arm / fname, rows, FIELDS)
            pos = sum(1 for r in rows if r["label"] == 1)
            arm_stats[split] = {"rows": len(rows), "positives": pos,
                                "negatives": len(rows) - pos, **counts}
            print(f"  {arm}/{fname:<15} {len(rows):>7,} rows  "
                  f"({pos:,} pos / {len(rows)-pos:,} neg"
                  f"  hard={counts['hard']:,} easy={counts['easy']:,})")
        stats["arms"][arm] = arm_stats

    # -- shared frozen test set -------------------------------------------
    test_rows, tcounts = build_test(by_split["test"], args.seed)
    write(out_root / "test.csv", test_rows, FIELDS)
    stats["test"] = {"rows": len(test_rows), **tcounts}
    print(f"\n  test.csv{'':<15}{len(test_rows):>7,} rows  "
          f"(pos={tcounts['positive']:,} easy={tcounts['easy']:,} "
          f"hard={tcounts['hard']:,})")

    gal = build_gallery(by_split["test"], args.seed, args.gallery_size)
    write(out_root / "gallery.csv", gal,
          ["query_id", "image_path", "candidate_rank", "caption",
           "is_correct", "pmcid", "figure_id"])
    stats["gallery"] = {"queries": len(by_split["test"]),
                        "candidates_per_query": args.gallery_size,
                        "rows": len(gal)}
    print(f"  gallery.csv{'':<12}{len(gal):>7,} rows  "
          f"({len(by_split['test']):,} queries x {args.gallery_size})")

    # -- invariants --------------------------------------------------------
    print("\ninvariant checks:")
    a1_tr = stats["arms"]["a1"]["train"]["rows"]
    a2_tr = stats["arms"]["a2"]["train"]["rows"]
    same_size = a1_tr == a2_tr
    print(f"  a1/a2 train rows equal      {a1_tr:,} vs {a2_tr:,}"
          f"   {'OK' if same_size else 'MISMATCH'}")

    # A negative must borrow its caption from the SAME split. Negatives are
    # drawn from the split's own row list so this holds by construction, but
    # it is the single failure that would silently invalidate every result --
    # test captions leaking into training -- so verify it against the data.
    split_of = {r["pmcid"]: r["split"] for r in pairs}
    leaks = 0
    for path in [out_root / "a1" / "train.csv", out_root / "a2" / "train.csv",
                 out_root / "a1" / "validation.csv",
                 out_root / "a2" / "validation.csv", out_root / "test.csv"]:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if split_of.get(r["caption_pmcid"]) != r["split"]:
                    leaks += 1
    print(f"  cross-split negatives       {leaks}"
          f"   {'OK' if leaks == 0 else 'LEAK'}")

    hard_frac = tcounts["hard"] / max(tcounts["positive"], 1)
    print(f"  test hard-negative coverage {hard_frac:.0%}")

    (out_root / "task_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nwrote {out_root}")
    print(f"wrote {out_root / 'task_stats.json'}")

    if not same_size:
        print("\n!! Arm sizes differ -- the a1-vs-a2 comparison would be "
              "confounded by dataset size. Do not train until this is fixed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
