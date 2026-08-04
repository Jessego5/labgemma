"""
Stage 3: turn the manifest + downloaded images into a training dataset.

  manifest.csv + download_log.csv  ->  dedup  ->  paper-level split  ->  pairs.csv
                                                                    ->  stats.json

What this stage enforces:

  * ONLY PAIRS THAT ACTUALLY HAVE AN IMAGE. Rows whose download failed are
    dropped here and counted, rather than quietly producing a dataset smaller
    than it claims to be.

  * DEDUPLICATION on exact image bytes (sha256 from the download log) and on
    normalized caption text. Journals reuse figures across corrections and
    republications; identical images landing in both train and test would
    inflate every metric you report.

  * PAPER-LEVEL SPLITTING. Splits are assigned by hashing the PMCID, so every
    figure from one article lands in the same split and no article straddles
    train/val/test. Random per-figure splitting leaks: figures from the same
    paper share visual style, antibodies, and caption phrasing, and a model
    can match on those instead of on content. Hashing also makes the split
    deterministic and stable as the dataset grows -- adding articles never
    reshuffles existing ones.

Run:
    python -m pipeline.dataset
    python -m pipeline.dataset --tiers high medium
    python -m pipeline.dataset --licenses "CC BY" CC0    # commercial-safe only
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict

from . import config

OUT_FIELDS = [
    "pmcid", "figure_id", "caption", "image_path", "license_url", "license",
    "license_class", "keyword_score", "gel_score", "caption_quality",
    "compound_figure", "tier", "split",
]


def norm_caption(c: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", re.sub(r"\s+", " ", (c or "").lower())).strip()


def assign_split(pmcid: str) -> str:
    """
    Deterministic paper-level split from a hash of the PMCID.

    Stable under dataset growth: an article's split never changes when new
    articles are added, so results stay comparable across collection runs.
    """
    h = hashlib.sha256(f"{config.SPLIT_SEED}:{pmcid}".encode()).digest()
    x = int.from_bytes(h[:8], "big") / float(1 << 64)
    cum = 0.0
    for name, frac in config.SPLIT_FRACTIONS.items():
        cum += frac
        if x < cum:
            return name
    return "test"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiers", nargs="*", default=config.DEFAULT_TIERS,
                    help=f"tiers to include (default: {config.DEFAULT_TIERS})")
    ap.add_argument("--licenses", nargs="*", default=None,
                    help="restrict to these license classes, e.g. 'CC BY' CC0")
    ap.add_argument("--min-quality", type=float, default=0.0)
    ap.add_argument("--drop-compound", action="store_true",
                    help="exclude multi-panel figures")
    args = ap.parse_args(argv)

    config.ensure_dirs()
    if not config.MANIFEST.exists():
        print(f"No manifest at {config.MANIFEST}. Run: python -m pipeline.extract")
        return 2

    with open(config.MANIFEST, newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    # Downloaded images, keyed by (pmcid, figure_id).
    have = {}
    if config.DOWNLOAD_LOG.exists():
        with open(config.DOWNLOAD_LOG, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["status"] in ("ok", "cached") and r["image_path"]:
                    have[(r["pmcid"], r["figure_id"])] = r
    else:
        print(f"!! No download log at {config.DOWNLOAD_LOG}. "
              f"Run: python -m pipeline.download")
        return 2

    stats = {"manifest_rows": len(manifest),
             "manifest_articles": len({r["pmcid"] for r in manifest}),
             "downloaded": len(have)}
    drops = Counter()

    rows = []
    for r in manifest:
        key = (r["pmcid"], r["figure_id"])
        if key not in have:
            drops["no_image"] += 1
            continue
        if args.tiers and r.get("tier") not in args.tiers:
            drops["tier"] += 1
            continue
        if args.licenses and r.get("license_class") not in args.licenses:
            drops["license"] += 1
            continue
        if float(r.get("caption_quality") or 0) < args.min_quality:
            drops["low_quality"] += 1
            continue
        if args.drop_compound and str(r.get("compound_figure")).lower() == "true":
            drops["compound"] += 1
            continue
        out = {k: r.get(k, "") for k in OUT_FIELDS if k in r}
        out["image_path"] = f"images/{have[key]['image_path']}"
        out["_sha"] = have[key].get("sha256", "")
        rows.append(out)

    # -- dedup ------------------------------------------------------------
    seen_img, seen_cap, deduped = set(), set(), []
    for r in rows:
        sha = r.pop("_sha", "")
        if sha and sha in seen_img:
            drops["dup_image"] += 1
            continue
        nc = norm_caption(r["caption"])
        if nc in seen_cap:
            drops["dup_caption"] += 1
            continue
        if sha:
            seen_img.add(sha)
        seen_cap.add(nc)
        deduped.append(r)

    # -- split ------------------------------------------------------------
    for r in deduped:
        r["split"] = assign_split(r["pmcid"])

    with open(config.PAIRS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(deduped)

    # -- stats ------------------------------------------------------------
    by_split = Counter(r["split"] for r in deduped)
    arts_by_split = defaultdict(set)
    for r in deduped:
        arts_by_split[r["split"]].add(r["pmcid"])
    overlap = set()
    for a, sa in arts_by_split.items():
        for b, sb in arts_by_split.items():
            if a < b:
                overlap |= sa & sb

    stats.update({
        "kept": len(deduped),
        "articles": len({r["pmcid"] for r in deduped}),
        "dropped": dict(drops),
        "by_split": dict(by_split),
        "articles_by_split": {k: len(v) for k, v in arts_by_split.items()},
        "split_leakage_articles": len(overlap),
        "by_tier": dict(Counter(r.get("tier") for r in deduped)),
        "by_license": dict(Counter(r.get("license_class") for r in deduped)),
        "compound_rate": round(
            sum(1 for r in deduped if str(r.get("compound_figure")).lower() == "true")
            / max(len(deduped), 1), 3),
        "download_success_rate": round(len(have) / max(len(manifest), 1), 3),
        "filters": {"tiers": args.tiers, "licenses": args.licenses,
                    "min_quality": args.min_quality,
                    "drop_compound": args.drop_compound},
    })
    config.STATS.write_text(json.dumps(stats, indent=2))

    # -- report -----------------------------------------------------------
    print(f"manifest rows        {stats['manifest_rows']:,} "
          f"({stats['manifest_articles']:,} articles)")
    print(f"images downloaded    {stats['downloaded']:,} "
          f"({stats['download_success_rate']:.1%} of manifest)")
    print("\ndropped:")
    for k, v in drops.most_common():
        print(f"  {k:<16} {v:,}")
    print(f"\nKEPT {stats['kept']:,} pairs from {stats['articles']:,} articles")
    print("\nsplit (by paper):")
    for s in ("train", "val", "test"):
        print(f"  {s:<6} {by_split.get(s,0):>7,} pairs   "
              f"{len(arts_by_split.get(s,())):>6,} articles")
    print(f"\nleakage check: {stats['split_leakage_articles']} articles in >1 split "
          f"({'OK' if not overlap else 'BUG'})")
    print(f"tiers    {stats['by_tier']}")
    print(f"licenses {stats['by_license']}")
    print(f"compound {stats['compound_rate']:.1%}")
    print(f"\nwrote {config.PAIRS}")
    print(f"wrote {config.STATS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
