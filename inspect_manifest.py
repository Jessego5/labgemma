"""
Inspect the PMC gel/blot manifest — automated eyeballing.

Instead of squinting at a CSV, this summarizes the manifest so you can judge
quality at a glance: how many figures, license breakdown, how many have images,
keyword hit distribution, caption-length sanity, and a sample of captions to
read. It also flags likely-junk rows (empty captions, no image, suspicious
matches) so you can decide what to keep.

Run:   python inspect_manifest.py
In:    pmc_gel_manifest.csv   (produced by pmc_gel_extractor.py)
Deps:  none (stdlib only)
"""

import csv
import re
from collections import Counter

MANIFEST = "pmc_gel_manifest.csv"

# Same keywords the extractor used, to see which ones are doing the matching.
LAB_KEYWORDS = [
    "western blot", "immunoblot", "blot", "sds-page", "sds page",
    "gel electrophoresis", "agarose", "electrophoresis", "lane", "kda",
    "coomassie", "loading control",
]

# Words that hint a caption might be a FALSE positive (matched a keyword but
# isn't really a standalone gel/blot image — e.g. a schematic or a graph).
SUSPICIOUS = ["schematic", "diagram", "flowchart", "workflow", "graph",
              "scatter", "survival curve", "kaplan", "flow cytometry"]


def classify_license(lic):
    """Bucket the raw license string into a reuse tier."""
    l = (lic or "").lower()
    if "cc0" in l or "cc-0" in l or "public domain" in l:
        return "CC0 (most permissive)"
    if "by-nc" in l or "by nc" in l or "noncommercial" in l or "non-commercial" in l:
        return "CC BY-NC (non-commercial)"
    if "by-sa" in l or "by-nd" in l:
        return "CC BY-SA/ND (attribution+)"
    if "cc by" in l or "cc-by" in l or "creativecommons.org/licenses/by" in l:
        return "CC BY (permissive)"
    if l.strip() == "":
        return "(none captured)"
    return f"other/unclear"


def main():
    try:
        with open(MANIFEST, newline="") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"Can't find {MANIFEST}. Run pmc_gel_extractor.py first.")
        return

    n = len(rows)
    print("=" * 68)
    print(f"MANIFEST INSPECTION: {MANIFEST}  ({n} rows)")
    print("=" * 68)
    if n == 0:
        print("Empty manifest — the search/filter returned nothing. Check the")
        print("SEARCH_TERM and keyword filter in the extractor.")
        return

    # ---- 1. license breakdown (which tier can you legally use?) ----
    lic_counts = Counter(classify_license(r.get("license", "")) for r in rows)
    print("\n[1] LICENSE BREAKDOWN (what you can safely reuse)")
    for tier, c in lic_counts.most_common():
        print(f"    {c:4d}  {tier}")
    permissive = sum(c for t, c in lic_counts.items()
                     if t.startswith("CC0") or t.startswith("CC BY (") )
    print(f"    -> {permissive}/{n} are clean CC BY / CC0 (safest to keep)")

    # ---- 2. image availability ----
    with_img = sum(1 for r in rows if r.get("image_href", "").strip())
    print(f"\n[2] IMAGE REFERENCES")
    print(f"    {with_img}/{n} rows have an image_href (needed for step 2 download)")
    if with_img < n:
        print(f"    {n - with_img} have NO image reference -> likely unusable, drop them")

    # ---- 3. which keywords are matching (is the filter sensible?) ----
    kw_counter = Counter()
    for r in rows:
        c = (r.get("caption", "") or "").lower()
        for kw in LAB_KEYWORDS:
            if kw in c:
                kw_counter[kw] += 1
    print(f"\n[3] KEYWORD HITS (which terms are pulling figures in)")
    for kw, c in kw_counter.most_common():
        print(f"    {c:4d}  '{kw}'")
    print("    -> if a vague keyword like 'lane' or 'blot' dominates with weird")
    print("       captions, consider tightening the filter.")

    # ---- 4. caption quality: length + suspicious flags ----
    lengths = [len((r.get("caption") or "").split()) for r in rows]
    empty = sum(1 for L in lengths if L == 0)
    veryshort = sum(1 for L in lengths if 0 < L < 5)
    avg = sum(lengths) / n if n else 0
    suspicious_rows = [r for r in rows
                       if any(s in (r.get("caption", "") or "").lower() for s in SUSPICIOUS)]
    print(f"\n[4] CAPTION QUALITY")
    print(f"    avg caption length: {avg:.0f} words")
    print(f"    empty captions:     {empty}  (drop these)")
    print(f"    very short (<5 wd):  {veryshort}  (probably low-value)")
    print(f"    suspicious matches:  {len(suspicious_rows)}  (caption mentions "
          f"schematic/graph/etc — may be false positives)")

    # ---- 5. read a sample of captions ----
    print(f"\n[5] SAMPLE CAPTIONS (read these — are they really gels/blots?)")
    step = max(1, n // 8)
    for r in rows[::step][:8]:
        cap = (r.get("caption", "") or "")[:150]
        print(f"    PMC{r.get('pmcid','?')} {r.get('label','')}: {cap}")

    # ---- 6. suggested clean subset ----
    clean = [r for r in rows
             if r.get("image_href", "").strip()
             and len((r.get("caption") or "").split()) >= 5
             and not any(s in (r.get("caption", "") or "").lower() for s in SUSPICIOUS)]
    permissive_clean = [r for r in clean
                        if classify_license(r.get("license", "")).startswith(("CC0", "CC BY ("))]
    print(f"\n[6] SUGGESTED KEEP SET")
    print(f"    {len(clean)}/{n} rows pass basic quality (has image, real caption,")
    print(f"        not obviously a schematic/graph)")
    print(f"    {len(permissive_clean)}/{n} ALSO have permissive CC BY/CC0 license")
    print(f"    ^ that permissive+clean set is your safest dataset to build from.")
    print("\nNext: decide your keep-rules, then step-2 script downloads the images")
    print("for the kept rows and pairs each with its caption.")


if __name__ == "__main__":
    main()