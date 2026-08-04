"""
Stage 1: build the figure manifest from PubMed Central.

  search (date-partitioned)  ->  batched efetch  ->  parse JATS  ->  score
                                                                      |
                                                              manifest.csv

Design notes, all of them lessons from the first version of this script:

  * RESUMABLE. Article XML is cached gzipped under DATA_ROOT/xml_cache, and
    the manifest is appended incrementally with a flush after every batch.
    Kill the process at any point and re-run: it skips what it already has.
    A partially-completed run is a normal state, not a corrupted one.

  * BATCHED. 50 PMCIDs per efetch call instead of one, which measured ~15x
    faster and drops the request count far below NCBI's rate limit.

  * PER-ARTICLE LICENSE. The previous version took the first <license> found
    in the document and applied it to every figure. In a batched response that
    would stamp article #1's license onto all 50 articles. Licenses and PMCIDs
    are now read from each <article> element individually.

Run:
    python -m pipeline.extract --max-articles 3000
    python -m pipeline.extract --max-articles 3000 --resume    # continue a run
"""

import argparse
import csv
import gzip
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config, scoring
from .ncbi import NCBIClient, article_s3_prefix, split_articleset

FIELDS = [
    "pmcid", "figure_id", "label", "image_href", "s3_prefix", "caption",
    "license_url", "license", "license_class", "keyword_score", "gel_score",
    "caption_quality", "compound_figure", "tier", "n_figs_in_article",
]

XLINK = "{http://www.w3.org/1999/xlink}href"
ALI_LICENSE_REF = "{http://www.niso.org/schemas/ali/1.0/}license_ref"


# --------------------------------------------------------------------------
# JATS parsing
# --------------------------------------------------------------------------


def text_of(elem) -> str:
    """Flatten an element's text content; captions contain nested markup."""
    if elem is None:
        return ""
    return re.sub(r"\s+", " ", "".join(elem.itertext())).strip()


def article_license(art):
    """
    Return (license_url, license_text) for this article.

    The machine-readable license lives in a NISO ALI <ali:license_ref> element,
    NOT in the license element's text or attributes:

        <license>
          <ali:license_ref content-type="ccbylicense">
            https://creativecommons.org/licenses/by/4.0/
          </ali:license_ref>
          <license-p>This is an open access article...</license-p>
        </license>

    Classifying on the prose instead of the URL is unreliable, because many
    PMC articles carry a publisher free-access notice (Elsevier's COVID-19
    statement is the common one) whose wording resembles a license but grants
    no CC rights at all. The URL is authoritative; the prose is a fallback.
    """
    url, text, ltype = "", "", ""
    # license_ref can sit under <license> or directly under <permissions>.
    for ref in art.iter(ALI_LICENSE_REF):
        u = (ref.text or "").strip()
        if u:
            url = u
            break
    for lic in art.iter("license"):
        if not url:
            url = (lic.get(XLINK) or "").strip()
        if not ltype:
            ltype = (lic.get("license-type") or "").strip()
        if not text:
            text = text_of(lic)
    parts = [p for p in (ltype, text) if p]
    return url, " | ".join(parts)[:400]


def classify_license(url: str, text: str = "") -> str:
    """
    Bucket into a reuse tier, trusting the license URL over the prose.

    'PUBLISHER-FREE' and 'PMC-OA-ONLY' are free to READ but carry no CC grant.
    They are kept and labeled rather than dropped, so you can decide per
    experiment -- but they should not be redistributed as a CC dataset.
    """
    u = (url or "").lower()
    if u:
        if "publicdomain" in u or "/zero/" in u or "cc0" in u:
            return "CC0"
        if "/by-nc-nd" in u:
            return "CC BY-NC-ND"
        if "/by-nc-sa" in u:
            return "CC BY-NC-SA"
        if "/by-nc" in u:
            return "CC BY-NC"
        if "/by-sa" in u:
            return "CC BY-SA"
        if "/by-nd" in u:
            return "CC BY-ND"
        if "/by" in u:
            return "CC BY"

    t = (text or "").lower()
    if "covid-19 resource cent" in t or "covid-19 public health emergency" in t:
        return "PUBLISHER-FREE"
    if "pmc open access subset" in t:
        return "PMC-OA-ONLY"
    if "creative commons" in t:
        # Prose names CC but no URL: record it, don't guess the flavour.
        return "CC-UNSPECIFIED"
    if not t.strip():
        return "UNKNOWN"
    return "OTHER"


def parse_article(pmcid: str, art_xml: str) -> list:
    """Extract every <fig> from one article, scored and license-stamped."""
    try:
        art = ET.fromstring(art_xml)
    except ET.ParseError:
        return []

    lic_url, lic_text = article_license(art)
    lic_class = classify_license(lic_url, lic_text)
    s3_prefix = article_s3_prefix(art)
    figs = list(art.iter("fig"))
    rows = []
    for fig in figs:
        caption = text_of(fig.find("caption"))
        if not scoring.is_candidate(caption):
            continue
        href = ""
        for g in fig.iter("graphic"):
            href = g.get(XLINK) or g.get("href") or ""
            if href:
                break
        if not href:
            continue
        row = {
            "pmcid": pmcid,
            "figure_id": fig.get("id", ""),
            "label": text_of(fig.find("label")),
            "image_href": href,
            "s3_prefix": s3_prefix,
            "caption": caption,
            "license_url": lic_url,
            "license": lic_text,
            "license_class": lic_class,
            "n_figs_in_article": len(figs),
        }
        row.update(scoring.score_all(caption))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# XML cache
# --------------------------------------------------------------------------


def cache_path(pmcid: str):
    return config.XML_CACHE / f"PMC{pmcid}.xml.gz"


def cache_write(pmcid: str, xml: str) -> None:
    """
    Write atomically: a kill mid-write would otherwise leave a truncated .gz
    that cache_read treats as a miss forever. With concurrent workers the
    window for that is wider, so write to a temp name and rename.
    """
    p = cache_path(pmcid)
    tmp = p.with_suffix(p.suffix + ".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        f.write(xml)
    tmp.replace(p)


def cache_read(pmcid: str):
    p = cache_path(pmcid)
    if not p.exists():
        return None
    try:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return f.read()
    except (OSError, EOFError):
        return None      # truncated by an interrupted write; refetch


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def collect_pmcids(client: NCBIClient, limit: int, verbose=True) -> list:
    """
    Enumerate PMCIDs, date-partitioning around the 9998 esearch cap.

    Sampling is spread PROPORTIONALLY across every date slice rather than
    filling greedily from the first one. Filling greedily draws an entire
    small run from the oldest slice, which biases the dataset hard: older
    articles are lower-resolution scans and are dominated by publisher
    free-access notices rather than CC licenses. Proportional allocation
    keeps the sample representative of the whole search pool.

    When a limit is set, the range is partitioned more finely than the API
    cap requires, so each slice contributes a small share and the sample is
    spread evenly over time instead of clustering at slice boundaries.
    """
    total = client.count(config.SEARCH_TERM)
    if verbose:
        print(f"Search matches {total:,} articles in the PMC OA subset.")

    if total <= config.ESEARCH_PAGE_LIMIT and not limit:
        return client.search_all(config.SEARCH_TERM)

    # Finer slices when sampling, so the draw is spread across the timeline.
    cap = config.ESEARCH_PAGE_LIMIT
    if limit:
        cap = max(500, min(cap, total // config.SAMPLE_SLICES))
    if verbose:
        print(f"Partitioning by date (slice cap {cap:,})...")
    slices = client.partition_query(config.SEARCH_TERM, config.DATE_FROM,
                                    config.DATE_TO, cap=cap)
    grand = sum(n for _, n in slices) or 1
    if verbose:
        print(f"  {len(slices)} date slices covering {grand:,} articles")

    # Proportional quota per slice, at least 1 each so no era is dropped.
    quotas = []
    for term, n in slices:
        q = max(1, round(limit * n / grand)) if limit else None
        quotas.append((term, n, q))

    out, seen = [], set()
    for i, (term, n, q) in enumerate(quotas, 1):
        for pmcid in client.search_all(term, limit=q):
            if pmcid not in seen:
                seen.add(pmcid)
                out.append(pmcid)
        if verbose and (i % 10 == 0 or i == len(quotas)):
            print(f"  [{i}/{len(quotas)}] {len(out):,} unique ids")

    # Rounding and duplicates can leave us short; top up from the largest
    # slices rather than returning fewer articles than asked for.
    if limit and len(out) < limit:
        for term, n, _ in sorted(quotas, key=lambda x: -x[1]):
            if len(out) >= limit:
                break
            for pmcid in client.search_all(term, limit=None):
                if pmcid not in seen:
                    seen.add(pmcid)
                    out.append(pmcid)
                    if len(out) >= limit:
                        break

    return out[:limit] if limit else out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-articles", type=int, default=3000,
                    help="how many articles to pull (default 3000)")
    ap.add_argument("--resume", action="store_true",
                    help="append to an existing manifest, skipping cached articles")
    ap.add_argument("--batch", type=int, default=config.EFETCH_BATCH)
    ap.add_argument("--workers", type=int, default=config.EXTRACT_WORKERS,
                    help="concurrent efetch batches (default "
                         f"{config.EXTRACT_WORKERS})")
    args = ap.parse_args(argv)

    config.ensure_dirs()
    print(config.describe(), "\n")

    client = NCBIClient()

    done = set()
    if args.resume and config.MANIFEST.exists():
        with open(config.MANIFEST, newline="", encoding="utf-8") as f:
            done = {r["pmcid"] for r in csv.DictReader(f)}
        print(f"Resuming: {len(done):,} articles already in manifest.\n")

    pmcids = collect_pmcids(client, args.max_articles)
    todo = [p for p in pmcids if p not in done]
    print(f"\n{len(pmcids):,} ids enumerated, {len(todo):,} still to fetch.\n")

    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]

    def fetch_and_parse(batch):
        """
        One batch: serve from cache, fetch the misses, parse to manifest rows.

        Runs in a worker thread. Does NOT touch the CSV -- rows are returned
        and written by the main thread, so the writer stays single-threaded
        and the manifest can never interleave a half-written row.
        """
        cached = {p: cache_read(p) for p in batch}
        missing = [p for p, x in cached.items() if x is None]
        if missing:
            xml = client.fetch_batch(missing)
            for pmcid, art_xml in split_articleset(xml):
                cache_write(pmcid, art_xml)
                cached[pmcid] = art_xml
        rows, n_art = [], 0
        for pmcid, art_xml in cached.items():
            if not art_xml:
                continue
            rows += parse_article(pmcid, art_xml)
            n_art += 1
        return rows, n_art, len(batch)

    mode = "a" if (args.resume and config.MANIFEST.exists()) else "w"
    t0 = time.time()
    n_rows = n_articles = n_done = n_failed = 0

    with open(config.MANIFEST, mode, newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if mode == "w":
            w.writeheader()
            fh.flush()

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_and_parse, b): b for b in batches}
            for k, fut in enumerate(as_completed(futs), 1):
                try:
                    rows, n_art, n_in = fut.result()
                except Exception as e:
                    n_failed += len(futs[fut])
                    print(f"  batch FETCH ERROR {type(e).__name__}: {e}")
                    continue
                w.writerows(rows)
                fh.flush()          # survive a kill
                n_rows += len(rows)
                n_articles += n_art
                n_done += n_in
                if k % 10 == 0 or n_done >= len(todo):
                    rate = n_done / max(time.time() - t0, 1e-6)
                    eta = (len(todo) - n_done) / max(rate, 1e-6)
                    print(f"  [{n_done:,}/{len(todo):,}] {n_rows:,} figures kept "
                          f"| {rate:.1f} art/s | ETA {eta/60:.1f} min")

    if n_failed:
        print(f"\n!! {n_failed:,} articles failed to fetch. Re-run with "
              f"--resume to retry them (cached articles are skipped).")

    dt = time.time() - t0
    print(f"\nDONE in {dt/60:.1f} min")
    print(f"  articles parsed : {n_articles:,}")
    print(f"  figures kept    : {n_rows:,}  ({n_rows/max(n_articles,1):.2f} per article)")
    print(f"  NCBI requests   : {client.n_requests:,} ({client.n_retries} retries)")
    print(f"  manifest        : {config.MANIFEST}")
    print(f"\nNext:  python -m pipeline.download")
    return 0


if __name__ == "__main__":
    sys.exit(main())
