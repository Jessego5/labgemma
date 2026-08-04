"""
Stage 2: download figure images from the PMC Open Access S3 bucket.

  manifest.csv  ->  parallel S3 fetch  ->  DATA_ROOT/images/*  +  download_log.csv

This is the stage that silently lost 60% of the first dataset. Three fixes:

  * IDEMPOTENT. Every image is skipped if it already exists on disk, so
    re-running after any interruption tops up the missing files instead of
    redoing the whole run. Resume is the default, not a flag.

  * INTERRUPT-SAFE. The original caught `except Exception`, which does NOT
    catch KeyboardInterrupt -- a Ctrl-C tore out of the loop and skipped the
    CSV writes entirely, leaving images on disk with no record of them. The
    log is now appended and flushed per image, and SIGINT is handled
    explicitly.

  * LOUD ABOUT PARTIAL RESULTS. Prints the download success rate and exits
    non-zero if any row failed, so a truncated run cannot be mistaken for a
    complete one by whatever runs next.

Speed: images are fetched with a thread pool over a shared connection pool.
S3 is not rate limited by NCBI and the transfer is pure latency, so this is
close to a linear speedup -- the single biggest lever on a slow connection.

Run:
    python -m pipeline.download
    python -m pipeline.download --workers 32
"""

import argparse
import csv
import hashlib
import os
import signal
import sys
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from . import config

S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".webp")

LOG_FIELDS = ["pmcid", "figure_id", "status", "s3_key", "image_path", "bytes", "sha256"]

_stop = threading.Event()
_session_local = threading.local()


def _session() -> requests.Session:
    """One pooled Session per worker thread; connection reuse matters here."""
    s = getattr(_session_local, "s", None)
    if s is None:
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8)
        s.mount("https://", adapter)
        _session_local.s = s
    return s


# --------------------------------------------------------------------------
# S3 key resolution
# --------------------------------------------------------------------------


def list_article_keys(pmcid: str) -> list:
    """List every object key for an article by the 'PMC{id}.' prefix."""
    url = f"{config.S3_BASE}/?list-type=2&prefix=PMC{pmcid}."
    r = _session().get(url, timeout=60)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    return [c.find(f"{S3_NS}Key").text for c in root.findall(f"{S3_NS}Contents")]


def pick_image_key(keys, image_href: str):
    """Match a manifest image_href to a real S3 key."""
    target = os.path.basename(image_href).lower()
    stem = os.path.splitext(target)[0]
    for k in keys:                                    # exact filename
        if os.path.basename(k).lower() == target:
            return k
    for k in keys:                                    # stem, any extension
        kb = os.path.basename(k).lower()
        if kb.endswith(IMAGE_EXTS) and os.path.splitext(kb)[0] == stem:
            return k
    for k in keys:                                    # loose contains
        kb = os.path.basename(k).lower()
        if kb.endswith(IMAGE_EXTS) and stem and stem in kb:
            return k
    return None


def resolve_key(pmcid: str, href: str, s3_prefix: str,
                key_cache: dict, lock: threading.Lock):
    """
    Find the S3 key for one figure.

    Fast path: the manifest carries the article's exact versioned folder
    (from JATS `pmcid-ver`, e.g. PMC3219872.2), so the key is known outright
    and costs one GET. Listing the bucket costs ~1.5s per article, and
    guessing version .1 is wrong for every article revised after publication.

    Falls back to a HEAD on .1, then a full listing, for older manifests or
    articles whose file was renamed in the OA package.
    """
    base = os.path.basename(href)
    if base and s3_prefix:
        return f"{s3_prefix}/{base}"

    if base:
        guess = f"PMC{pmcid}.1/{base}"
        try:
            r = _session().head(f"{config.S3_BASE}/{guess}", timeout=30)
            if r.status_code == 200:
                return guess
        except requests.RequestException:
            pass

    with lock:
        keys = key_cache.get(pmcid)
    if keys is None:
        try:
            keys = list_article_keys(pmcid)
        except Exception:
            keys = []
        with lock:
            key_cache[pmcid] = keys
    return pick_image_key(keys, href) if keys else None


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------


def fetch_one(row, key_cache, lock) -> dict:
    """Resolve, download, and hash one figure image. Never raises."""
    pmcid, fig_id = row["pmcid"], row["figure_id"]
    out = {"pmcid": pmcid, "figure_id": fig_id, "status": "", "s3_key": "",
           "image_path": "", "bytes": 0, "sha256": ""}
    if _stop.is_set():
        out["status"] = "aborted"
        return out

    # Skip if we already have it -- this is what makes re-runs cheap.
    for ext in IMAGE_EXTS:
        p = config.IMAGE_DIR / f"PMC{pmcid}_{fig_id}{ext}"
        if p.exists() and p.stat().st_size > 0:
            out.update(status="cached", image_path=p.name, bytes=p.stat().st_size)
            return out

    try:
        prefix = row.get("s3_prefix", "")
        key = resolve_key(pmcid, row["image_href"], prefix, key_cache, lock)
        if not key:
            out["status"] = "no_key"
            return out

        r = _session().get(f"{config.S3_BASE}/{key}", timeout=120)
        if r.status_code == 404 and prefix:
            # The versioned fast path missed (renamed file in the OA package).
            # Pay for a listing once and retry properly.
            key = resolve_key(pmcid, row["image_href"], "", key_cache, lock)
            if not key:
                out["status"] = "no_key"
                return out
            r = _session().get(f"{config.S3_BASE}/{key}", timeout=120)

        out["s3_key"] = key
        ext = os.path.splitext(key)[1].lower() or ".jpg"
        dest = config.IMAGE_DIR / f"PMC{pmcid}_{fig_id}{ext}"

        if r.status_code != 200:
            out["status"] = f"http_{r.status_code}"
            return out
        data = r.content
        if not data:
            out["status"] = "empty"
            return out

        # Write to a temp name then rename, so an interrupted write never
        # leaves a truncated file that the skip-if-exists check would trust.
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(dest)

        out.update(status="ok", image_path=dest.name, bytes=len(data),
                   sha256=hashlib.sha256(data).hexdigest())
        return out
    except Exception as e:
        out["status"] = f"error:{type(e).__name__}"
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=config.S3_WORKERS)
    ap.add_argument("--manifest", default=str(config.MANIFEST))
    ap.add_argument("--tiers", nargs="*", default=None,
                    help="only download these tiers (default: all)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    config.ensure_dirs()
    if not os.path.exists(args.manifest):
        print(f"No manifest at {args.manifest}. Run: python -m pipeline.extract")
        return 2

    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.tiers:
        rows = [r for r in rows if r.get("tier") in args.tiers]
    if args.limit:
        rows = rows[:args.limit]

    n_articles = len({r["pmcid"] for r in rows})
    print(f"{len(rows):,} figures across {n_articles:,} articles "
          f"-> {config.IMAGE_DIR}")
    print(f"{args.workers} workers\n")

    def _sigint(signum, frame):
        if not _stop.is_set():
            print("\n!! Interrupt received -- finishing in-flight downloads and "
                  "writing the log. Re-run to resume.\n")
        _stop.set()
    signal.signal(signal.SIGINT, _sigint)

    key_cache, lock = {}, threading.Lock()
    counts = {}
    t0 = time.time()
    total_bytes = 0

    # The log is opened in append mode and flushed per result, so whatever
    # completed before an interrupt is always on disk.
    write_header = not config.DOWNLOAD_LOG.exists()
    with open(config.DOWNLOAD_LOG, "a", newline="", encoding="utf-8") as lf:
        w = csv.DictWriter(lf, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_one, r, key_cache, lock): r for r in rows}
            for i, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                counts[res["status"]] = counts.get(res["status"], 0) + 1
                total_bytes += res["bytes"] or 0
                w.writerow(res)
                lf.flush()
                if i % 50 == 0 or i == len(rows):
                    rate = i / max(time.time() - t0, 1e-6)
                    eta = (len(rows) - i) / max(rate, 1e-6)
                    print(f"  [{i:,}/{len(rows):,}] {rate:.1f} img/s "
                          f"| {total_bytes/1e6:.0f} MB | ETA {eta/60:.1f} min")

    dt = time.time() - t0
    got = counts.get("ok", 0) + counts.get("cached", 0)
    rate = got / max(len(rows), 1)

    print(f"\nDONE in {dt/60:.1f} min")
    for k in sorted(counts):
        print(f"  {k:<16} {counts[k]:,}")
    print(f"  {'-'*24}")
    print(f"  download success rate: {rate:.1%}  ({got:,}/{len(rows):,})")
    print(f"  bytes on disk        : {total_bytes/1e9:.2f} GB")
    print(f"  log                  : {config.DOWNLOAD_LOG}")

    if _stop.is_set():
        print("\nRun was INTERRUPTED. Re-run the same command to resume.")
        return 130
    if rate < 1.0:
        print(f"\n!! {len(rows)-got:,} figures did not download. This is the "
              f"failure that silently truncated the first dataset.\n"
              f"   Re-run to retry them; inspect {config.DOWNLOAD_LOG} for the "
              f"status breakdown before building the dataset.")
        return 1
    print(f"\nNext:  python -m pipeline.dataset")
    return 0


if __name__ == "__main__":
    sys.exit(main())
